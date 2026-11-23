from __future__ import annotations

import ast
import json
from pathlib import Path

import mido
import numpy as np
import pretty_midi
import pytest

from instrument_agnostic_amt.beat_chord.tempo_map_export import MeterSegmentSpec, export_tempo_mapped_midi
from instrument_agnostic_amt.instrument_refinement.inference.render_compare import RenderCompareConfig
from instrument_agnostic_amt.quantization.render_compare_defragment import (
    RenderCompareDefragmentConfig,
    _BoundaryCandidate,
    _build_active_boundary_windows,
    _crop_instruments_to_packed_windows,
)
from instrument_agnostic_amt.quantization.render_compare_quantize import (
    DEFAULT_QUANTIZE_CANDIDATES,
    RenderCompareQuantizeConfig,
    _select_adaptive_candidates_for_stem,
)
from instrument_agnostic_amt.soundfont import (
    FluidSynthRenderOptions,
    clear_resident_fluidsynth_pool,
    default_parallel_render_workers,
    is_libfluidsynth_available,
    render_midi_batch_to_stereo_arrays,
    render_midi_to_stereo_array,
    resolve_signal_processing_device,
    resolve_soundfont_path,
    write_packed_smf_type0,
)
from instrument_agnostic_amt.velocity.render_compare_velocity import RenderCompareVelocityConfig


def test_resolve_soundfont_path_fallback_download_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    monkeypatch.delenv("TSUMUGI_SOUNDFONT_PATH", raising=False)
    monkeypatch.delenv("TSUMUGI_SOUNDFONT_DIR", raising=False)
    attempted_urls: list[str] = []

    class _FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._buffer = io.BytesIO(payload)
            self.headers = {"Content-Length": str(len(payload))}

        def read(self, size: int = -1) -> bytes:
            return self._buffer.read(size)

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_urlopen(request: object, timeout: int = 60, context: object = None) -> _FakeResponse:
        url = request.full_url if hasattr(request, "full_url") else str(request)
        attempted_urls.append(url)
        if "primary-404" in url:
            raise RuntimeError("HTTP 404 Not Found")
        if "tiny-broken" in url:
            return _FakeResponse(b"<html>404</html>")
        return _FakeResponse(b"RIFF" + b"\x00" * 1_100_000)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    resolved = resolve_soundfont_path(
        None,
        cache_directory=tmp_path / "sf_cache",
        filename="Test_General.sf2",
        urls=(
            "https://example.invalid/primary-404.sf2",
            "https://example.invalid/tiny-broken.sf2",
            "https://example.invalid/valid.sf2",
        ),
    )
    assert resolved.is_file()
    assert resolved.name == "Test_General.sf2"
    assert not resolved.with_name("Test_General.sf2.part").exists()
    assert len(attempted_urls) == 3

    # 2 回目はキャッシュから即座に返り、再ダウンロードしないこと。
    resolved_again = resolve_soundfont_path(
        None,
        cache_directory=tmp_path / "sf_cache",
        filename="Test_General.sf2",
        urls=("https://example.invalid/should-not-be-called.sf2",),
    )
    assert resolved_again == resolved
    assert len(attempted_urls) == 3


def test_resolve_soundfont_path_prefers_explicit_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit_sf2 = tmp_path / "explicit.sf2"
    env_sf2 = tmp_path / "env.sf2"
    explicit_sf2.write_bytes(b"RIFF" + b"\x00" * 16)
    env_sf2.write_bytes(b"RIFF" + b"\x00" * 16)

    monkeypatch.setenv("TSUMUGI_SOUNDFONT_PATH", str(env_sf2))
    assert resolve_soundfont_path(explicit_sf2) == explicit_sf2.resolve()
    assert resolve_soundfont_path("") == env_sf2.resolve()
    assert resolve_soundfont_path("auto") == env_sf2.resolve()


def test_default_parallel_render_workers_caps_large_soundfont(tmp_path: Path) -> None:
    small_sf2 = tmp_path / "small.sf2"
    small_sf2.write_bytes(b"\x00" * 1024)
    assert default_parallel_render_workers(small_sf2, job_count=6, max_workers=4) == 4
    assert default_parallel_render_workers(small_sf2, job_count=1, max_workers=4) == 1


def test_speed_presets_construct_valid_configs(tmp_path: Path) -> None:
    fake_sf2 = tmp_path / "dummy.sf2"
    fake_sf2.write_bytes(b"\x00" * 64)
    for speed in ("fast", "balanced", "quality"):
        ref_cfg = RenderCompareConfig.for_speed_preset(fake_sf2, speed=speed)
        defrag_cfg = RenderCompareDefragmentConfig.for_speed_preset(fake_sf2, speed=speed)
        quant_cfg = RenderCompareQuantizeConfig.for_speed_preset(fake_sf2, speed=speed)
        vel_cfg = RenderCompareVelocityConfig.for_speed_preset(fake_sf2, speed=speed)
        assert ref_cfg.soundfont_path == fake_sf2
        assert defrag_cfg.soundfont_path == fake_sf2
        assert quant_cfg.soundfont_path == fake_sf2
        assert vel_cfg.soundfont_path == fake_sf2

    balanced_quant = RenderCompareQuantizeConfig.for_speed_preset(fake_sf2, speed="balanced")
    assert balanced_quant.adaptive_candidates is False
    assert balanced_quant.verify_full_mix is False

    balanced_vel = RenderCompareVelocityConfig.for_speed_preset(fake_sf2, speed="balanced")
    assert balanced_vel.sample_rate == 16_000
    assert balanced_vel.disable_reverb is False
    assert balanced_vel.disable_chorus is False
    assert balanced_vel.evaluate_final is False
    assert balanced_vel.num_iterations == 3


def test_adaptive_quantize_candidates_skip_unneeded_subdivisions() -> None:
    beat_times = np.arange(0.0, 8.5, 0.5, dtype=np.float64)  # 120 BPM

    # 16 分音符主体のストレート曲では triplet_12th / fine_32nd が自動省略される。
    straight_midi = pretty_midi.PrettyMIDI()
    straight_inst = pretty_midi.Instrument(program=0)
    for step in range(32):
        onset = step * 0.125
        straight_inst.notes.append(pretty_midi.Note(velocity=96, pitch=60, start=onset, end=onset + 0.10))
    straight_midi.instruments.append(straight_inst)
    straight_names = [
        candidate.name
        for candidate in _select_adaptive_candidates_for_stem(straight_midi, beat_times, DEFAULT_QUANTIZE_CANDIDATES)
    ]
    assert straight_names == ["original", "soft_16th", "tight_16th"]

    # 3 連符曲では triplet_12th が自動で含まれる。
    triplet_midi = pretty_midi.PrettyMIDI()
    triplet_inst = pretty_midi.Instrument(program=0)
    for beat_index in range(12):
        for sub in range(3):
            onset = beat_index * 0.5 + sub * (0.5 / 3.0)
            triplet_inst.notes.append(pretty_midi.Note(velocity=96, pitch=64, start=onset, end=onset + 0.12))
    triplet_midi.instruments.append(triplet_inst)
    triplet_names = [
        candidate.name
        for candidate in _select_adaptive_candidates_for_stem(triplet_midi, beat_times, DEFAULT_QUANTIZE_CANDIDATES)
    ]
    assert "triplet_12th" in triplet_names


def test_defragment_active_windows_compress_long_track() -> None:
    candidates = [
        _BoundaryCandidate(
            stem_name="other",
            instrument_index=0,
            pitch=60,
            left_index_in_pitch=0,
            split_time_seconds=20.0,
            gap_seconds=0.02,
            left_duration_seconds=0.5,
            right_duration_seconds=0.5,
            is_sustained=True,
            is_overlapping_or_contained=False,
            reattack_rise_db=1.0,
            valley_dip_db=1.0,
            net_jump_db=0.5,
        ),
        _BoundaryCandidate(
            stem_name="other",
            instrument_index=0,
            pitch=60,
            left_index_in_pitch=2,
            split_time_seconds=180.0,
            gap_seconds=0.02,
            left_duration_seconds=0.5,
            right_duration_seconds=0.5,
            is_sustained=True,
            is_overlapping_or_contained=False,
            reattack_rise_db=1.0,
            valley_dip_db=1.0,
            net_jump_db=0.5,
        ),
    ]
    windows, packed_duration = _build_active_boundary_windows(
        candidates,
        stem_duration_seconds=240.0,
        margin_seconds=0.65,
    )
    assert len(windows) == 2
    assert packed_duration < 5.0

    instrument = pretty_midi.Instrument(program=48, name="Strings")
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=19.6, end=20.0))
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=20.02, end=20.5))
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=90.0, end=91.0))  # 窓外
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=179.5, end=180.4))
    cropped = _crop_instruments_to_packed_windows([instrument], windows)
    assert len(cropped) == 1
    assert len(cropped[0].notes) == 3  # 窓外の 90s ノートは除外される


def test_tempo_map_export_preserves_cc7_cc10_and_lcr_tracks(tmp_path: Path) -> None:
    source_midi_path = tmp_path / "velocity_lcr.mid"
    output_midi_path = tmp_path / "tempo_mapped.mid"

    source_midi = pretty_midi.PrettyMIDI()
    for name, pan, vol in (("Distorted Guitar (L)", 18, 92), ("Distorted Guitar (R)", 110, 95)):
        instrument = pretty_midi.Instrument(program=30, name=name)
        instrument.control_changes.append(pretty_midi.ControlChange(number=7, value=vol, time=0.0))
        instrument.control_changes.append(pretty_midi.ControlChange(number=10, value=pan, time=0.0))
        instrument.notes.append(pretty_midi.Note(velocity=105, pitch=52, start=0.5, end=1.5))
        source_midi.instruments.append(instrument)
    source_midi.write(str(source_midi_path))

    export_tempo_mapped_midi(
        source_midi_path=source_midi_path,
        output_midi_path=output_midi_path,
        beat_times=(0.0, 0.5, 1.0, 1.5, 2.0),
        meter_segments=(MeterSegmentSpec(start_seconds=0.0, end_seconds=2.0, numerator=4, denominator=4, bar_count=1),),
        chord_segments=(),
        duration_seconds=2.5,
    )

    reloaded = mido.MidiFile(str(output_midi_path))
    track_names = [
        message.name
        for track in reloaded.tracks
        for message in track
        if message.type == "track_name"
    ]
    cc7_values = sorted(
        message.value
        for track in reloaded.tracks
        for message in track
        if message.type == "control_change" and message.control == 7
    )
    cc10_values = sorted(
        message.value
        for track in reloaded.tracks
        for message in track
        if message.type == "control_change" and message.control == 10
    )
    assert "Distorted Guitar (L)" in track_names
    assert "Distorted Guitar (R)" in track_names
    assert cc7_values == [92, 95]
    assert cc10_values == [18, 110]


def test_colab_notebook_json_syntax_and_release_options() -> None:
    notebook_path = Path("Colab_Inference.ipynb")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    run_cell = next(cell for cell in payload["cells"] if cell.get("metadata", {}).get("id") == "stem-sep-run")
    source_text = "".join(run_cell["source"])

    # Python 構文エラーがないことを AST パースで検証する。
    ast.parse(source_text)

    # 個別 R&C オプション・速度プリセット・SF2 任意指定・FluidSynth 自動導入が揃っていること。
    for required_token in (
        "RENDER_COMPARE_INSTRUMENTS = True",
        "RENDER_COMPARE_DEFRAGMENT = True",
        "RENDER_COMPARE_QUANTIZE = True",
        "RENDER_COMPARE_VELOCITY = True",
        "RENDER_COMPARE_SPEED = \"balanced\"",
        "SOUNDFONT_PATH = \"\"",
        "fluidsynth",
        "(PREDICT_BEAT_CHORD or RENDER_COMPARE_QUANTIZE) and BEAT_SOURCE == \"sheetsage2\"",
    ):
        assert required_token in source_text

    # Colab で不要な低レベルオプションがフォームから削除されていること。
    for removed_param in (
        "STEM_SPLITTER_BATCH_SIZE =",
        "WINDOW_BATCH_SIZE =",
        "LOW_VRAM_MODE =",
        "NO_HALF =",
        "DEVICE =",
        "AMP =",
        "COMPILE_MODEL =",
        "COMPILE_VELOCITY =",
        "SEMI_CRF_BACKEND =",
    ):
        assert removed_param not in source_text


def test_write_packed_smf_type0_roundtrips_through_mido_with_multibyte_vlq(tmp_path: Path) -> None:
    midi_path = tmp_path / "packed_fast.mid"
    scheduled_events = [
        # 同一 tick で priority 順にソートされることを確認する (program_change -> cc -> note_on)。
        (100, 3, bytes([0x90 | 2, 60, 104])),
        (100, 1, bytes([0xC0 | 2, 30])),
        (100, 2, bytes([0xB0 | 2, 10, 64])),
        # 2 バイト以上 (128〜16383) および 3 バイト以上 (>16383) の VLQ デルタタイムを検証する。
        (400, 0, bytes([0x80 | 2, 60, 0])),
        (25_000, 3, bytes([0x90 | 9, 36, 110])),
        (25_240, 0, bytes([0x80 | 9, 36, 0])),
    ]
    write_packed_smf_type0(midi_path, scheduled_events, ticks_per_beat=480, tempo_microseconds=500_000)

    loaded = mido.MidiFile(str(midi_path))
    assert loaded.type == 0
    assert loaded.ticks_per_beat == 480
    assert len(loaded.tracks) == 1

    decoded: list[tuple[int, str, dict[str, int]]] = []
    current_tick = 0
    for message in loaded.tracks[0]:
        current_tick += int(message.time)
        if message.type == "set_tempo":
            decoded.append((current_tick, "set_tempo", {"tempo": int(message.tempo)}))
        elif message.type == "program_change":
            decoded.append((current_tick, "program_change", {"channel": int(message.channel), "program": int(message.program)}))
        elif message.type == "control_change":
            decoded.append((current_tick, "control_change", {"channel": int(message.channel), "control": int(message.control), "value": int(message.value)}))
        elif message.type in ("note_on", "note_off"):
            decoded.append((current_tick, message.type, {"channel": int(message.channel), "note": int(message.note), "velocity": int(message.velocity)}))

    assert decoded == [
        (0, "set_tempo", {"tempo": 500_000}),
        (100, "program_change", {"channel": 2, "program": 30}),
        (100, "control_change", {"channel": 2, "control": 10, "value": 64}),
        (100, "note_on", {"channel": 2, "note": 60, "velocity": 104}),
        (400, "note_off", {"channel": 2, "note": 60, "velocity": 0}),
        (25_000, "note_on", {"channel": 9, "note": 36, "velocity": 110}),
        (25_240, "note_off", {"channel": 9, "note": 36, "velocity": 0}),
    ]


def test_resolve_signal_processing_device_returns_valid_torch_device() -> None:
    device = resolve_signal_processing_device()
    assert device.type in ("cuda", "cpu")


def test_render_midi_to_stereo_array_cli_fallback_when_libfluidsynth_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import soundfile as sf

    fake_sf2 = tmp_path / "dummy.sf2"
    fake_sf2.write_bytes(b"RIFF" + b"\x00" * 64)
    midi_path = tmp_path / "sample.mid"
    write_packed_smf_type0(midi_path, [(0, 0, bytes([0x90, 60, 100])), (480, 0, bytes([0x80, 60, 0]))])

    monkeypatch.setenv("TSUMUGI_FLUIDSYNTH_BACKEND", "cli")
    assert is_libfluidsynth_available() is False

    def fake_cli_render(_midi: Path, wav_path: Path, *, options: FluidSynthRenderOptions) -> None:
        waveform = np.full((160, 2), 0.25, dtype=np.float32)
        sf.write(str(wav_path), waveform, options.sample_rate, subtype="PCM_16")

    def fake_cli_batch(
        render_jobs: list[tuple[Path, Path]],
        *,
        options: FluidSynthRenderOptions,
        max_workers: int | None = None,
    ) -> None:
        del max_workers
        for job_index, (_midi, wav_path) in enumerate(render_jobs):
            waveform = np.full((160, 2), 0.1 * (job_index + 1), dtype=np.float32)
            sf.write(str(wav_path), waveform, options.sample_rate, subtype="PCM_16")

    monkeypatch.setattr("instrument_agnostic_amt.soundfont.render_midi_with_fluidsynth", fake_cli_render)
    monkeypatch.setattr("instrument_agnostic_amt.soundfont.render_midi_batch_parallel", fake_cli_batch)

    options = FluidSynthRenderOptions(soundfont_path=fake_sf2, sample_rate=16_000)
    single_stereo = render_midi_to_stereo_array(midi_path, options=options)
    assert single_stereo.shape == (2, 160)
    assert np.allclose(single_stereo, 0.25, atol=1e-3)

    batch_stereos = render_midi_batch_to_stereo_arrays([midi_path, midi_path.read_bytes()], options=options, max_workers=2)
    assert len(batch_stereos) == 2
    assert batch_stereos[0].shape == (2, 160)
    assert np.allclose(batch_stereos[0], 0.1, atol=1e-3)
    assert np.allclose(batch_stereos[1], 0.2, atol=1e-3)

    clear_resident_fluidsynth_pool()
