from __future__ import annotations

import ast
import json
from pathlib import Path

import mido
import numpy as np
import pretty_midi
import pytest

from instrument_agnostic_amt.beat_chord.sheetsage import SheetSageMelodyNote, SheetSageMelodyPrediction
from instrument_agnostic_amt.beat_chord.tempo_map_export import MeterSegmentSpec, export_tempo_mapped_midi
from instrument_agnostic_amt.beat_chord.vocal_melody_fusion import (
    VocalMelodyFusionConfig,
    fuse_sheetsage_melody_with_tsumugi_vocals,
)
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

    # 個別 R&C オプション・速度プリセット・SF2 任意指定・FluidSynth/SheetSage2 0.5.2 自動導入が揃っていること。
    for required_token in (
        "RENDER_COMPARE_INSTRUMENTS = True",
        "RENDER_COMPARE_DEFRAGMENT = True",
        "RENDER_COMPARE_QUANTIZE = True",
        "RENDER_COMPARE_VELOCITY = True",
        "VOCAL_MELODY_SOURCE = \"sheetsage2\"",
        "RENDER_COMPARE_SPEED = \"balanced\"",
        "SOUNDFONT_PATH = \"\"",
        "fluidsynth",
        "importlib.util.find_spec(\"sheetsage2_refined\") is None",
        "sheetsage2_refined.melody",
        "edb9e7562be134aac92d1eadf5bfe868453d9386",
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


def _synthesize_harmonic_tone(
    sample_rate: int,
    duration_seconds: float,
    segments: list[tuple[float, float, int, float]],
) -> np.ndarray:
    """単体テスト用に指定ピッチ・区間・振幅の倍音付き波形を合成する。"""
    num_samples = int(round(duration_seconds * sample_rate))
    waveform = np.zeros(num_samples, dtype=np.float32)
    for start_seconds, end_seconds, midi_pitch, amplitude in segments:
        start_index = max(0, int(round(start_seconds * sample_rate)))
        end_index = min(num_samples, int(round(end_seconds * sample_rate)))
        if end_index <= start_index:
            continue
        times = np.arange(end_index - start_index, dtype=np.float64) / float(sample_rate)
        frequency_hz = 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))
        tone = (
            1.0 * np.sin(2.0 * np.pi * frequency_hz * times)
            + 0.4 * np.sin(2.0 * np.pi * 2.0 * frequency_hz * times)
            + 0.2 * np.sin(2.0 * np.pi * 3.0 * frequency_hz * times)
        )
        waveform[start_index:end_index] += (amplitude * tone).astype(np.float32)
    return waveform


def test_vocal_melody_fusion_rejects_all_notes_on_instrumental_song() -> None:
    sample_rate = 16_000
    duration_seconds = 3.0
    vocal_mono = np.zeros(int(duration_seconds * sample_rate), dtype=np.float32)
    mix_mono = _synthesize_harmonic_tone(sample_rate, duration_seconds, [(0.2, 2.5, 67, 0.35)])
    prediction = SheetSageMelodyPrediction(
        duration_seconds=duration_seconds,
        notes=(
            SheetSageMelodyNote(start_seconds=0.2, end_seconds=1.2, pitch=67, track=0, vocal_probability=0.9),
            SheetSageMelodyNote(start_seconds=1.3, end_seconds=2.5, pitch=69, track=1, vocal_probability=0.1),
        ),
        vocal_probabilities=np.ones(75, dtype=np.float32),
    )

    fused_midi, report = fuse_sheetsage_melody_with_tsumugi_vocals(
        prediction,
        preloaded_vocal_mono=vocal_mono,
        preloaded_mix_mono=mix_mono,
    )
    assert report["is_instrumental_song"] is True
    assert report["final_melody_notes"] == 0
    assert report["final_harmony_notes"] == 0
    assert len(fused_midi.instruments) == 0


def test_vocal_melody_fusion_gates_instruments_and_fixes_lead_harmony_swaps(tmp_path: Path) -> None:
    sample_rate = 16_000
    duration_seconds = 5.0

    # 0.2〜1.0s: 楽器ソロのみ（vocals ステムは無音、mix のみ強発音） -> 棄却されるべき
    # 1.5〜2.5s: ボーカル主旋律 C5 (72) + ハモリ E5 (76) が同時発音 -> 主旋律=72、ハモリ=76 になるべき
    # 2.8〜3.6s: ボーカル主旋律 D5 (74) 単独（SheetSage2 は track=0、tsumugi は誤って harmony に入れた） -> ハモリ重複除去
    # 4.0〜4.6s: SheetSage2 休符区間の独立歌唱 G5 (79)（tsumugi が melody 検出） -> 主旋律へ補完されるべき
    vocal_mono = _synthesize_harmonic_tone(
        sample_rate,
        duration_seconds,
        [
            (1.5, 2.5, 72, 0.30),
            (1.5, 2.5, 76, 0.22),
            (2.8, 3.6, 74, 0.30),
            (4.0, 4.6, 79, 0.28),
        ],
    )
    mix_mono = vocal_mono + _synthesize_harmonic_tone(
        sample_rate,
        duration_seconds,
        [(0.2, 1.0, 69, 0.35), (1.5, 4.6, 48, 0.15)],
    )

    prediction = SheetSageMelodyPrediction(
        duration_seconds=duration_seconds,
        notes=(
            # 楽器ソロなのに誤って track=0 になったケース
            SheetSageMelodyNote(start_seconds=0.2, end_seconds=1.0, pitch=69, track=0, vocal_probability=0.85),
            # 真の主旋律 2 音
            SheetSageMelodyNote(start_seconds=1.5, end_seconds=2.5, pitch=72, track=0, vocal_probability=0.95),
            SheetSageMelodyNote(start_seconds=2.8, end_seconds=3.6, pitch=74, track=0, vocal_probability=0.92),
        ),
        vocal_probabilities=np.ones(125, dtype=np.float32),
    )

    # tsumugi 側の誤分類シナリオを再現:
    # - melody トラックにハモリ E5 (76) が混入 + 単独歌唱時のピッチズレ F#5 (78) + 独立フレーズ G5 (79)
    # - vocal_harmony トラックに主旋律 D5 (74) が誤混入
    tsumugi_midi_path = tmp_path / "tsumugi_vocals.mid"
    tsumugi_midi = pretty_midi.PrettyMIDI()
    ts_mel = pretty_midi.Instrument(program=65, name="melody")
    ts_mel.notes.append(pretty_midi.Note(velocity=100, pitch=72, start=1.5, end=2.5))  # 主旋律重複 -> 破棄
    ts_mel.notes.append(pretty_midi.Note(velocity=100, pitch=76, start=1.5, end=2.5))  # 真のハモリが誤混入 -> ハモリへ昇格
    ts_mel.notes.append(pretty_midi.Note(velocity=100, pitch=78, start=2.8, end=3.6))  # 単独歌唱の誤ピッチ -> 音響ガードで破棄
    ts_mel.notes.append(pretty_midi.Note(velocity=100, pitch=79, start=4.0, end=4.6))  # 独立歌唱 -> 主旋律へ補完
    ts_harm = pretty_midi.Instrument(program=65, name="vocal_harmony")
    ts_harm.notes.append(pretty_midi.Note(velocity=95, pitch=74, start=2.8, end=3.6))  # 主旋律が誤混入 -> 重複破棄
    tsumugi_midi.instruments.extend([ts_mel, ts_harm])
    tsumugi_midi.write(str(tsumugi_midi_path))

    fused_midi, report = fuse_sheetsage_melody_with_tsumugi_vocals(
        prediction,
        tsumugi_vocal_midi_path=tsumugi_midi_path,
        preloaded_vocal_mono=vocal_mono,
        preloaded_mix_mono=mix_mono,
        config=VocalMelodyFusionConfig(),
    )

    by_name = {inst.name: inst.notes for inst in fused_midi.instruments}
    melody_pitches = [note.pitch for note in by_name.get("melody", [])]
    harmony_pitches = [note.pitch for note in by_name.get("vocal_harmony", [])]

    assert report["is_instrumental_song"] is False
    assert report["sheetsage_rejected_instrumental_notes"] == 1
    assert report["tsumugi_harmony_promoted_from_melody"] == 1
    assert report["tsumugi_uncovered_lead_filled"] == 1
    assert melody_pitches == [72, 74, 79]
    assert harmony_pitches == [76]


def test_vocal_melody_fusion_aligns_homorhythmic_harmony_without_breaking_counterpoint(tmp_path: Path) -> None:
    sample_rate = 16_000
    duration_seconds = 4.0

    # 0.50〜1.00s と 1.00〜1.50s: 連続レガート主旋律 C5 (72) + 同音高連続ハモリ E5 (76)
    # 1.80〜2.80s: ロング主旋律 D5(1.80〜2.30) -> E5(2.30〜2.80) にまたがる持続ハモリ G5(79)
    # 3.10〜3.60s: 主旋律休符中の追っかけハモリ A5(81)（主旋律から 300ms 離れているので吸着されないこと）
    vocal_mono = _synthesize_harmonic_tone(
        sample_rate,
        duration_seconds,
        [
            (0.50, 1.50, 72, 0.30),
            (0.50, 1.50, 76, 0.24),
            (1.80, 2.30, 74, 0.30),
            (2.30, 2.80, 76, 0.30),
            (1.80, 2.80, 79, 0.24),
            (3.10, 3.60, 81, 0.24),
        ],
    )
    prediction = SheetSageMelodyPrediction(
        duration_seconds=duration_seconds,
        notes=(
            SheetSageMelodyNote(start_seconds=0.50, end_seconds=1.00, pitch=72, track=0, vocal_probability=0.95),
            SheetSageMelodyNote(start_seconds=1.00, end_seconds=1.50, pitch=72, track=0, vocal_probability=0.95),
            SheetSageMelodyNote(start_seconds=1.80, end_seconds=2.30, pitch=74, track=0, vocal_probability=0.95),
            SheetSageMelodyNote(start_seconds=2.30, end_seconds=2.80, pitch=76, track=0, vocal_probability=0.95),
        ),
        vocal_probabilities=np.ones(100, dtype=np.float32),
    )

    # tsumugi ハモリが +25〜30ms 遅れてフラムしている状態を再現する。
    tsumugi_midi_path = tmp_path / "tsumugi_flam.mid"
    tsumugi_midi = pretty_midi.PrettyMIDI()
    ts_harm = pretty_midi.Instrument(program=65, name="vocal_harmony")
    ts_harm.notes.append(pretty_midi.Note(velocity=95, pitch=76, start=0.53, end=1.02))
    ts_harm.notes.append(pretty_midi.Note(velocity=95, pitch=76, start=1.03, end=1.53))
    ts_harm.notes.append(pretty_midi.Note(velocity=95, pitch=79, start=1.83, end=2.82))
    ts_harm.notes.append(pretty_midi.Note(velocity=95, pitch=81, start=3.12, end=3.58))
    tsumugi_midi.instruments.append(ts_harm)
    tsumugi_midi.write(str(tsumugi_midi_path))

    fused_midi, report = fuse_sheetsage_melody_with_tsumugi_vocals(
        prediction,
        tsumugi_vocal_midi_path=tsumugi_midi_path,
        preloaded_vocal_mono=vocal_mono,
        preloaded_mix_mono=vocal_mono,
        config=VocalMelodyFusionConfig(),
    )
    harm_notes = next(inst.notes for inst in fused_midi.instruments if inst.name == "vocal_harmony")
    assert len(harm_notes) == 4  # 連続同音高 2 音節 (0.5..1.0, 1.0..1.5) が誤結合されず保持されること
    assert abs(harm_notes[0].start - 0.50) < 1e-4 and abs(harm_notes[0].end - 1.00) < 1e-4
    assert abs(harm_notes[1].start - 1.00) < 1e-4 and abs(harm_notes[1].end - 1.50) < 1e-4
    # 2 音にまたがるロングハモリは開始を 1.80、終了を 2.80 へ綺麗に吸着すること
    assert abs(harm_notes[2].start - 1.80) < 1e-4 and abs(harm_notes[2].end - 2.80) < 1e-4
    # 追っかけハモリ (3.12s) は主旋律 (2.80s) へ引っ張られずに保持されること（MIDI tick 丸め誤差 ±3ms を許容）
    assert abs(harm_notes[3].start - 3.12) < 3e-3 and abs(harm_notes[3].end - 3.58) < 3e-3
    assert report["harmony_snapped_onsets"] == 3
    assert report["harmony_snapped_offsets"] == 3


def test_vocal_melody_fusion_restores_swing_onset_when_audio_confirms_tsumugi(tmp_path: Path) -> None:
    sample_rate = 16_000
    duration_seconds = 3.0

    # 120 BPM (1 拍 = 0.50s) のスウィング裏拍:
    # - 1 音目: 0.50〜0.833s (C5=72)
    # - 2 音目（スウィング 3 連裏拍 2/3）: 0.833〜1.00s ではなく 0.833〜1.20s (E5=76) + 同時ハモリ G5=79
    # SheetSage2 は 16 分量子化バイアスで 2 音目をストレート裏拍 0.75s に早出ししてしまうケースを再現。
    vocal_mono = _synthesize_harmonic_tone(
        sample_rate,
        duration_seconds,
        [
            (0.50, 0.833, 72, 0.30),
            (0.833, 1.25, 76, 0.30),
            (0.833, 1.25, 79, 0.24),
        ],
    )
    prediction = SheetSageMelodyPrediction(
        duration_seconds=duration_seconds,
        notes=(
            SheetSageMelodyNote(start_seconds=0.50, end_seconds=0.75, pitch=72, track=0, vocal_probability=0.95),
            # 16 分量子化バイアスで 83ms 早くずれたスウィング裏拍
            SheetSageMelodyNote(start_seconds=0.75, end_seconds=1.25, pitch=76, track=0, vocal_probability=0.95),
        ),
        vocal_probabilities=np.ones(75, dtype=np.float32),
    )

    tsumugi_midi_path = tmp_path / "tsumugi_swing.mid"
    tsumugi_midi = pretty_midi.PrettyMIDI()
    ts_mel = pretty_midi.Instrument(program=65, name="melody")
    ts_mel.notes.append(pretty_midi.Note(velocity=100, pitch=72, start=0.50, end=0.833))
    ts_mel.notes.append(pretty_midi.Note(velocity=100, pitch=76, start=0.833, end=1.25))
    ts_harm = pretty_midi.Instrument(program=65, name="vocal_harmony")
    ts_harm.notes.append(pretty_midi.Note(velocity=95, pitch=79, start=0.845, end=1.24))
    tsumugi_midi.instruments.extend([ts_mel, ts_harm])
    tsumugi_midi.write(str(tsumugi_midi_path))

    fused_midi, report = fuse_sheetsage_melody_with_tsumugi_vocals(
        prediction,
        tsumugi_vocal_midi_path=tsumugi_midi_path,
        preloaded_vocal_mono=vocal_mono,
        preloaded_mix_mono=vocal_mono,
        config=VocalMelodyFusionConfig(),
    )
    by_name = {inst.name: inst.notes for inst in fused_midi.instruments}
    mel_notes = by_name["melody"]
    harm_notes = by_name["vocal_harmony"]

    assert report["lead_swing_onsets_refined"] == 1
    # 2 音目のスウィング裏拍 (0.833s) が復元され、前音の終端とハモリの開始も 0.833s に揃うこと。
    assert abs(mel_notes[0].end - 0.833) < 0.005
    assert abs(mel_notes[1].start - 0.833) < 0.005
    assert abs(harm_notes[0].start - 0.833) < 0.005
