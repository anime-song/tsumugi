from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mido
import numpy as np
import pretty_midi
import soundfile as sf
import torch

from instrument_agnostic_amt.instrument_refinement.cli.infer import parse_args as parse_refinement_args
from instrument_agnostic_amt.instrument_refinement.data.midi import load_refinement_note_table
from instrument_agnostic_amt.instrument_refinement.inference import refine as refine_module
from instrument_agnostic_amt.instrument_refinement.inference.refine import refine_midi_instruments
from instrument_agnostic_amt.instrument_refinement.inference.render_compare import (
    RenderCompareConfig,
    _build_packed_candidate_midi,
    _collect_candidate_programs,
    _select_cluster_snippets,
    select_gm_programs_by_render_compare,
)
from instrument_agnostic_amt.instrument_refinement.modeling.model import InstrumentRefinementConfig
from instrument_agnostic_amt.taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_gm_program_name,
    get_instrument_class_id_by_name,
    get_program_numbers_for_class_id,
)


def _synthesize_timbre_wave(
    duration_seconds: float,
    sample_rate: int,
    *,
    notes: list[tuple[float, float, int]],
    harmonic_weights: tuple[float, ...],
    decay_rate: float,
) -> np.ndarray:
    """テスト用に倍音構成と減衰率の異なる合成波形 [2, T] を作る。"""
    sample_count = max(1, int(round(duration_seconds * sample_rate)))
    mono = np.zeros(sample_count, dtype=np.float32)
    for note_start, note_end, pitch in notes:
        start_sample = max(0, int(round(note_start * sample_rate)))
        end_sample = min(sample_count, max(start_sample + 1, int(round(note_end * sample_rate))))
        length = end_sample - start_sample
        if length <= 0:
            continue
        time = np.arange(length, dtype=np.float64) / float(sample_rate)
        fundamental = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        envelope = np.exp(-decay_rate * time)
        signal = np.zeros(length, dtype=np.float64)
        for harmonic_index, weight in enumerate(harmonic_weights, start=1):
            frequency = fundamental * harmonic_index
            if frequency >= sample_rate * 0.48:
                continue
            signal += weight * np.sin(2.0 * np.pi * frequency * time)
        mono[start_sample:end_sample] += (0.2 * signal * envelope).astype(np.float32)
    return np.stack((mono, mono), axis=0)


def _render_packed_midi_with_profiles(
    midi_path: Path,
    wav_path: Path,
    sample_rate: int,
    *,
    profiles_by_program: dict[int, tuple[tuple[float, ...], float]],
    silent_programs: set[int] | None = None,
) -> None:
    """パック済み MIDI の program_change を解釈して決定的な合成 WAV を作る偽レンダラー。"""
    midi_file = mido.MidiFile(str(midi_path))
    ticks_per_beat = midi_file.ticks_per_beat
    tempo = 500_000
    seconds_per_tick = float(tempo) / 1_000_000.0 / float(ticks_per_beat)

    program_by_channel: dict[int, int] = {}
    active_notes: dict[tuple[int, int], tuple[float, int]] = {}
    events: list[tuple[float, float, int, int]] = []

    current_seconds = 0.0
    for message in midi_file.tracks[0]:
        current_seconds += float(message.time) * seconds_per_tick
        if message.type == "set_tempo":
            tempo = int(message.tempo)
            seconds_per_tick = float(tempo) / 1_000_000.0 / float(ticks_per_beat)
        elif message.type == "program_change":
            program_by_channel[int(message.channel)] = int(message.program)
        elif message.type == "note_on" and int(message.velocity) > 0:
            channel = int(message.channel)
            pitch = int(message.note)
            active_notes[(channel, pitch)] = (current_seconds, program_by_channel.get(channel, 0))
        elif message.type == "note_off" or (message.type == "note_on" and int(message.velocity) == 0):
            channel = int(message.channel)
            pitch = int(message.note)
            started = active_notes.pop((channel, pitch), None)
            if started is not None:
                events.append((started[0], current_seconds, pitch, started[1]))

    total_samples = max(1, int(round((current_seconds + 0.1) * sample_rate)))
    mono = np.zeros(total_samples, dtype=np.float32)
    silent = silent_programs or set()
    for start_seconds, end_seconds, pitch, program_number in events:
        if program_number in silent:
            continue
        harmonic_weights, decay_rate = profiles_by_program.get(program_number, ((1.0, 0.2), 1.0))
        start_sample = max(0, int(round(start_seconds * sample_rate)))
        end_sample = min(total_samples, max(start_sample + 1, int(round(end_seconds * sample_rate))))
        length = end_sample - start_sample
        if length <= 0:
            continue
        time = np.arange(length, dtype=np.float64) / float(sample_rate)
        fundamental = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        envelope = np.exp(-decay_rate * time)
        signal = np.zeros(length, dtype=np.float64)
        for harmonic_index, weight in enumerate(harmonic_weights, start=1):
            frequency = fundamental * harmonic_index
            if frequency >= sample_rate * 0.48:
                continue
            signal += weight * np.sin(2.0 * np.pi * frequency * time)
        mono[start_sample:end_sample] += (0.2 * signal * envelope).astype(np.float32)

    stereo = np.column_stack((mono, mono))
    sf.write(str(wav_path), stereo, sample_rate, subtype="PCM_16")


class _SpectralCentroidFakeModel:
    """音声のスペクトル重心から簡易な音色埋め込みを返すテスト用モデル。"""

    def __init__(self, *, sample_rate: int, default_class_id: int) -> None:
        self.sample_rate = sample_rate
        self.default_class_id = default_class_id

    def to(self, _device: object) -> _SpectralCentroidFakeModel:
        return self

    def eval(self) -> _SpectralCentroidFakeModel:
        return self

    def __call__(
        self,
        audio: torch.Tensor,
        *,
        note_pitch: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        batch_size, max_notes = note_pitch.shape
        device = audio.device
        window_logits = torch.full((batch_size, len(INSTRUMENT_CLASSES)), -5.0, device=device)
        window_logits[:, self.default_class_id] = 4.0
        note_logits = window_logits[:, None, :].expand(batch_size, max_notes, -1).clone()

        # 高域エネルギー比を 2 次元単位ベクトルにして音色埋め込みとする。
        mono = audio.mean(dim=1)
        spectrum = torch.fft.rfft(mono, dim=-1).abs()
        frequencies = torch.linspace(0.0, 1.0, spectrum.shape[-1], device=device)
        centroid = (spectrum * frequencies[None, :]).sum(dim=-1) / spectrum.sum(dim=-1).clamp_min(1e-6)
        angle = centroid.clamp(0.0, 1.0) * (np.pi / 2.0)
        embedding = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        note_embedding = embedding[:, None, :].expand(batch_size, max_notes, -1).contiguous()
        return {
            "window_logits": window_logits,
            "window_embedding": embedding,
            "note_logits": note_logits,
            "note_embedding": note_embedding,
        }


def test_taxonomy_exposes_all_class_programs_and_gm_names() -> None:
    strings_id = get_instrument_class_id_by_name("strings")
    strings_programs = get_program_numbers_for_class_id(strings_id)
    assert strings_programs == (40, 41, 42, 43, 44, 48, 49, 50, 51)
    assert get_gm_program_name(40) == "Violin"
    assert get_gm_program_name(42) == "Cello"
    assert get_gm_program_name(48) == "String Ensemble 1"

    piano_id = get_instrument_class_id_by_name("piano")
    assert get_program_numbers_for_class_id(piano_id) == (0, 1, 3)


def test_select_cluster_snippets_prefers_pure_non_overlapping_region(tmp_path: Path) -> None:
    sample_rate = 8_000
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="mix")
    # 前半 (0..3s) はクラスタ 0 とクラスタ 1 が重なり、後半 (5..8s) はクラスタ 0 のみ。
    for onset in (0.2, 0.8, 1.4, 2.0):
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=onset, end=onset + 0.4))
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=72, start=onset, end=onset + 0.4))
    for onset in (5.2, 5.8, 6.4, 7.0):
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=onset, end=onset + 0.4))
    midi.instruments.append(instrument)
    midi_path = tmp_path / "two_clusters.mid"
    midi.write(str(midi_path))

    notes = load_refinement_note_table(midi_path)
    cluster_ids = np.where(notes.pitch == 60, 0, 1).astype(np.int64)
    waveform = _synthesize_timbre_wave(
        9.0,
        sample_rate,
        notes=[(float(s), float(e), int(p)) for s, e, p in zip(notes.start_seconds, notes.end_seconds, notes.pitch)],
        harmonic_weights=(1.0, 0.4),
        decay_rate=1.0,
    )
    config = RenderCompareConfig(
        soundfont_path=tmp_path / "dummy.sf2",
        snippet_seconds=3.0,
        max_snippets_per_cluster=1,
    )

    snippets = _select_cluster_snippets(
        notes,
        waveform,
        cluster_ids,
        cluster_id=0,
        sample_rate=sample_rate,
        config=config,
    )
    assert len(snippets) == 1
    assert snippets[0].start_seconds >= 4.5
    assert snippets[0].purity > 0.95


def test_packed_midi_rotates_melodic_channels_and_emits_cleanup_controls(tmp_path: Path) -> None:
    sample_rate = 8_000
    midi_path = tmp_path / "single.mid"
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.2, end=0.7))
    midi.instruments.append(instrument)
    midi.write(str(midi_path))

    notes = load_refinement_note_table(midi_path)
    waveform = _synthesize_timbre_wave(
        2.0,
        sample_rate,
        notes=[(0.2, 0.7, 60)],
        harmonic_weights=(1.0,),
        decay_rate=0.5,
    )
    config = RenderCompareConfig(
        soundfont_path=tmp_path / "dummy.sf2",
        snippet_seconds=1.5,
        slot_gap_seconds=0.5,
    )
    snippets = _select_cluster_snippets(
        notes,
        waveform,
        np.zeros(notes.note_count, dtype=np.int64),
        cluster_id=0,
        sample_rate=sample_rate,
        config=config,
    )
    piano_id = get_instrument_class_id_by_name("piano")
    logits = np.full(len(INSTRUMENT_CLASSES), -5.0, dtype=np.float64)
    logits[piano_id] = 5.0
    candidates = _collect_candidate_programs(logits, allowed_class_ids=(piano_id,), config=config)

    packed_path = tmp_path / "packed.mid"
    slots = _build_packed_candidate_midi(
        {0: snippets},
        {0: candidates},
        output_midi_path=packed_path,
        config=config,
    )
    assert len(slots) == 3
    assert [slot.program_number for slot in slots] == [0, 1, 3]
    assert all(slot.channel != 9 for slot in slots)
    assert len({slot.channel for slot in slots}) == 3

    loaded = mido.MidiFile(str(packed_path))
    control_numbers = {message.control for message in loaded.tracks[0] if message.type == "control_change"}
    assert {7, 11, 64, 120, 123}.issubset(control_numbers)


def test_render_compare_picks_timbrally_closest_gm_program_and_skips_silent_candidate(tmp_path: Path) -> None:
    sample_rate = 8_000
    midi_path = tmp_path / "melody.mid"
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="strings")
    note_specs = [(0.1, 0.7, 52), (0.8, 1.4, 55), (1.5, 2.2, 57), (2.3, 3.0, 55)]
    for start, end, pitch in note_specs:
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
    midi.instruments.append(instrument)
    midi.write(str(midi_path))

    notes = load_refinement_note_table(midi_path)
    strings_id = get_instrument_class_id_by_name("strings")

    # program 42 (Cello) 相当の豊かな倍音＋持続音を狙い、program 40 (Violin) は細い高域、
    # program 41 (Viola) は急減衰、program 43 (Contrabass) は無音（音域外想定）にする。
    profiles = {
        40: ((0.2, 0.3, 0.6, 1.0, 0.9), 0.3),
        41: ((1.0, 0.7, 0.4, 0.2), 8.0),
        42: ((1.0, 0.8, 0.5, 0.3, 0.15), 0.4),
        43: ((1.0, 0.1), 0.4),
        44: ((0.4, 0.9, 0.8, 0.7), 4.0),
        48: ((0.5, 0.2, 0.1), 0.2),
        49: ((0.3, 0.8, 0.7, 0.5), 0.2),
        50: ((0.1, 0.2, 0.9, 0.9), 0.2),
        51: ((0.2, 0.9, 0.2, 0.8), 0.2),
    }
    target_audio = _synthesize_timbre_wave(
        3.5,
        sample_rate,
        notes=note_specs,
        harmonic_weights=profiles[42][0],
        decay_rate=profiles[42][1],
    )
    model = _SpectralCentroidFakeModel(sample_rate=sample_rate, default_class_id=strings_id)
    model_config = InstrumentRefinementConfig(sample_rate=sample_rate, embedding_size=2)
    target_output = model(
        torch.from_numpy(target_audio).unsqueeze(0),
        note_pitch=torch.from_numpy(notes.pitch).unsqueeze(0),
    )

    note_logits = target_output["note_logits"].squeeze(0).numpy()
    note_embeddings = target_output["note_embedding"].squeeze(0).numpy()
    global_logits = target_output["window_logits"].squeeze(0).numpy()
    cluster_ids = np.zeros(notes.note_count, dtype=np.int64)
    clusters = [
        {
            "cluster_id": 0,
            "note_count": notes.note_count,
            "start_seconds": 0.1,
            "end_seconds": 3.0,
            "candidates": [{"class_id": strings_id, "class_name": "strings", "probability": 1.0}],
            "embedding": note_embeddings.mean(axis=0).tolist(),
            "note_indices": list(range(notes.note_count)),
        }
    ]
    render_config = RenderCompareConfig(
        soundfont_path=tmp_path / "dummy.sf2",
        top_candidate_classes=1,
        max_candidate_programs=12,
        snippet_seconds=3.5,
        max_snippets_per_cluster=1,
        top_report_programs=9,
    )

    updated_classes, selected_programs, updated_clusters = select_gm_programs_by_render_compare(
        model=model,  # type: ignore[arg-type]
        model_config=model_config,
        waveform=torch.from_numpy(target_audio),
        notes=notes,
        cluster_ids=cluster_ids,
        note_logits=note_logits,
        note_embeddings=note_embeddings,
        global_logits=global_logits,
        clusters=clusters,
        allowed_class_ids=(strings_id,),
        global_logit_weight=0.0,
        stem_name="other",
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        render_compare_config=render_config,
        packed_midi_renderer=lambda midi_file, wav_file, rate: _render_packed_midi_with_profiles(
            midi_file,
            wav_file,
            rate,
            profiles_by_program=profiles,
            silent_programs={43},
        ),
    )

    # 1 番目に近いプログラムが選ばれ、音域外の無音候補 (43) が除外されていることを確認する。
    assert updated_classes.tolist() == [strings_id] * notes.note_count
    assert selected_programs.tolist() == [42] * notes.note_count
    assert updated_clusters[0]["selected_program"] == 42
    assert updated_clusters[0]["selected_program_name"] == "Cello"
    silent_entry = next(
        item for item in updated_clusters[0]["render_compare_candidates"] if item["program_number"] == 43
    )
    assert silent_entry["audible"] is False
    assert silent_entry["total_distance"] >= 100.0


def test_short_snippet_is_padded_to_minimum_backbone_length(tmp_path: Path) -> None:
    """短いクラスタ (0.4 秒) でもモデル入力が最低 2.0 秒へパディングされて落ちないこと。"""
    sample_rate = 8_000
    midi_path = tmp_path / "short.mid"
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="piano")
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.1, end=0.35))
    midi.instruments.append(instrument)
    midi.write(str(midi_path))

    notes = load_refinement_note_table(midi_path)
    piano_id = get_instrument_class_id_by_name("piano")
    target_audio = _synthesize_timbre_wave(
        1.0,
        sample_rate,
        notes=[(0.1, 0.35, 60)],
        harmonic_weights=(1.0, 0.5),
        decay_rate=3.0,
    )
    observed_lengths: list[int] = []

    class _LengthCheckingModel(_SpectralCentroidFakeModel):
        def __call__(self, audio: torch.Tensor, *, note_pitch: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            observed_lengths.append(int(audio.shape[-1]))
            return super().__call__(audio, note_pitch=note_pitch, **kwargs)

    model = _LengthCheckingModel(sample_rate=sample_rate, default_class_id=piano_id)
    model_config = InstrumentRefinementConfig(sample_rate=sample_rate, embedding_size=2)
    note_logits = np.full((1, len(INSTRUMENT_CLASSES)), -5.0, dtype=np.float64)
    note_logits[0, piano_id] = 4.0
    note_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float64)
    clusters = [
        {
            "cluster_id": 0,
            "note_count": 1,
            "start_seconds": 0.1,
            "end_seconds": 0.35,
            "candidates": [{"class_id": piano_id, "class_name": "piano", "probability": 1.0}],
            "embedding": [1.0, 0.0],
            "note_indices": [0],
        }
    ]
    profiles = {0: ((1.0, 0.5), 3.0), 1: ((0.2, 1.0), 3.0), 3: ((0.1, 0.2, 1.0), 3.0)}
    _, selected_programs, _ = select_gm_programs_by_render_compare(
        model=model,  # type: ignore[arg-type]
        model_config=model_config,
        waveform=torch.from_numpy(target_audio),
        notes=notes,
        cluster_ids=np.zeros(1, dtype=np.int64),
        note_logits=note_logits,
        note_embeddings=note_embeddings,
        global_logits=note_logits[0],
        clusters=clusters,
        allowed_class_ids=(piano_id,),
        global_logit_weight=0.0,
        stem_name="piano",
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        render_compare_config=RenderCompareConfig(
            soundfont_path=tmp_path / "dummy.sf2",
            snippet_seconds=0.4,
            max_snippets_per_cluster=1,
        ),
        packed_midi_renderer=lambda midi_file, wav_file, rate: _render_packed_midi_with_profiles(
            midi_file,
            wav_file,
            rate,
            profiles_by_program=profiles,
        ),
    )
    assert observed_lengths
    assert min(observed_lengths) >= int(round(2.0 * sample_rate))
    assert selected_programs.tolist() == [0]


def test_refine_midi_instruments_rewrites_selected_gm_program_and_keeps_default_compatible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample_rate = 8_000
    audio_path = tmp_path / "other.wav"
    midi_path = tmp_path / "other.mid"
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="amt")
    note_specs = [(0.1, 0.7, 52), (0.8, 1.4, 55), (1.5, 2.1, 57)]
    for start, end, pitch in note_specs:
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
    midi.instruments.append(instrument)
    midi.write(str(midi_path))

    strings_id = get_instrument_class_id_by_name("strings")
    target_audio = _synthesize_timbre_wave(
        2.5,
        sample_rate,
        notes=note_specs,
        harmonic_weights=(1.0, 0.8, 0.5, 0.3),
        decay_rate=0.4,
    )
    sf.write(str(audio_path), target_audio.T, sample_rate, subtype="FLOAT")

    model = _SpectralCentroidFakeModel(sample_rate=sample_rate, default_class_id=strings_id)
    model_config = InstrumentRefinementConfig(sample_rate=sample_rate, embedding_size=2)

    # 1. SoundFont 未指定（既定）は従来通り代表プログラム (strings -> 40) とクラス名を書く。
    baseline_midi = tmp_path / "baseline.mid"
    baseline_report = refine_midi_instruments(
        audio_path,
        midi_path,
        output_midi_path=baseline_midi,
        stem_name="other",
        device="cpu",
        window_seconds=2.5,
        stride_seconds=2.5,
        mode="single",
        disable_tqdm=True,
        preloaded_model=model,  # type: ignore[arg-type]
        preloaded_config=model_config,
    )
    assert baseline_report["render_compare_enabled"] is False
    baseline_pm = pretty_midi.PrettyMIDI(str(baseline_midi))
    assert baseline_pm.instruments[0].program == 40
    assert baseline_pm.instruments[0].name == "strings"

    # 2. Render & Compare 有効時は選定された具体的 GM プログラム (42: Cello) を書く。
    monkeypatch.setattr(
        refine_module,
        "select_gm_programs_by_render_compare",
        lambda **kwargs: (
            np.full(kwargs["notes"].note_count, strings_id, dtype=np.int64),
            np.full(kwargs["notes"].note_count, 42, dtype=np.int64),
            [
                {
                    **kwargs["clusters"][0],
                    "selected_program": 42,
                    "selected_program_name": "Cello",
                    "selected_class_id": strings_id,
                    "selected_class_name": "strings",
                }
            ],
        ),
    )
    rc_midi = tmp_path / "render_compare.mid"
    rc_report = refine_midi_instruments(
        audio_path,
        midi_path,
        output_midi_path=rc_midi,
        stem_name="other",
        device="cpu",
        window_seconds=2.5,
        stride_seconds=2.5,
        mode="single",
        disable_tqdm=True,
        preloaded_model=model,  # type: ignore[arg-type]
        preloaded_config=model_config,
        soundfont_path=tmp_path / "dummy.sf2",
    )
    assert rc_report["render_compare_enabled"] is True
    assert rc_report["notes"][0]["program_number"] == 42
    assert rc_report["notes"][0]["program_name"] == "Cello"
    rc_pm = pretty_midi.PrettyMIDI(str(rc_midi))
    assert rc_pm.instruments[0].program == 42
    assert rc_pm.instruments[0].name == "Cello"


def test_cli_parses_render_compare_arguments(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer",
            "--audio",
            "other.wav",
            "--midi",
            "other.mid",
            "--soundfont",
            "gm.sf2",
            "--fluidsynth-executable",
            "fluidsynth.exe",
            "--render-compare-mode",
            "stem_all",
            "--render-compare-max-programs",
            "16",
        ],
    )
    args = parse_refinement_args()
    assert args.soundfont == Path("gm.sf2")
    assert args.fluidsynth_executable == Path("fluidsynth.exe")
    assert args.render_compare_mode == "stem_all"
    assert args.render_compare_max_programs == 16
