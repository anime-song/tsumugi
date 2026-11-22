from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import pretty_midi
import soundfile as sf

from instrument_agnostic_amt.beat_chord.tempo_map_export import MeterSegmentSpec
from instrument_agnostic_amt.quantization.render_compare_quantize import (
    BeatGridInput,
    RenderCompareQuantizeConfig,
    align_beat_grid_to_audio,
    quantize_stems_by_render_compare,
)


def _render_percussive_test_midi(midi_path: Path, wav_path: Path, sample_rate: int) -> None:
    """鋭いアタックを持つ決定的なテスト用偽レンダラー。"""
    midi_file = mido.MidiFile(str(midi_path))
    ticks_per_beat = midi_file.ticks_per_beat
    tempo = 500_000
    seconds_per_tick = float(tempo) / 1_000_000.0 / float(ticks_per_beat)

    active_notes: dict[tuple[int, int], tuple[float, int]] = {}
    events: list[tuple[float, float, int, int, bool]] = []
    current_seconds = 0.0
    for message in mido.merge_tracks(midi_file.tracks):
        current_seconds += float(message.time) * seconds_per_tick
        if message.type == "set_tempo":
            tempo = int(message.tempo)
            seconds_per_tick = float(tempo) / 1_000_000.0 / float(ticks_per_beat)
        elif message.type == "note_on" and int(message.velocity) > 0:
            active_notes[(int(message.channel), int(message.note))] = (current_seconds, int(message.velocity))
        elif message.type == "note_off" or (message.type == "note_on" and int(message.velocity) == 0):
            started = active_notes.pop((int(message.channel), int(message.note)), None)
            if started is not None:
                events.append((started[0], current_seconds, int(message.note), started[1], int(message.channel) == 9))

    total_samples = max(1, int(round((current_seconds + 0.15) * sample_rate)))
    mono = np.zeros(total_samples, dtype=np.float32)
    for start_seconds, end_seconds, pitch, velocity, is_drum in events:
        start_sample = max(0, int(round(start_seconds * sample_rate)))
        end_sample = min(total_samples, max(start_sample + 1, int(round(end_seconds * sample_rate))))
        length = end_sample - start_sample
        if length <= 0:
            continue
        time_axis = np.arange(length, dtype=np.float64) / float(sample_rate)
        amplitude = 0.25 * (float(velocity) / 127.0)
        if is_drum:
            signal = np.sin(2.0 * np.pi * 110.0 * time_axis) * np.exp(-28.0 * time_axis)
            signal += 0.35 * np.sin(2.0 * np.pi * 2400.0 * time_axis) * np.exp(-55.0 * time_axis)
        else:
            fundamental = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
            signal = (
                np.sin(2.0 * np.pi * fundamental * time_axis)
                + 0.45 * np.sin(2.0 * np.pi * min(fundamental * 2.0, sample_rate * 0.45) * time_axis)
            ) * np.exp(-8.0 * time_axis)
        mono[start_sample:end_sample] += (amplitude * signal).astype(np.float32)

    stereo = np.column_stack((mono, mono))
    sf.write(str(wav_path), stereo, sample_rate, subtype="PCM_16")


def test_align_beat_grid_to_audio_recovers_phase_shift() -> None:
    sample_rate = 16_000
    true_beats = np.arange(0.5, 4.5, 0.5, dtype=np.float64)  # 120 BPM
    waveform = np.zeros(int(5.0 * sample_rate), dtype=np.float32)
    for beat_time in true_beats:
        start_sample = int(round(beat_time * sample_rate))
        length = int(0.06 * sample_rate)
        time_axis = np.arange(length, dtype=np.float64) / float(sample_rate)
        pulse = 0.5 * np.sin(2.0 * np.pi * 800.0 * time_axis) * np.exp(-45.0 * time_axis)
        waveform[start_sample : start_sample + length] += pulse.astype(np.float32)

    # ビート推定が +14 ms 遅れている状態を模擬する。
    delayed_grid = BeatGridInput(
        beat_times=tuple(float(beat + 0.014) for beat in true_beats),
        downbeat_times=(0.514, 2.514),
    )
    aligned, shift_seconds, gain_ratio = align_beat_grid_to_audio(
        delayed_grid,
        waveform,
        sample_rate=sample_rate,
    )
    assert abs(shift_seconds - (-0.014)) <= 0.004
    assert gain_ratio > 0.1
    assert abs(aligned.beat_times[0] - 0.5) <= 0.004


def test_render_compare_quantize_selects_16th_triplet_and_protects_rubato(tmp_path: Path) -> None:
    sample_rate = 16_000
    # 120 BPM (1 拍 = 0.5 s, 1 小節 = 2.0 s) × 3 小節 = 6.0 s。
    # 小節 0 (0.0..2.0s): 正確な 16 分音符刻み -> 入力 MIDI に +-22 ms のジッターを載せる。
    # 小節 1 (2.0..4.0s): 正確な 3 連符刻み (1拍3分割) -> 入力 MIDI に +-18 ms のジッターを載せる。
    # 小節 2 (4.0..6.0s): グリッドに全く乗らない自由なルバート -> original を維持するはず。
    beat_times = tuple(float(index) * 0.5 for index in range(13))
    downbeat_times = (0.0, 2.0, 4.0, 6.0)
    meter_segments = (
        MeterSegmentSpec(start_seconds=0.0, end_seconds=2.0, numerator=4, denominator=4),
        MeterSegmentSpec(start_seconds=2.0, end_seconds=4.0, numerator=4, denominator=4),
        MeterSegmentSpec(start_seconds=4.0, end_seconds=6.0, numerator=4, denominator=4),
    )
    beat_grid = BeatGridInput(
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        meter_segments=meter_segments,
    )

    true_starts_bar0 = [0.0, 0.25, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 1.75]
    true_starts_bar1 = [2.0 + step * (0.5 / 3.0) for step in range(12)]
    true_starts_bar2 = [4.11, 4.41, 4.78, 5.19, 5.63]
    all_true_starts = [*true_starts_bar0, *true_starts_bar1, *true_starts_bar2]

    true_midi = pretty_midi.PrettyMIDI()
    true_instrument = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    for note_index, start_time in enumerate(all_true_starts):
        pitch = 60 + (note_index % 7)
        true_instrument.notes.append(
            pretty_midi.Note(velocity=105, pitch=pitch, start=start_time, end=start_time + 0.14)
        )
    true_midi.instruments.append(true_instrument)
    true_midi_path = tmp_path / "piano_true.mid"
    true_wav_path = tmp_path / "piano_true.wav"
    true_midi.write(str(true_midi_path))
    _render_percussive_test_midi(true_midi_path, true_wav_path, sample_rate)

    # ジッター付き入力 MIDI（小節 0・1 にノイズ、小節 2 は元から原曲通りのルバート）。
    jitter_bar0 = [0.022, -0.024, 0.020, -0.018, 0.025, -0.022, 0.020, -0.024, 0.018]
    jitter_bar1 = [0.018, -0.016, 0.019, -0.018, 0.016, -0.019, 0.018, -0.017, 0.018, -0.016, 0.019, -0.018]
    jitter_bar2 = [0.0, 0.0, 0.0, 0.0, 0.0]
    noisy_midi = pretty_midi.PrettyMIDI()
    noisy_instrument = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    for note_index, (start_time, jitter) in enumerate(zip(all_true_starts, [*jitter_bar0, *jitter_bar1, *jitter_bar2])):
        shifted = max(0.0, start_time + jitter)
        pitch = 60 + (note_index % 7)
        noisy_instrument.notes.append(
            pretty_midi.Note(velocity=105, pitch=pitch, start=shifted, end=shifted + 0.14)
        )
    noisy_midi.instruments.append(noisy_instrument)
    noisy_midi_path = tmp_path / "piano_noisy.mid"
    noisy_midi.write(str(noisy_midi_path))

    output_midi_path = tmp_path / "piano_quantized.mid"
    output_stem_dir = tmp_path / "stems_out"
    report = quantize_stems_by_render_compare(
        stem_midis={"piano": noisy_midi_path},
        stem_audios={"piano": true_wav_path},
        beat_grid=beat_grid,
        config=RenderCompareQuantizeConfig(
            soundfont_path=tmp_path / "dummy.sf2",
            sample_rate=sample_rate,
            align_beats_to_audio=False,
            viterbi_transition_penalty=0.002,
            write_tempo_map=True,
        ),
        output_midi_path=output_midi_path,
        output_stem_dir=output_stem_dir,
        packed_midi_renderer=_render_percussive_test_midi,
    )

    piano_report = report["stems_report"]["piano"]
    assert piano_report["after_timing_distance"] < piano_report["before_timing_distance"] * 0.65
    assert report["full_mix"]["after_timing_distance"] < report["full_mix"]["before_timing_distance"] * 0.65
    assert report["full_mix"]["after_onset_cosine"] > report["full_mix"]["before_onset_cosine"] + 0.1
    assert report["tempo_map_written"] is True

    # 書き出された MIDI の各小節オンセット誤差が劇的に改善し、小節 2 のルバートが壊れていないこと。
    quantized_midi = pretty_midi.PrettyMIDI(str(output_midi_path))
    quantized_starts = sorted(note.start for note in quantized_midi.instruments[0].notes)
    assert len(quantized_starts) == len(all_true_starts)

    bar0_error_ms = np.mean([abs(a - b) * 1000.0 for a, b in zip(quantized_starts[:9], true_starts_bar0)])
    bar1_error_ms = np.mean([abs(a - b) * 1000.0 for a, b in zip(quantized_starts[9:21], true_starts_bar1)])
    bar2_error_ms = np.mean([abs(a - b) * 1000.0 for a, b in zip(quantized_starts[21:], true_starts_bar2)])
    assert bar0_error_ms < 4.0
    assert bar1_error_ms < 4.0
    assert bar2_error_ms < 5.0


def test_quantize_preserves_slides_falls_and_prevents_pitch_loss_or_overlap(tmp_path: Path) -> None:
    from collections import Counter
    from instrument_agnostic_amt.quantization.render_compare_quantize import (
        DEFAULT_QUANTIZE_CANDIDATES,
        _build_quantized_instrument_variant,
    )

    # 120 BPM (1 拍 = 0.5s, 16分 = 0.125s)。
    beat_times = np.arange(0.0, 5.0, 0.5, dtype=np.float64)
    tight_candidate = next(item for item in DEFAULT_QUANTIZE_CANDIDATES if item.name == "tight_16th")
    config = RenderCompareQuantizeConfig(soundfont_path=tmp_path / "dummy.sf2")

    # 1. ベース／ブラスのスライド（頭スライド F#2->G2 と フォール A2->G#2->G2）の剛体保護を検証。
    slide_instrument = pretty_midi.Instrument(program=33, name="Electric Bass (finger)")
    slide_instrument.notes = [
        # 頭スライド: 短い F#2 (40ms) が着地音 G2 (t=0.514, 350ms) の直前 40ms にある。
        pretty_midi.Note(velocity=95, pitch=42, start=0.474, end=0.514),
        pretty_midi.Note(velocity=110, pitch=43, start=0.514, end=0.864),
        # フォール: 主音 A2 (t=1.512, 300ms) の終端直後に短い G#2 (40ms) -> G2 (40ms) が落ちる。
        pretty_midi.Note(velocity=110, pitch=45, start=1.512, end=1.812),
        pretty_midi.Note(velocity=88, pitch=44, start=1.812, end=1.852),
        pretty_midi.Note(velocity=80, pitch=43, start=1.852, end=1.892),
    ]
    quantized_slide, timings = _build_quantized_instrument_variant(
        slide_instrument,
        beat_times=beat_times,
        candidate=tight_candidate,
        config=config,
    )
    assert len(timings) == len(slide_instrument.notes)
    # 頭スライドの着地音 G2 が 0.500s に揃い、直前の F#2 との 40ms 相対間隔がそのまま保たれること。
    assert abs(timings[1][0] - 0.500) < 0.002
    assert abs((timings[1][0] - timings[0][0]) - 0.040) < 0.002
    # フォールの主音 A2 が 1.500s に揃い、末尾の G#2 -> G2 の相対間隔がそのまま保たれること。
    assert abs(timings[2][0] - 1.500) < 0.002
    assert abs((timings[3][0] - timings[2][0]) - 0.300) < 0.002
    assert abs((timings[4][0] - timings[3][0]) - 0.040) < 0.002

    # 2. 順序入れ替わりが起きうる和音＋同ピッチ高速連打で、ピッチ別ノート数不変＆同ピッチ重なり 0 を検証。
    dense_instrument = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    dense_instrument.notes = [
        # わずかにズレた和音（順序入れ替わり誘発）
        pretty_midi.Note(velocity=100, pitch=67, start=0.495, end=0.700),
        pretty_midi.Note(velocity=100, pitch=60, start=0.508, end=0.710),
        # 同ピッチ高速連打（同じ 16 分グリッド 1.000s に吸い寄せられやすい 45ms 間隔）
        pretty_midi.Note(velocity=100, pitch=64, start=0.985, end=1.025),
        pretty_midi.Note(velocity=100, pitch=64, start=1.030, end=1.180),
    ]
    before_counts = Counter(note.pitch for note in dense_instrument.notes)
    quantized_dense, dense_timings = _build_quantized_instrument_variant(
        dense_instrument,
        beat_times=beat_times,
        candidate=tight_candidate,
        config=config,
    )
    after_counts = Counter(note.pitch for note in quantized_dense.notes)
    assert before_counts == after_counts
    assert len(dense_timings) == len(dense_instrument.notes)
    e4_notes = sorted((note for note in quantized_dense.notes if note.pitch == 64), key=lambda item: item.start)
    assert len(e4_notes) == 2
    # 同ピッチ連打が極短に押し潰されず、かつ先行ノート終端と後続ノート開始にリトリガー隙間があること。
    assert e4_notes[1].start - e4_notes[0].start >= 0.035
    assert e4_notes[0].end <= e4_notes[1].start - 0.007


def test_quantize_cli_parses_flags(monkeypatch) -> None:
    from instrument_agnostic_amt.quantization.cli import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "quantize",
            "--stem-midis-dir",
            "refined_stem_midis",
            "--stems-dir",
            "stems",
            "--soundfont",
            "gm.sf2",
            "--beats-json",
            "song.beats.json",
            "--output-midi",
            "quantized.mid",
        ],
    )
    args = parse_args()
    assert args.stem_midis_dir == Path("refined_stem_midis")
    assert args.stems_dir == Path("stems")
    assert args.soundfont == Path("gm.sf2")
    assert args.beats_json == Path("song.beats.json")
    assert args.output_midi == Path("quantized.mid")
    assert args.no_align_beats is False


