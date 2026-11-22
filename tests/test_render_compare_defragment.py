from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import pretty_midi
import soundfile as sf

from instrument_agnostic_amt.quantization.render_compare_defragment import (
    RenderCompareDefragmentConfig,
    defragment_stems_by_render_compare,
)


def _render_articulation_test_midi(midi_path: Path, wav_path: Path, sample_rate: int) -> None:
    """持続音と減衰音のアタック・リリースを忠実に模擬する決定的なテスト用偽レンダラー。

    Note-On ごとに 25 ms の立ち上がり＋アタック強調、Note-Off で 25 ms のリリースを行うため、
    過剰分割されたストリングスには境界で明確な V 字ディップと再アタックが生じる。
    """
    midi_file = mido.MidiFile(str(midi_path))
    ticks_per_beat = midi_file.ticks_per_beat
    tempo = 500_000
    seconds_per_tick = float(tempo) / 1_000_000.0 / float(ticks_per_beat)

    program_by_channel: dict[int, int] = {}
    active_notes: dict[tuple[int, int], tuple[float, int, int]] = {}
    events: list[tuple[float, float, int, int, int]] = []
    current_seconds = 0.0
    for message in mido.merge_tracks(midi_file.tracks):
        current_seconds += float(message.time) * seconds_per_tick
        if message.type == "set_tempo":
            tempo = int(message.tempo)
            seconds_per_tick = float(tempo) / 1_000_000.0 / float(ticks_per_beat)
        elif message.type == "program_change":
            program_by_channel[int(message.channel)] = int(message.program)
        elif message.type == "note_on" and int(message.velocity) > 0:
            channel = int(message.channel)
            active_notes[(channel, int(message.note))] = (
                current_seconds,
                int(message.velocity),
                program_by_channel.get(channel, 0),
            )
        elif message.type == "note_off" or (message.type == "note_on" and int(message.velocity) == 0):
            started = active_notes.pop((int(message.channel), int(message.note)), None)
            if started is not None:
                events.append((started[0], current_seconds, int(message.note), started[1], started[2]))

    total_samples = max(1, int(round((current_seconds + 0.2) * sample_rate)))
    mono = np.zeros(total_samples, dtype=np.float32)

    for start_seconds, end_seconds, pitch, velocity, program in events:
        start_sample = max(0, int(round(start_seconds * sample_rate)))
        end_sample = min(total_samples, max(start_sample + 1, int(round(end_seconds * sample_rate))))
        length = end_sample - start_sample
        if length <= 0:
            continue
        time_axis = np.arange(length, dtype=np.float64) / float(sample_rate)
        fundamental = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        carrier = np.sin(2.0 * np.pi * fundamental * time_axis) + 0.5 * np.sin(2.0 * np.pi * fundamental * 2.0 * time_axis)
        amplitude = 0.22 * (float(velocity) / 127.0)

        if program == 48:
            # ストリングス: 立ち上がり 30 ms、終了前 25 ms リリース、途中は滑らか持続。
            attack = np.minimum(1.0, time_axis / 0.030)
            release = np.minimum(1.0, (time_axis[-1] - time_axis) / 0.025)
            envelope = attack * release
        else:
            # ピアノ等: 鋭いアタックと指数減衰。
            envelope = np.minimum(1.0, time_axis / 0.005) * np.exp(-5.5 * time_axis)

        mono[start_sample:end_sample] += (amplitude * carrier * envelope).astype(np.float32)

    stereo = np.column_stack((mono, mono))
    sf.write(str(wav_path), stereo, sample_rate, subtype="PCM_16")


def test_defragment_merges_false_sustained_splits_but_preserves_true_repeated_notes(tmp_path: Path) -> None:
    sample_rate = 16_000

    # 1. 正解 MIDI（ストリングス）:
    #    - 0.20..2.20s: A4 (69) の 1 本の長い持続音（途中で C4->D4 の伴奏コードチェンジも入れる）。
    #    - 2.60..3.60s: E4 (64) の明確な 4 連打（各 0.20s、隙間 0.05s）。これは絶対に結合してはいけない。
    true_midi = pretty_midi.PrettyMIDI()
    true_strings = pretty_midi.Instrument(program=48, name="String Ensemble 1")
    true_strings.notes = [
        pretty_midi.Note(velocity=105, pitch=69, start=0.20, end=2.20),
        pretty_midi.Note(velocity=95, pitch=60, start=0.20, end=1.18),
        pretty_midi.Note(velocity=95, pitch=62, start=1.20, end=2.20),
        # 本物の同音 4 連打
        pretty_midi.Note(velocity=105, pitch=64, start=2.60, end=2.80),
        pretty_midi.Note(velocity=105, pitch=64, start=2.85, end=3.05),
        pretty_midi.Note(velocity=105, pitch=64, start=3.10, end=3.30),
        pretty_midi.Note(velocity=105, pitch=64, start=3.35, end=3.55),
    ]
    true_midi.instruments.append(true_strings)
    true_midi_path = tmp_path / "strings_true.mid"
    true_wav_path = tmp_path / "strings_true.wav"
    true_midi.write(str(true_midi_path))
    _render_articulation_test_midi(true_midi_path, true_wav_path, sample_rate)

    # 2. 過剰分割された入力 MIDI: A4 の 2.0s ロングトーンが 3 分割（隙間 15ms / 25ms）されている。
    fragmented_midi = pretty_midi.PrettyMIDI()
    frag_strings = pretty_midi.Instrument(program=48, name="String Ensemble 1")
    frag_strings.notes = [
        pretty_midi.Note(velocity=105, pitch=69, start=0.20, end=0.88),
        pretty_midi.Note(velocity=95, pitch=69, start=0.90, end=1.50),
        pretty_midi.Note(velocity=92, pitch=69, start=1.525, end=2.20),
        # 伴奏コード
        pretty_midi.Note(velocity=95, pitch=60, start=0.20, end=1.18),
        pretty_midi.Note(velocity=95, pitch=62, start=1.20, end=2.20),
        # 本物の同音 4 連打（そのまま残るべき）
        pretty_midi.Note(velocity=105, pitch=64, start=2.60, end=2.80),
        pretty_midi.Note(velocity=105, pitch=64, start=2.85, end=3.05),
        pretty_midi.Note(velocity=105, pitch=64, start=3.10, end=3.30),
        pretty_midi.Note(velocity=105, pitch=64, start=3.35, end=3.55),
    ]
    fragmented_midi.instruments.append(frag_strings)
    frag_midi_path = tmp_path / "strings_frag.mid"
    fragmented_midi.write(str(frag_midi_path))

    output_midi_path = tmp_path / "strings_defrag.mid"
    report = defragment_stems_by_render_compare(
        stem_midis={"other": frag_midi_path},
        stem_audios={"other": true_wav_path},
        config=RenderCompareDefragmentConfig(
            soundfont_path=tmp_path / "dummy.sf2",
            sample_rate=sample_rate,
        ),
        output_midi_path=output_midi_path,
        packed_midi_renderer=_render_articulation_test_midi,
    )

    assert report["total_merged_boundaries"] == 2
    result_midi = pretty_midi.PrettyMIDI(str(output_midi_path))
    a4_notes = [note for note in result_midi.instruments[0].notes if note.pitch == 69]
    e4_notes = [note for note in result_midi.instruments[0].notes if note.pitch == 64]
    # A4 の 3 分割は 1 本の長いノート (0.20..2.20s) へ結合されていること。
    assert len(a4_notes) == 1
    assert abs(a4_notes[0].start - 0.20) < 0.01
    assert abs(a4_notes[0].end - 2.20) < 0.01
    # E4 の本物の 4 連打は 1 つも誤結合されず 4 音のまま残っていること。
    assert len(e4_notes) == 4


def test_defragment_merges_overlapping_and_tight_smooth_sustained_seams(tmp_path: Path) -> None:
    sample_rate = 16_000
    # 正解: 0.20..1.80s の滑らかな C5 (72) 1 本のストリングス持続音。
    true_midi = pretty_midi.PrettyMIDI()
    true_inst = pretty_midi.Instrument(program=49, name="String Ensemble 2")
    true_inst.notes = [pretty_midi.Note(velocity=100, pitch=72, start=0.20, end=1.80)]
    true_midi.instruments.append(true_inst)
    true_midi_path = tmp_path / "true_seam.mid"
    true_wav_path = tmp_path / "true_seam.wav"
    true_midi.write(str(true_midi_path))
    _render_articulation_test_midi(true_midi_path, true_wav_path, sample_rate)

    # 入力: 5ms の密着ブツ切り + 完全包含重複ノート + 末尾の 25ms 極短ゴミ断片。
    frag_midi = pretty_midi.PrettyMIDI()
    frag_inst = pretty_midi.Instrument(program=49, name="String Ensemble 2")
    frag_inst.notes = [
        pretty_midi.Note(velocity=100, pitch=72, start=0.20, end=0.75),
        pretty_midi.Note(velocity=92, pitch=72, start=0.40, end=0.65),  # 包含重複
        pretty_midi.Note(velocity=95, pitch=72, start=0.755, end=1.76),  # 5ms 密着ブツ切り
        pretty_midi.Note(velocity=80, pitch=72, start=1.77, end=1.80),  # 30ms 極短ゴミ断片
    ]
    frag_midi.instruments.append(frag_inst)
    frag_midi_path = tmp_path / "frag_seam.mid"
    frag_midi.write(str(frag_midi_path))

    out_midi_path = tmp_path / "defrag_seam.mid"
    report = defragment_stems_by_render_compare(
        stem_midis={"other": frag_midi_path},
        stem_audios={"other": true_wav_path},
        config=RenderCompareDefragmentConfig(
            soundfont_path=tmp_path / "dummy.sf2",
            sample_rate=sample_rate,
        ),
        output_midi_path=out_midi_path,
        packed_midi_renderer=_render_articulation_test_midi,
    )
    assert report["total_merged_boundaries"] == 3
    result_midi = pretty_midi.PrettyMIDI(str(out_midi_path))
    assert len(result_midi.instruments[0].notes) == 1
    assert abs(result_midi.instruments[0].notes[0].start - 0.20) < 0.01
    assert abs(result_midi.instruments[0].notes[0].end - 1.80) < 0.01

    # 一方、隙間 20ms でも明確に再発音している速いストリングス 16 分連打は誤結合されないこと。
    fast_midi = pretty_midi.PrettyMIDI()
    fast_inst = pretty_midi.Instrument(program=48, name="String Ensemble 1")
    fast_inst.notes = [
        pretty_midi.Note(velocity=105, pitch=67, start=0.20, end=0.30),
        pretty_midi.Note(velocity=105, pitch=67, start=0.32, end=0.42),
        pretty_midi.Note(velocity=105, pitch=67, start=0.44, end=0.54),
        pretty_midi.Note(velocity=105, pitch=67, start=0.56, end=0.66),
    ]
    fast_midi.instruments.append(fast_inst)
    fast_midi_path = tmp_path / "fast_staccato.mid"
    fast_wav_path = tmp_path / "fast_staccato.wav"
    fast_midi.write(str(fast_midi_path))
    _render_articulation_test_midi(fast_midi_path, fast_wav_path, sample_rate)

    fast_out = tmp_path / "fast_staccato_defrag.mid"
    fast_report = defragment_stems_by_render_compare(
        stem_midis={"other": fast_midi_path},
        stem_audios={"other": fast_wav_path},
        config=RenderCompareDefragmentConfig(
            soundfont_path=tmp_path / "dummy.sf2",
            sample_rate=sample_rate,
        ),
        output_midi_path=fast_out,
        packed_midi_renderer=_render_articulation_test_midi,
    )
    assert fast_report["total_merged_boundaries"] == 0
    assert len(pretty_midi.PrettyMIDI(str(fast_out)).instruments[0].notes) == 4

    # 既定 (sustained_only=True) では歪みギターの刻み連打は対象外となり一切誤結合されないこと。
    guitar_midi = pretty_midi.PrettyMIDI()
    guitar_inst = pretty_midi.Instrument(program=30, name="Distortion Guitar")
    guitar_inst.notes = [
        pretty_midi.Note(velocity=110, pitch=52, start=0.20, end=0.34),
        pretty_midi.Note(velocity=110, pitch=52, start=0.35, end=0.49),
    ]
    guitar_midi.instruments.append(guitar_inst)
    guitar_path = tmp_path / "guitar_chug.mid"
    guitar_wav = tmp_path / "guitar_chug.wav"
    guitar_midi.write(str(guitar_path))
    _render_articulation_test_midi(guitar_path, guitar_wav, sample_rate)
    guitar_report = defragment_stems_by_render_compare(
        stem_midis={"guitar": guitar_path},
        stem_audios={"guitar": guitar_wav},
        config=RenderCompareDefragmentConfig(
            soundfont_path=tmp_path / "dummy.sf2",
            sample_rate=sample_rate,
        ),
        packed_midi_renderer=_render_articulation_test_midi,
    )
    assert guitar_report["total_candidate_boundaries"] == 0
    assert guitar_report["total_merged_boundaries"] == 0


def test_quantize_cli_parses_defragment_flags(monkeypatch) -> None:
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
            "--defragment-only",
            "--defragment-include-plucked",
        ],
    )
    args = parse_args()
    assert args.defragment_only is True
    assert args.defragment_include_plucked is True




