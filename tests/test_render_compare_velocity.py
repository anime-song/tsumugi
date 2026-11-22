from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import pretty_midi
import soundfile as sf

from instrument_agnostic_amt.velocity.cli.infer_velocity import parse_args as parse_velocity_args
from instrument_agnostic_amt.velocity.render_compare_velocity import (
    RenderCompareVelocityConfig,
    optimize_velocity_and_mix_by_render_compare,
)


def _render_gm_curve_midi(
    midi_path: Path,
    wav_path: Path,
    sample_rate: int,
    *,
    program_gain_db: dict[int, float] | None = None,
    drum_gain_db: float = 0.0,
) -> None:
    """GM 標準 40 dB カーブ（velocity + CC#7）に従って決定的に合成するテスト用偽レンダラー。

    program_gain_db で楽器ごとの基準音量差を模擬できる（例：ギター +8 dB、ベース -6 dB）。
    """
    midi_file = mido.MidiFile(str(midi_path))
    ticks_per_beat = midi_file.ticks_per_beat
    tempo = 500_000
    seconds_per_tick = float(tempo) / 1_000_000.0 / float(ticks_per_beat)

    program_by_channel: dict[int, int] = {}
    cc7_by_channel: dict[int, int] = {}
    cc10_by_channel: dict[int, int] = {}
    active_notes: dict[tuple[int, int], tuple[float, int, int, int, int]] = {}
    events: list[tuple[float, float, int, int, int, int, int, bool]] = []

    current_seconds = 0.0
    for message in mido.merge_tracks(midi_file.tracks):
        current_seconds += float(message.time) * seconds_per_tick
        if message.type == "set_tempo":
            tempo = int(message.tempo)
            seconds_per_tick = float(tempo) / 1_000_000.0 / float(ticks_per_beat)
        elif message.type == "program_change":
            program_by_channel[int(message.channel)] = int(message.program)
        elif message.type == "control_change" and int(message.control) == 7:
            cc7_by_channel[int(message.channel)] = int(message.value)
        elif message.type == "control_change" and int(message.control) == 10:
            cc10_by_channel[int(message.channel)] = int(message.value)
        elif message.type == "note_on" and int(message.velocity) > 0:
            channel = int(message.channel)
            pitch = int(message.note)
            active_notes[(channel, pitch)] = (
                current_seconds,
                int(message.velocity),
                program_by_channel.get(channel, 0),
                cc7_by_channel.get(channel, 100),
                cc10_by_channel.get(channel, 64),
            )
        elif message.type == "note_off" or (message.type == "note_on" and int(message.velocity) == 0):
            channel = int(message.channel)
            pitch = int(message.note)
            started = active_notes.pop((channel, pitch), None)
            if started is not None:
                events.append((started[0], current_seconds, pitch, started[1], started[2], started[3], started[4], channel == 9))

    total_samples = max(1, int(round((current_seconds + 0.1) * sample_rate)))
    left = np.zeros(total_samples, dtype=np.float32)
    right = np.zeros(total_samples, dtype=np.float32)
    gains = program_gain_db or {}

    for start_seconds, end_seconds, pitch, velocity, program, volume_cc7, pan_cc10, is_drum in events:
        start_sample = max(0, int(round(start_seconds * sample_rate)))
        end_sample = min(total_samples, max(start_sample + 1, int(round(end_seconds * sample_rate))))
        length = end_sample - start_sample
        if length <= 0:
            continue
        time = np.arange(length, dtype=np.float64) / float(sample_rate)
        # GM 2 乗カーブ: gain = (velocity / 127)^2 * (cc7 / 127)^2
        base_gain_db = drum_gain_db if is_drum else gains.get(program, 0.0)
        amplitude = 0.25 * ((velocity / 127.0) ** 2) * ((volume_cc7 / 127.0) ** 2) * (10.0 ** (base_gain_db / 20.0))

        if is_drum:
            # Kick (36) は 80 Hz、Hi-Hat (42) は 5000 Hz（テスト用低 SR では 3000 Hz）で帯域を分ける。
            if pitch in (35, 36):
                signal = np.sin(2.0 * np.pi * 85.0 * time) * np.exp(-18.0 * time)
            elif pitch in (42, 44, 46):
                freq = min(3200.0, sample_rate * 0.38)
                signal = np.sin(2.0 * np.pi * freq * time) * np.exp(-30.0 * time)
            else:
                signal = np.sin(2.0 * np.pi * 500.0 * time) * np.exp(-20.0 * time)
        else:
            fundamental = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
            signal = np.sin(2.0 * np.pi * fundamental * time)
            if fundamental * 2.0 < sample_rate * 0.48:
                signal += 0.5 * np.sin(2.0 * np.pi * fundamental * 2.0 * time)
            signal *= np.exp(-2.0 * time)
        theta = (float(pan_cc10) / 127.0) * (np.pi * 0.5)
        rendered = (amplitude * signal).astype(np.float32)
        left[start_sample:end_sample] += rendered * float(np.cos(theta))
        right[start_sample:end_sample] += rendered * float(np.sin(theta))

    stereo = np.column_stack((left, right))
    sf.write(str(wav_path), stereo, sample_rate, subtype="PCM_16")


def test_render_compare_velocity_converges_for_dynamics_chords_mix_and_drums(tmp_path: Path) -> None:
    sample_rate = 8_000
    # 1. 正解ダイナミクス入りの MIDI を 3 ステム作成し、目標音声をレンダリングする。
    #    piano: 弱音(45) -> 中音(80) -> 強音(118) + 和音(C4 弱=50, G4 強=115)
    #    guitar: 基準音量が +8 dB 大きい SoundFont パッチを想定し、CC#7 が下がることを検証。
    #    drums: Kick 強(115) + Hi-Hat 弱(45) の同時打撃と、Kick 弱(50) + Hi-Hat 強(115) を検証。
    piano_true = tmp_path / "piano_true.mid"
    guitar_true = tmp_path / "guitar_true.mid"
    drums_true = tmp_path / "drums_true.mid"

    pm_piano = pretty_midi.PrettyMIDI()
    inst_piano = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    inst_piano.control_changes.append(pretty_midi.ControlChange(number=7, value=115, time=0.0))
    inst_piano.notes = [
        pretty_midi.Note(velocity=45, pitch=60, start=0.2, end=0.6),
        pretty_midi.Note(velocity=80, pitch=64, start=0.8, end=1.2),
        pretty_midi.Note(velocity=118, pitch=67, start=1.4, end=1.8),
        # 同時和音：下の音は弱く、上の音は強い。
        pretty_midi.Note(velocity=50, pitch=60, start=2.2, end=2.7),
        pretty_midi.Note(velocity=115, pitch=67, start=2.2, end=2.7),
    ]
    pm_piano.instruments.append(inst_piano)
    pm_piano.write(str(piano_true))

    pm_guitar = pretty_midi.PrettyMIDI()
    inst_guitar = pretty_midi.Instrument(program=30, name="Distortion Guitar")
    inst_guitar.control_changes.append(pretty_midi.ControlChange(number=7, value=65, time=0.0))
    inst_guitar.notes = [
        pretty_midi.Note(velocity=90, pitch=52, start=0.2, end=0.7),
        pretty_midi.Note(velocity=90, pitch=55, start=1.0, end=1.5),
        pretty_midi.Note(velocity=90, pitch=57, start=1.8, end=2.3),
    ]
    pm_guitar.instruments.append(inst_guitar)
    pm_guitar.write(str(guitar_true))

    pm_drums = pretty_midi.PrettyMIDI()
    inst_drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    inst_drums.control_changes.append(pretty_midi.ControlChange(number=7, value=105, time=0.0))
    inst_drums.notes = [
        pretty_midi.Note(velocity=115, pitch=36, start=0.2, end=0.35),
        pretty_midi.Note(velocity=45, pitch=42, start=0.2, end=0.35),
        pretty_midi.Note(velocity=50, pitch=36, start=1.0, end=1.15),
        pretty_midi.Note(velocity=115, pitch=42, start=1.0, end=1.15),
    ]
    pm_drums.instruments.append(inst_drums)
    pm_drums.write(str(drums_true))

    # 目標ステム音声を合成（曲全体のマスタリング差として +4 dB の定数差も載せる）。
    soundfont_offsets = {0: 0.0, 30: 8.0}
    target_audios: dict[str, Path] = {}
    for stem_name, true_midi in (("piano", piano_true), ("guitar", guitar_true), ("drums", drums_true)):
        wav_path = tmp_path / f"{stem_name}.wav"
        _render_gm_curve_midi(
            true_midi,
            wav_path,
            sample_rate,
            program_gain_db={0: 4.0, 30: 12.0},
            drum_gain_db=4.0,
        )
        target_audios[stem_name] = wav_path

    # 2. 入力 MIDI は全ノート velocity=80 固定・CC#7 なしの平坦な状態にする。
    flat_midis: dict[str, Path] = {}
    for stem_name, true_midi in (("piano", piano_true), ("guitar", guitar_true), ("drums", drums_true)):
        pm = pretty_midi.PrettyMIDI(str(true_midi))
        for inst in pm.instruments:
            inst.control_changes = []
            for note in inst.notes:
                note.velocity = 80
        flat_path = tmp_path / f"{stem_name}_flat.mid"
        pm.write(str(flat_path))
        flat_midis[stem_name] = flat_path

    # 3. 閉ループ最適化を実行する。
    output_merged = tmp_path / "merged_optimized.mid"
    output_stems_dir = tmp_path / "stem_outputs"
    report = optimize_velocity_and_mix_by_render_compare(
        stem_midis=flat_midis,
        stem_audios=target_audios,
        config=RenderCompareVelocityConfig(
            soundfont_path=tmp_path / "dummy.sf2",
            sample_rate=sample_rate,
            num_iterations=3,
            dynamics_compression=1.0,
            chord_voicing_strength=1.0,
            local_smoothing_strength=0.0,
            min_velocity=20,
            drum_min_velocity=20,
        ),
        output_midi_path=output_merged,
        output_stem_dir=output_stems_dir,
        packed_midi_renderer=lambda midi_file, wav_file, rate: _render_gm_curve_midi(
            midi_file,
            wav_file,
            rate,
            program_gain_db=soundfont_offsets,
            drum_gain_db=0.0,
        ),
    )

    # ステム間の K-weighted ミックス誤差と総合誤差が明確に減少していること。
    initial_mix_mae = report["iterations"][0]["stem_mix_mae_db"]
    final_mix_mae = report["iterations"][-1]["stem_mix_mae_db"]
    assert final_mix_mae < initial_mix_mae * 0.35
    assert final_mix_mae < 1.5

    initial_mae = report["iterations"][0]["overall_mae_db"]
    final_mae = report["iterations"][-1]["overall_mae_db"]
    assert final_mae < initial_mae * 0.45
    assert final_mae < 2.5

    # 書き出されたマージ MIDI のトラック別 CC#7 とノート Velocity を検証する。
    merged_pm = pretty_midi.PrettyMIDI(str(output_merged))
    by_name = {inst.name: inst for inst in merged_pm.instruments}
    assert set(by_name) == {"Acoustic Grand Piano", "Distortion Guitar", "Drums"}

    # Guitar は基準音量が大きく目標 CC#7 が低いので、Piano より明確に低い CC#7 になっていること。
    piano_cc7 = next(cc.value for cc in by_name["Acoustic Grand Piano"].control_changes if cc.number == 7)
    guitar_cc7 = next(cc.value for cc in by_name["Distortion Guitar"].control_changes if cc.number == 7)
    assert guitar_cc7 < piano_cc7 - 20

    # Piano の単音クレッシェンド (45 < 80 < 118) と和音バランス (C4 < G4) が復元されていること。
    piano_notes = sorted(by_name["Acoustic Grand Piano"].notes, key=lambda n: (n.start, n.pitch))
    assert piano_notes[0].velocity < piano_notes[1].velocity < piano_notes[2].velocity
    assert piano_notes[3].velocity < piano_notes[4].velocity - 15

    # Drums の同時打撃で (Kick 強 / HH 弱) と (Kick 弱 / HH 強) が正しく分離復元されていること。
    drum_notes = sorted(by_name["Drums"].notes, key=lambda n: (n.start, n.pitch))
    assert drum_notes[0].velocity > drum_notes[1].velocity + 20  # t=0.2: Kick > HH
    assert drum_notes[2].velocity < drum_notes[3].velocity - 20  # t=1.0: Kick < HH

    # 既定では CC#10 (Pan) が抽出され、CC#11 (Expression) は 127 固定のまま保持されること。
    for track_name in ("Acoustic Grand Piano", "Distortion Guitar", "Drums"):
        cc_numbers = {cc.number for cc in by_name[track_name].control_changes}
        assert {7, 10, 11}.issubset(cc_numbers)
        cc11_values = [cc.value for cc in by_name[track_name].control_changes if cc.number == 11]
        assert cc11_values == [127]
    drums_pan = [cc.value for cc in by_name["Drums"].control_changes if cc.number == 10]
    assert drums_pan == [64]


def test_velocity_cli_parses_render_compare_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer_velocity",
            "--stem-midis-dir",
            "refined_stem_midis",
            "--stems-dir",
            "stems",
            "--soundfont",
            "gm.sf2",
            "--fluidsynth-executable",
            "fluidsynth.exe",
            "--render-compare-iterations",
            "4",
            "--output-midi",
            "out.mid",
        ],
    )
    args = parse_velocity_args()
    assert args.stem_midis_dir == Path("refined_stem_midis")
    assert args.soundfont == Path("gm.sf2")
    assert args.fluidsynth_executable == Path("fluidsynth.exe")
    assert args.render_compare_iterations == 4


def test_render_compare_velocity_splits_double_guitar_and_boosts_center_solo(tmp_path: Path) -> None:
    sample_rate = 8_000
    # 左静かバッキング (CC#7=65, pan=18), 右静かバッキング (CC#7=65, pan=110),
    # 中央大音量ソロ (CC#7=120, pan=64) の正解 3 トラックからステレオ目標 WAV を合成する。
    true_guitar = pretty_midi.PrettyMIDI()
    left_inst = pretty_midi.Instrument(program=29, name="Left")
    left_inst.control_changes = [
        pretty_midi.ControlChange(number=7, value=65, time=0.0),
        pretty_midi.ControlChange(number=10, value=18, time=0.0),
    ]
    center_inst = pretty_midi.Instrument(program=29, name="Center")
    center_inst.control_changes = [
        pretty_midi.ControlChange(number=7, value=120, time=0.0),
        pretty_midi.ControlChange(number=10, value=64, time=0.0),
    ]
    right_inst = pretty_midi.Instrument(program=29, name="Right")
    right_inst.control_changes = [
        pretty_midi.ControlChange(number=7, value=65, time=0.0),
        pretty_midi.ControlChange(number=10, value=110, time=0.0),
    ]

    flat_guitar = pretty_midi.PrettyMIDI()
    merged_flat = pretty_midi.Instrument(program=29, name="Overdriven Guitar")
    for index in range(54):
        start = 0.1 + index * 0.24
        end = start + 0.16
        if index < 48:
            left_inst.notes.append(pretty_midi.Note(velocity=100, pitch=55, start=start, end=end))
            merged_flat.notes.append(pretty_midi.Note(velocity=90, pitch=55, start=start, end=end))
        center_inst.notes.append(pretty_midi.Note(velocity=110, pitch=74, start=start + 0.03, end=end + 0.03))
        merged_flat.notes.append(pretty_midi.Note(velocity=90, pitch=74, start=start + 0.03, end=end + 0.03))
        right_inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=start + 0.06, end=end + 0.06))
        merged_flat.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=start + 0.06, end=end + 0.06))

    true_guitar.instruments = [left_inst, center_inst, right_inst]
    flat_guitar.instruments = [merged_flat]
    true_midi_path = tmp_path / "guitar_true.mid"
    flat_midi_path = tmp_path / "guitar_flat.mid"
    target_wav_path = tmp_path / "guitar.wav"
    true_guitar.write(str(true_midi_path))
    flat_guitar.write(str(flat_midi_path))
    _render_gm_curve_midi(true_midi_path, target_wav_path, sample_rate)

    out_midi_path = tmp_path / "guitar_opt.mid"
    report = optimize_velocity_and_mix_by_render_compare(
        stem_midis={"guitar": flat_midi_path},
        stem_audios={"guitar": target_wav_path},
        config=RenderCompareVelocityConfig(
            soundfont_path=tmp_path / "dummy.sf2",
            sample_rate=sample_rate,
            num_iterations=3,
        ),
        output_midi_path=out_midi_path,
        packed_midi_renderer=lambda midi_file, wav_file, rate: _render_gm_curve_midi(midi_file, wav_file, rate),
    )

    assert len(report["stereo_splits"]) == 1
    out_pm = pretty_midi.PrettyMIDI(str(out_midi_path))
    by_name = {inst.name: inst for inst in out_pm.instruments}
    assert set(by_name) == {"Overdriven Guitar (L)", "Overdriven Guitar (C)", "Overdriven Guitar (R)"}
    cc7 = {name: next(cc.value for cc in inst.control_changes if cc.number == 7) for name, inst in by_name.items()}
    cc10 = {name: next(cc.value for cc in inst.control_changes if cc.number == 10) for name, inst in by_name.items()}
    assert cc10["Overdriven Guitar (L)"] < 40
    assert 54 <= cc10["Overdriven Guitar (C)"] <= 74
    assert cc10["Overdriven Guitar (R)"] > 90
    # 中央ギターソロ (C) の CC#7 が左右の静かなバッキング (L)/(R) より明確に大きくなること。
    assert cc7["Overdriven Guitar (C)"] > cc7["Overdriven Guitar (L)"] + 15
    assert cc7["Overdriven Guitar (C)"] > cc7["Overdriven Guitar (R)"] + 15

