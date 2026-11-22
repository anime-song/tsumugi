from __future__ import annotations

import numpy as np
import pretty_midi

from instrument_agnostic_amt.velocity.expression_pan import (
    ExpressionPanConfig,
    apply_pan_and_expression_to_stem_midi,
    is_sustained_gm_program,
    split_stereo_pan_tracks_in_stem_midi,
)


def test_sustained_program_classification() -> None:
    assert is_sustained_gm_program(48, is_drum=False) is True  # String Ensemble 1
    assert is_sustained_gm_program(73, is_drum=False) is True  # Flute
    assert is_sustained_gm_program(61, is_drum=False) is True  # Brass Section
    assert is_sustained_gm_program(0, is_drum=False) is False  # Piano
    assert is_sustained_gm_program(30, is_drum=False) is False  # Distortion Guitar
    assert is_sustained_gm_program(45, is_drum=False) is False  # Pizzicato Strings
    assert is_sustained_gm_program(48, is_drum=True) is False  # Drum channel


def test_pan_extraction_tracks_left_to_right_movement_and_locks_center_stems() -> None:
    sample_rate = 8_000
    duration_seconds = 3.0
    time = np.arange(int(duration_seconds * sample_rate), dtype=np.float32) / float(sample_rate)
    tone = 0.25 * np.sin(2.0 * np.pi * 440.0 * time)

    # 前半 (0..1.2s) は左寄り、後半 (1.8..3.0s) は右寄りのステレオ音声。
    left = np.zeros_like(tone)
    right = np.zeros_like(tone)
    first_half = (time >= 0.1) & (time <= 1.2)
    second_half = (time >= 1.8) & (time <= 2.9)
    left[first_half] = tone[first_half] * 0.95
    right[first_half] = tone[first_half] * 0.20
    left[second_half] = tone[second_half] * 0.20
    right[second_half] = tone[second_half] * 0.95
    stereo = np.stack((left, right), axis=0)

    def _build_midi(program: int) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=program, name="track")
        inst.notes = [
            pretty_midi.Note(velocity=100, pitch=69, start=0.15, end=0.50),
            pretty_midi.Note(velocity=100, pitch=69, start=0.60, end=1.05),
            pretty_midi.Note(velocity=100, pitch=69, start=1.85, end=2.25),
            pretty_midi.Note(velocity=100, pitch=69, start=2.35, end=2.80),
        ]
        pm.instruments.append(inst)
        return pm

    # 1. guitar ステムでは前半左 (< 45) -> 後半右 (> 85) の動的 CC#10 が生成される。
    guitar_pm = _build_midi(30)
    guitar_reports = apply_pan_and_expression_to_stem_midi(
        guitar_pm,
        stereo,
        stem_name="guitar",
        config=ExpressionPanConfig(sample_rate=sample_rate, enable_expression_cc11=False),
    )
    pan_events = [cc for cc in guitar_pm.instruments[0].control_changes if cc.number == 10]
    assert len(pan_events) >= 2
    assert pan_events[0].value < 45
    assert pan_events[-1].value > 85
    assert guitar_reports[0]["pan_min"] < 45
    assert guitar_reports[0]["pan_max"] > 85

    # 2. bass ステムは左右に偏った音声でも中央 (64) に保護される。
    bass_pm = _build_midi(33)
    bass_reports = apply_pan_and_expression_to_stem_midi(
        bass_pm,
        stereo,
        stem_name="bass",
        config=ExpressionPanConfig(sample_rate=sample_rate, enable_expression_cc11=False),
    )
    bass_pan_events = [cc for cc in bass_pm.instruments[0].control_changes if cc.number == 10]
    assert len(bass_pan_events) == 1
    assert bass_pan_events[0].value == 64
    assert bass_reports[0]["center_locked"] is True


def test_expression_extraction_tracks_strings_crescendo_and_resets_after_phrase() -> None:
    sample_rate = 8_000
    duration_seconds = 3.0
    time = np.arange(int(duration_seconds * sample_rate), dtype=np.float32) / float(sample_rate)
    carrier = np.sin(2.0 * np.pi * 440.0 * time)

    # 0.2..1.6s: 明確なクレッシェンド (0.04 -> 0.40) -> デクレッシェンド (0.40 -> 0.05) のスウェル。
    # 2.0..2.2s: 後続の短いノート。
    envelope = np.zeros_like(time)
    rise = (time >= 0.2) & (time < 0.9)
    fall = (time >= 0.9) & (time <= 1.6)
    short_note = (time >= 2.0) & (time <= 2.2)
    envelope[rise] = np.linspace(0.04, 0.40, int(rise.sum()), dtype=np.float32)
    envelope[fall] = np.linspace(0.40, 0.05, int(fall.sum()), dtype=np.float32)
    envelope[short_note] = 0.25
    wave = carrier * envelope
    stereo = np.stack((wave, wave), axis=0)

    # 1. String Ensemble 1 (48) には CC#11 スウェルカーブが付き、終了後に 127 へ戻る。
    strings_pm = pretty_midi.PrettyMIDI()
    strings_inst = pretty_midi.Instrument(program=48, name="String Ensemble 1")
    strings_inst.notes = [
        pretty_midi.Note(velocity=100, pitch=69, start=0.2, end=1.6),
        pretty_midi.Note(velocity=100, pitch=69, start=2.0, end=2.2),
    ]
    strings_pm.instruments.append(strings_inst)

    reports = apply_pan_and_expression_to_stem_midi(
        strings_pm,
        stereo,
        stem_name="other",
        config=ExpressionPanConfig(
            sample_rate=sample_rate,
            enable_pan_cc10=False,
            enable_expression_cc11=True,
        ),
    )
    cc11_events = [cc for cc in strings_inst.control_changes if cc.number == 11]
    assert reports[0]["expression_applied"] is True
    assert reports[0]["modulated_phrases"] == 1
    assert len(cc11_events) >= 5
    # フレーズ途中でピーク 127 に達し、開始/終了付近は低い値になり、最終的に 127 へリセットされる。
    phrase_events = [cc for cc in cc11_events if 0.19 <= cc.time <= 1.61]
    assert min(cc.value for cc in phrase_events) < 95
    assert max(cc.value for cc in phrase_events) == 127
    assert cc11_events[-1].value == 127

    # 2. Piano (0) は減衰楽器なので CC#11 カーブを付けず 127 固定のまま。
    piano_pm = pretty_midi.PrettyMIDI()
    piano_inst = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    piano_inst.notes = [pretty_midi.Note(velocity=100, pitch=69, start=0.2, end=1.6)]
    piano_pm.instruments.append(piano_inst)

    piano_reports = apply_pan_and_expression_to_stem_midi(
        piano_pm,
        stereo,
        stem_name="piano",
        config=ExpressionPanConfig(
            sample_rate=sample_rate,
            enable_pan_cc10=False,
            enable_expression_cc11=True,
        ),
    )
    piano_cc11 = [cc for cc in piano_inst.control_changes if cc.number == 11]
    assert piano_reports[0]["expression_applied"] is False
    assert len(piano_cc11) == 1 and piano_cc11[0].value == 127


def test_split_stereo_pan_tracks_splits_double_guitar_plus_solo_but_keeps_centered_piano() -> None:
    sample_rate = 8_000
    duration_seconds = 16.0
    time = np.arange(int(duration_seconds * sample_rate), dtype=np.float32) / float(sample_rate)
    left = np.zeros_like(time)
    right = np.zeros_like(time)

    # ハード左 (48 notes @ C4), 中央ソロ (54 notes @ A4), ハード右 (54 notes @ E4) を同時に鳴らすステレオ音声。
    guitar_pm = pretty_midi.PrettyMIDI()
    guitar_inst = pretty_midi.Instrument(program=29, name="Overdriven Guitar")

    def _add_tone(pitch: int, start: float, end: float, left_gain: float, right_gain: float) -> None:
        s = int(round(start * sample_rate))
        e = min(time.size, int(round(end * sample_rate)))
        t = time[s:e] - start
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        wave = 0.20 * np.sin(2.0 * np.pi * freq * t)
        left[s:e] += (wave * left_gain).astype(np.float32)
        right[s:e] += (wave * right_gain).astype(np.float32)
        guitar_inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))

    for index in range(54):
        start = 0.1 + index * 0.28
        end = start + 0.18
        if index < 48:
            _add_tone(60, start, end, 0.95, 0.10)  # 左ギター
        _add_tone(69, start + 0.04, end + 0.04, 0.70, 0.70)  # 中央ソロ
        _add_tone(64, start + 0.08, end + 0.08, 0.10, 0.95)  # 右ギター

    guitar_pm.instruments.append(guitar_inst)
    stereo = np.stack((left, right), axis=0)

    reports = split_stereo_pan_tracks_in_stem_midi(
        guitar_pm,
        stereo,
        stem_name="guitar",
        config=ExpressionPanConfig(sample_rate=sample_rate),
        max_extra_tracks=3,
    )
    assert len(reports) == 1
    names = [inst.name for inst in guitar_pm.instruments]
    assert names == ["Overdriven Guitar (L)", "Overdriven Guitar (C)", "Overdriven Guitar (R)"]
    pans = {inst.name: next(cc.value for cc in inst.control_changes if cc.number == 10) for inst in guitar_pm.instruments}
    assert pans["Overdriven Guitar (L)"] < 40
    assert 54 <= pans["Overdriven Guitar (C)"] <= 74
    assert pans["Overdriven Guitar (R)"] > 90

    # ワイドステレオ配置（低音=L, 高音=R）であってもピアノ (program=0) は 1 トラックのまま誤分割されないこと。
    wide_piano_pm = pretty_midi.PrettyMIDI()
    wide_piano_inst = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    wide_piano_inst.notes = [
        pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=note.start,
            end=note.end,
        )
        for subtrack in guitar_pm.instruments
        for note in subtrack.notes
    ]
    wide_piano_pm.instruments.append(wide_piano_inst)
    assert (
        split_stereo_pan_tracks_in_stem_midi(
            wide_piano_pm,
            stereo,
            stem_name="other",
            config=ExpressionPanConfig(sample_rate=sample_rate),
        )
        == []
    )
    assert len(wide_piano_pm.instruments) == 1

    # 位相が無相関なダブルトラック（L と R が別位相で同時発音）は振幅比が中央付近でも (C) ソロへ誤分離されないこと。
    uncorr_left = np.zeros_like(time)
    uncorr_right = np.zeros_like(time)
    uncorr_pm = pretty_midi.PrettyMIDI()
    uncorr_inst = pretty_midi.Instrument(program=29, name="Overdriven Guitar")
    for index in range(60):
        start = 0.1 + index * 0.24
        end = start + 0.16
        s = int(round(start * sample_rate))
        e = min(time.size, int(round(end * sample_rate)))
        t = time[s:e] - start
        if index < 20:
            # ハード左フレーズ
            wave = 0.20 * np.sin(2.0 * np.pi * 261.63 * t)
            uncorr_left[s:e] += (wave * 0.95).astype(np.float32)
            uncorr_right[s:e] += (wave * 0.10).astype(np.float32)
            uncorr_inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=start, end=end))
        elif index < 40:
            # ハード右フレーズ
            wave = 0.20 * np.sin(2.0 * np.pi * 329.63 * t)
            uncorr_left[s:e] += (wave * 0.10).astype(np.float32)
            uncorr_right[s:e] += (wave * 0.95).astype(np.float32)
            uncorr_inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=start, end=end))
        else:
            # L と R が 90 度位相ズレ（無相関）で等振幅のダブルトラックユニゾン -> (C) は作らず (L)/(R) 2 トラックに収まる。
            uncorr_left[s:e] += (0.20 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
            uncorr_right[s:e] += (0.20 * np.cos(2.0 * np.pi * 440.0 * t)).astype(np.float32)
            uncorr_inst.notes.append(pretty_midi.Note(velocity=100, pitch=69, start=start, end=end))
    uncorr_pm.instruments.append(uncorr_inst)
    uncorr_reports = split_stereo_pan_tracks_in_stem_midi(
        uncorr_pm,
        np.stack((uncorr_left, uncorr_right), axis=0),
        stem_name="guitar",
        config=ExpressionPanConfig(sample_rate=sample_rate),
        min_notes_for_split=40,
        min_side_notes=15,
        min_center_notes=15,
    )
    assert len(uncorr_reports) == 1
    assert [inst.name for inst in uncorr_pm.instruments] == [
        "Overdriven Guitar (L)",
        "Overdriven Guitar (R)",
    ]

    # 中央寄りの自然な広がりしかないシンセ／ピアノも 1 トラックのまま誤分割されないこと。
    piano_left = np.zeros_like(time)
    piano_right = np.zeros_like(time)
    piano_pm = pretty_midi.PrettyMIDI()
    piano_inst = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    for index in range(100):
        start = 0.1 + index * 0.15
        end = start + 0.12
        s = int(round(start * sample_rate))
        e = min(time.size, int(round(end * sample_rate)))
        wave = 0.20 * np.sin(2.0 * np.pi * 440.0 * (time[s:e] - start))
        piano_left[s:e] += (wave * 0.75).astype(np.float32)
        piano_right[s:e] += (wave * 0.65).astype(np.float32)
        piano_inst.notes.append(pretty_midi.Note(velocity=100, pitch=69, start=start, end=end))
    piano_pm.instruments.append(piano_inst)
    piano_reports = split_stereo_pan_tracks_in_stem_midi(
        piano_pm,
        np.stack((piano_left, piano_right), axis=0),
        stem_name="piano",
        config=ExpressionPanConfig(sample_rate=sample_rate),
    )
    assert piano_reports == []
    assert len(piano_pm.instruments) == 1

