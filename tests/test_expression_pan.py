from __future__ import annotations

import numpy as np
import pretty_midi

from instrument_agnostic_amt.velocity.expression_pan import (
    ExpressionPanConfig,
    apply_pan_and_expression_to_stem_midi,
    is_sustained_gm_program,
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
