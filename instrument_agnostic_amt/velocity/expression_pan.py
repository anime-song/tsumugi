"""ステレオ分離ステムと MIDI から左右パン (CC#10) と持続系楽器の動的強弱 (CC#11) を抽出する。

1. 左右パン (CC#10 Pan):
   - ステレオ音声の L/R パワー比（等パワーパン法則: atan2(R_rms, L_rms)）から
     トラック基準パンおよびフレーズ単位のパン移動を抽出する。
   - ボーカル・ベース・ドラムはミックスの土台が崩れないよう中央 (64) に固定する。

2. 持続系楽器の動的強弱 (CC#11 Expression):
   - ストリングス・フルート・木管・金管・オルガン・シンセパッド・ボーカルなど、
     発音中に息や弓で強弱を変化させられる楽器の一定長以上のノート／レガートフレーズに対し、
     平滑化した K-weighted エネルギー包絡線からクレッシェンド／デクレッシェンド／スウェルを
     CC#11 イベント列として生成する。
   - ピークを 127 に揃えて SoundFont の抜けを保ち、フレーズ終了直後に 127 へ戻すことで
     後続の短いノートが小さい CC#11 を引きずる事故を防ぐ。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pretty_midi
import torch

from .render_compare_velocity import _apply_k_weighting, _build_harmonic_pitch_filterbank

# 発音中の強弱変化（クレッシェンド・デクレッシェンド・スウェル）が意味を持つ持続系 GM プログラム群。
# Pizzicato Strings (45), Orchestral Harp (46), Timpani (47), Orchestra Hit (55) は減衰系なので除外する。
SUSTAINED_GM_PROGRAMS: frozenset[int] = frozenset(
    {
        *range(16, 24),  # Organ / Accordion / Harmonica
        40,
        41,
        42,
        43,
        44,  # Bowed Strings
        48,
        49,
        50,
        51,  # String Ensemble / Synth Strings
        52,
        53,
        54,  # Choir / Voice
        *range(56, 64),  # Brass
        *range(64, 72),  # Reed / Sax / Woodwind
        *range(72, 80),  # Pipe / Flute
        *range(80, 96),  # Synth Lead / Synth Pad
    }
)


@dataclass(frozen=True)
class ExpressionPanConfig:
    """左右パン (CC#10) と持続系 CC#11 Expression 抽出の設定。"""

    sample_rate: int = 22_050
    enable_pan_cc10: bool = True
    # CC#11 は SoundFont 内蔵のアタックエンベロープと二重に掛かりアタック感を損ないやすいため、
    # 既定では無効（オプトイン）とする。
    enable_expression_cc11: bool = False
    # パンを中央 (64) に固定して保護するステム名フラグメント。
    center_locked_stems: tuple[str, ...] = ("vocal", "bass", "drum")
    min_pan: int = 16
    max_pan: int = 112
    # フレーズ間でこの値以上パンが動いたときのみ動的 CC#10 イベントを追加する。
    dynamic_pan_threshold: int = 8
    dynamic_pan_phrase_gap_seconds: float = 0.25
    # CC#11 抽出のパラメータ。
    min_expression_duration_seconds: float = 0.32
    legato_merge_gap_seconds: float = 0.08
    expression_step_seconds: float = 0.045
    expression_smoothing_seconds: float = 0.12
    expression_depth: float = 0.75
    min_expression_cc11: int = 54
    min_expression_range: int = 10
    expression_change_threshold: int = 3
    expression_curve_range_db: float = 40.0
    compensate_velocity_for_expression: bool = True
    max_velocity: int = 127

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 <= self.min_pan <= 64 <= self.max_pan <= 127:
            raise ValueError("min_pan and max_pan must bracket center 64 within 0..127")
        if self.dynamic_pan_threshold <= 0:
            raise ValueError("dynamic_pan_threshold must be positive")
        if self.min_expression_duration_seconds <= 0.0 or self.expression_step_seconds <= 0.0:
            raise ValueError("expression duration and step must be positive")
        if not 1 <= self.min_expression_cc11 <= 127:
            raise ValueError("min_expression_cc11 must be in 1..127")
        if self.expression_depth < 0.0 or self.expression_curve_range_db <= 0.0:
            raise ValueError("expression_depth must be nonnegative and curve range positive")


def is_sustained_gm_program(program_number: int, *, is_drum: bool) -> bool:
    """GM プログラム番号が CC#11 Expression 追従に適した持続系楽器かを判定する。"""
    if is_drum:
        return False
    return int(program_number) in SUSTAINED_GM_PROGRAMS


def _energy_pair_to_pan_cc10(left_power: float, right_power: float, *, min_pan: int, max_pan: int) -> int:
    """等パワーパン法則に基づき、左右パワー比を MIDI CC#10 (0=L, 64=C, 127=R) へ変換する。"""
    left_rms = math.sqrt(max(float(left_power), 1e-12))
    right_rms = math.sqrt(max(float(right_power), 1e-12))
    theta = math.atan2(right_rms, left_rms)
    pan_value = int(round(127.0 * theta / (math.pi * 0.5)))
    return int(np.clip(pan_value, min_pan, max_pan))


def _compute_channel_power_maps(
    stereo_audio: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    pitch_filterbank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ステレオ音声 [2, T] から K-weighted の左右広帯域パワー・モノラル広帯域パワー・
    左右ピッチ別倍音パワーを計算する。
    """
    if stereo_audio.ndim != 2 or int(stereo_audio.shape[0]) != 2:
        raise ValueError("stereo_audio must have shape [2, T]")

    left_k = _apply_k_weighting(stereo_audio[0], sample_rate=sample_rate)
    right_k = _apply_k_weighting(stereo_audio[1], sample_rate=sample_rate)
    if left_k.size < n_fft:
        pad = n_fft - left_k.size
        left_k = np.pad(left_k, (0, pad))
        right_k = np.pad(right_k, (0, pad))

    window = torch.hann_window(n_fft, dtype=torch.float32)
    window_energy = float(window.square().sum().item())
    batch_wave = torch.from_numpy(np.stack((left_k, right_k), axis=0))
    spectrum = torch.stft(
        batch_wave,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power_lr = (spectrum.real.square() + spectrum.imag.square()).numpy() / max(window_energy, 1e-8)

    broadband_weights = np.full(power_lr.shape[1], 2.0, dtype=np.float32)
    broadband_weights[0] = 1.0
    if n_fft % 2 == 0:
        broadband_weights[-1] = 1.0
    left_broadband = (broadband_weights @ power_lr[0]) / float(n_fft)
    right_broadband = (broadband_weights @ power_lr[1]) / float(n_fft)
    mono_broadband = 0.5 * (left_broadband + right_broadband)

    left_pitch = pitch_filterbank @ power_lr[0]
    right_pitch = pitch_filterbank @ power_lr[1]
    return left_broadband, right_broadband, mono_broadband, left_pitch, right_pitch


def _extract_track_pan_events(
    instrument: pretty_midi.Instrument,
    *,
    stem_name: str,
    track_count_in_stem: int,
    left_broadband: np.ndarray,
    right_broadband: np.ndarray,
    left_pitch: np.ndarray,
    right_pitch: np.ndarray,
    frames_per_second: float,
    config: ExpressionPanConfig,
    pitch_min: int = 21,
    pitch_max: int = 108,
) -> tuple[list[pretty_midi.ControlChange], dict[str, Any]]:
    """1 トラック分の CC#10 左右パンイベント列を生成する。"""
    normalized_stem = str(stem_name).lower()
    if instrument.is_drum or any(token in normalized_stem for token in config.center_locked_stems):
        return (
            [pretty_midi.ControlChange(number=10, value=64, time=0.0)],
            {"base_pan": 64, "pan_events": 1, "pan_min": 64, "pan_max": 64, "center_locked": True},
        )

    ordered_notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    if not ordered_notes:
        return (
            [pretty_midi.ControlChange(number=10, value=64, time=0.0)],
            {"base_pan": 64, "pan_events": 1, "pan_min": 64, "pan_max": 64, "center_locked": False},
        )

    num_frames = int(left_broadband.size)
    use_pitch_channel = track_count_in_stem > 1

    # 1. 各ノートの左右パワー比からノート単位のパン推定値を求める。
    note_pans: list[float] = []
    note_energies: list[float] = []
    for note in ordered_notes:
        start_frame = max(0, min(num_frames - 1, int(math.floor(float(note.start) * frames_per_second))))
        end_frame = min(num_frames, max(start_frame + 1, int(math.ceil(float(note.end) * frames_per_second))))
        if use_pitch_channel:
            clamped_pitch = max(pitch_min, min(pitch_max, int(note.pitch)))
            left_energy = float(np.mean(left_pitch[clamped_pitch - pitch_min, start_frame:end_frame]))
            right_energy = float(np.mean(right_pitch[clamped_pitch - pitch_min, start_frame:end_frame]))
        else:
            left_energy = float(np.mean(left_broadband[start_frame:end_frame]))
            right_energy = float(np.mean(right_broadband[start_frame:end_frame]))
        note_pans.append(
            float(
                _energy_pair_to_pan_cc10(
                    left_energy,
                    right_energy,
                    min_pan=int(config.min_pan),
                    max_pan=int(config.max_pan),
                )
            )
        )
        note_energies.append(left_energy + right_energy)

    # 2. トラック全体の基準パン（中央値）を計算する。
    pan_array = np.asarray(note_pans, dtype=np.float64)
    base_pan = int(np.clip(round(float(np.median(pan_array))), config.min_pan, config.max_pan))
    pan_std = float(np.std(pan_array))

    # パンの変化が小さければ t=0 に基準パンを 1 個だけ書いてシンプルに保つ。
    if pan_std < float(config.dynamic_pan_threshold) * 0.65 or len(ordered_notes) < 4:
        return (
            [pretty_midi.ControlChange(number=10, value=base_pan, time=0.0)],
            {"base_pan": base_pan, "pan_events": 1, "pan_min": base_pan, "pan_max": base_pan, "center_locked": False},
        )

    # 3. フレーズ（休符 >= dynamic_pan_phrase_gap_seconds で区切る）ごとに代表パンを算出し、
    #    大きく左右へ移動するフレーズ先頭でのみ CC#10 を更新する。
    phrases: list[list[int]] = [[0]]
    current_phrase_end = float(ordered_notes[0].end)
    for index in range(1, len(ordered_notes)):
        note_start = float(ordered_notes[index].start)
        if note_start - current_phrase_end >= float(config.dynamic_pan_phrase_gap_seconds):
            phrases.append([index])
            current_phrase_end = float(ordered_notes[index].end)
        else:
            phrases[-1].append(index)
            current_phrase_end = max(current_phrase_end, float(ordered_notes[index].end))

    # 最初のフレーズのパンを t=0 の初期値にして、発音直前の不要なパン移動を防ぐ。
    first_phrase_pan = int(np.clip(round(float(np.median(pan_array[phrases[0]]))), config.min_pan, config.max_pan))
    events: list[pretty_midi.ControlChange] = [pretty_midi.ControlChange(number=10, value=first_phrase_pan, time=0.0)]
    last_pan = first_phrase_pan
    for phrase_indices in phrases[1:]:
        phrase_pan = int(np.clip(round(float(np.median(pan_array[phrase_indices]))), config.min_pan, config.max_pan))
        if abs(phrase_pan - last_pan) >= int(config.dynamic_pan_threshold):
            event_time = max(0.0, float(ordered_notes[phrase_indices[0]].start) - 0.005)
            events.append(pretty_midi.ControlChange(number=10, value=phrase_pan, time=event_time))
            last_pan = phrase_pan

    emitted_values = [int(event.value) for event in events]
    return (
        events,
        {
            "base_pan": base_pan,
            "pan_events": len(events),
            "pan_min": min(emitted_values),
            "pan_max": max(emitted_values),
            "center_locked": False,
        },
    )


def _extract_track_expression_events(
    instrument: pretty_midi.Instrument,
    *,
    track_count_in_stem: int,
    mono_broadband: np.ndarray,
    mono_pitch: np.ndarray,
    frames_per_second: float,
    config: ExpressionPanConfig,
    pitch_min: int = 21,
    pitch_max: int = 108,
) -> tuple[list[pretty_midi.ControlChange], dict[str, Any]]:
    """持続系楽器トラックの長音／レガートフレーズから CC#11 Expression カーブを生成する。"""
    if not is_sustained_gm_program(int(instrument.program), is_drum=bool(instrument.is_drum)):
        return (
            [pretty_midi.ControlChange(number=11, value=127, time=0.0)],
            {"expression_applied": False, "expression_events": 1, "modulated_phrases": 0},
        )

    ordered_notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    if not ordered_notes:
        return (
            [pretty_midi.ControlChange(number=11, value=127, time=0.0)],
            {"expression_applied": False, "expression_events": 1, "modulated_phrases": 0},
        )

    # 1. レガートで繋がっているノート群を 1 つの連続フレーズへまとめる。
    phrases: list[list[pretty_midi.Note]] = [[ordered_notes[0]]]
    phrase_ends: list[float] = [float(ordered_notes[0].end)]
    for note in ordered_notes[1:]:
        if float(note.start) <= phrase_ends[-1] + float(config.legato_merge_gap_seconds):
            phrases[-1].append(note)
            phrase_ends[-1] = max(phrase_ends[-1], float(note.end))
        else:
            phrases.append([note])
            phrase_ends.append(float(note.end))

    num_frames = int(mono_broadband.size)
    use_pitch_channel = track_count_in_stem > 1
    smooth_frames = max(1, int(round(float(config.expression_smoothing_seconds) * frames_per_second)))
    # ガウス平滑化カーネル（フレーム単位）。ビブラートの小刻みな揺れを抑えつつ膨らみを残す。
    kernel_radius = max(1, smooth_frames)
    kernel_grid = np.arange(-kernel_radius, kernel_radius + 1, dtype=np.float64)
    sigma = max(0.75, float(smooth_frames) * 0.5)
    smooth_kernel = np.exp(-0.5 * np.square(kernel_grid / sigma))
    smooth_kernel /= float(smooth_kernel.sum())

    events: list[pretty_midi.ControlChange] = [pretty_midi.ControlChange(number=11, value=127, time=0.0)]
    modulated_phrase_count = 0
    last_cc11_value = 127

    for phrase_index, phrase_notes in enumerate(phrases):
        phrase_start = float(phrase_notes[0].start)
        phrase_end = float(phrase_ends[phrase_index])
        phrase_duration = phrase_end - phrase_start
        if phrase_duration < float(config.min_expression_duration_seconds):
            # 短いフレーズでは CC#11 を 127 に戻して音量低下を引きずらないようにする。
            if last_cc11_value != 127:
                reset_time = max(0.0, phrase_start - 0.004)
                events.append(pretty_midi.ControlChange(number=11, value=127, time=reset_time))
                last_cc11_value = 127
            continue

        start_frame = max(0, min(num_frames - 1, int(math.floor(phrase_start * frames_per_second))))
        end_frame = min(num_frames, max(start_frame + 2, int(math.ceil(phrase_end * frames_per_second))))
        frame_count = end_frame - start_frame
        if frame_count < 3:
            continue

        # 2. フレーズ内のフレーム別エネルギー推移を取得する。
        #    同ステムに複数トラックが共存する場合は、その時刻に鳴っている対象ノート群の
        #    ピッチ倍音エネルギー合計を使うことで、他楽器の音に引っ張られるのを防ぐ。
        if use_pitch_channel:
            power_curve = np.zeros(frame_count, dtype=np.float64)
            active_counts = np.zeros(frame_count, dtype=np.int32)
            for note in phrase_notes:
                clamped_pitch = max(pitch_min, min(pitch_max, int(note.pitch)))
                note_left = max(0, int(math.floor(float(note.start) * frames_per_second)) - start_frame)
                note_right = min(frame_count, max(note_left + 1, int(math.ceil(float(note.end) * frames_per_second)) - start_frame))
                power_curve[note_left:note_right] += mono_pitch[
                    clamped_pitch - pitch_min,
                    start_frame + note_left : start_frame + note_right,
                ]
                active_counts[note_left:note_right] += 1
            empty_frames = active_counts == 0
            if np.any(empty_frames):
                power_curve[empty_frames] = mono_broadband[start_frame:end_frame][empty_frames]
        else:
            power_curve = mono_broadband[start_frame:end_frame].astype(np.float64, copy=True)

        # 対数 dB 空間でガウス平滑化する（エッジの歪みを防ぐため端を reflect/edge 補完）。
        curve_db = 10.0 * np.log10(np.maximum(power_curve, 1e-12))
        padded_db = np.pad(curve_db, (kernel_radius, kernel_radius), mode="edge")
        smoothed_db = np.convolve(padded_db, smooth_kernel, mode="valid")

        # 3. フレーズ内のピーク (92 パーセンタイル) を 0 dB (= CC#11 127) とする相対カーブへ変換。
        peak_db = float(np.percentile(smoothed_db, 92))
        relative_db = np.clip(smoothed_db - peak_db, -24.0, 0.0)
        cc11_float = 127.0 * np.power(
            10.0,
            float(config.expression_depth) * relative_db / float(config.expression_curve_range_db),
        )
        cc11_curve = np.clip(np.rint(cc11_float), config.min_expression_cc11, 127).astype(np.int32)

        # ダイナミクス変化が乏しいフラットな音には不要な CC#11 を書かない。
        if int(cc11_curve.max() - cc11_curve.min()) < int(config.min_expression_range):
            if last_cc11_value != 127:
                reset_time = max(0.0, phrase_start - 0.004)
                events.append(pretty_midi.ControlChange(number=11, value=127, time=reset_time))
                last_cc11_value = 127
            continue

        # 4. expression_step_seconds 間隔でサンプリングし、変化の大きい点だけ間引き書き出す。
        sample_times = np.arange(phrase_start, phrase_end - 1e-4, float(config.expression_step_seconds), dtype=np.float64)
        if sample_times.size == 0 or (phrase_end - float(sample_times[-1])) > 0.015:
            sample_times = np.append(sample_times, phrase_end)

        emitted_in_phrase = False
        sampled_values = [
            int(
                cc11_curve[
                    int(np.clip(round((sample_time - phrase_start) * frames_per_second), 0, frame_count - 1))
                ]
            )
            for sample_time in sample_times.tolist()
        ]
        peak_sample_index = int(np.argmax(sampled_values))
        for sample_index, (sample_time, value) in enumerate(zip(sample_times.tolist(), sampled_values)):
            # 始点・終点およびスウェルの頂点は間引き対象外にし、膨らみのピークが削れるのを防ぐ。
            is_boundary_or_peak = (
                sample_index == 0
                or sample_index == len(sample_times) - 1
                or (sample_index == peak_sample_index and value != last_cc11_value)
            )
            if is_boundary_or_peak or abs(value - last_cc11_value) >= int(config.expression_change_threshold):
                event_time = max(0.0, float(sample_time))
                if event_time <= 0.005 and len(events) == 1:
                    events[0] = pretty_midi.ControlChange(number=11, value=value, time=0.0)
                else:
                    events.append(pretty_midi.ControlChange(number=11, value=value, time=event_time))
                last_cc11_value = value
                emitted_in_phrase = True

        if emitted_in_phrase:
            modulated_phrase_count += 1
            # CC#11 による平均音量低下分を Note-On Velocity 側でわずかに補償し、1-B のミックスバランスを保つ。
            if config.compensate_velocity_for_expression:
                mean_ratio = float(np.mean(cc11_curve)) / 127.0
                if 0.1 < mean_ratio < 0.98:
                    mean_drop_db = float(config.expression_curve_range_db) * math.log10(mean_ratio)
                    boost_ratio = 10.0 ** (-0.45 * mean_drop_db / 36.0)
                    for note in phrase_notes:
                        note.velocity = int(
                            np.clip(round(float(note.velocity) * boost_ratio), 1, int(config.max_velocity))
                        )

            # フレーズ終端直後に 127 へ戻しておく（次ノートがすぐ来るときは直前に置く）。
            if last_cc11_value != 127:
                next_start = (
                    float(phrases[phrase_index + 1][0].start)
                    if phrase_index + 1 < len(phrases)
                    else phrase_end + 0.05
                )
                reset_time = min(phrase_end + 0.015, max(phrase_end, next_start - 0.004))
                events.append(pretty_midi.ControlChange(number=11, value=127, time=reset_time))
                last_cc11_value = 127

    return (
        events,
        {
            "expression_applied": modulated_phrase_count > 0,
            "expression_events": len(events),
            "modulated_phrases": modulated_phrase_count,
        },
    )


def apply_pan_and_expression_to_stem_midi(
    midi_object: pretty_midi.PrettyMIDI,
    stereo_audio: np.ndarray,
    *,
    stem_name: str,
    config: ExpressionPanConfig | None = None,
) -> list[dict[str, Any]]:
    """1 ステム分のステレオ音声と PrettyMIDI へ CC#10 (Pan) と CC#11 (Expression) を抽出適用する。"""
    effective_config = config or ExpressionPanConfig()
    active_instruments = [instrument for instrument in midi_object.instruments if instrument.notes]
    if not active_instruments:
        return []

    sample_rate = int(effective_config.sample_rate)
    n_fft = min(2048, max(128, 2 ** int(round(math.log2(max(128.0, float(sample_rate) * 0.09))))))
    hop_length = max(32, min(256, n_fft // 8))
    frames_per_second = float(sample_rate) / float(hop_length)
    pitch_filterbank, _ = _build_harmonic_pitch_filterbank(sample_rate=sample_rate, n_fft=n_fft)

    left_broadband, right_broadband, mono_broadband, left_pitch, right_pitch = _compute_channel_power_maps(
        stereo_audio,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        pitch_filterbank=pitch_filterbank,
    )
    mono_pitch = 0.5 * (left_pitch + right_pitch)
    track_count = len(active_instruments)

    reports: list[dict[str, Any]] = []
    for instrument in active_instruments:
        preserved_controls = [
            control
            for control in instrument.control_changes
            if int(control.number) not in (
                (10 if effective_config.enable_pan_cc10 else -1),
                (11 if effective_config.enable_expression_cc11 else -1),
            )
        ]
        track_report: dict[str, Any] = {
            "stem_name": stem_name,
            "instrument_name": str(instrument.name),
            "program_number": int(instrument.program),
            "is_drum": bool(instrument.is_drum),
        }

        if effective_config.enable_pan_cc10:
            pan_events, pan_info = _extract_track_pan_events(
                instrument,
                stem_name=stem_name,
                track_count_in_stem=track_count,
                left_broadband=left_broadband,
                right_broadband=right_broadband,
                left_pitch=left_pitch,
                right_pitch=right_pitch,
                frames_per_second=frames_per_second,
                config=effective_config,
            )
            preserved_controls.extend(pan_events)
            track_report.update(pan_info)

        if effective_config.enable_expression_cc11:
            expression_events, expression_info = _extract_track_expression_events(
                instrument,
                track_count_in_stem=track_count,
                mono_broadband=mono_broadband,
                mono_pitch=mono_pitch,
                frames_per_second=frames_per_second,
                config=effective_config,
            )
            preserved_controls.extend(expression_events)
            track_report.update(expression_info)

        preserved_controls.sort(key=lambda control: (control.time, control.number))
        instrument.control_changes = preserved_controls
        reports.append(track_report)

    return reports
