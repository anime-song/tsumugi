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


def split_stereo_pan_tracks_in_stem_midi(
    midi_object: pretty_midi.PrettyMIDI,
    stereo_audio: np.ndarray,
    *,
    stem_name: str,
    config: ExpressionPanConfig | None = None,
    max_extra_tracks: int = 4,
    min_notes_for_split: int = 80,
    left_pan_upper: int = 48,
    right_pan_lower: int = 80,
    min_side_notes: int = 40,
    min_side_ratio: float = 0.12,
    min_outer_ratio: float = 0.45,
    min_side_pan_gap: float = 44.0,
    min_center_notes: int = 45,
    min_center_ratio: float = 0.035,
    min_center_coherence: float = 0.48,
) -> list[dict[str, Any]]:
    """左右に明確に振り分けられたダブルトラック（ダブルギター＋中央ソロ等）を (L)/(C)/(R) サブトラックへ自動分割する。

    1 台のピアノやストリングスの自然なステレオ広がり（中央集中・左右パン差が小さい）は分割せず、
    ハード L/R パン振りされた複数楽器だけを MIDI 16ch 予算 (max_extra_tracks) の範囲内で優先分割する。
    さらに左右チャンネルの複素位相コヒーレンス Re(L·R*) / (|L|^2 + |R|^2) と旋律・和音グラフ上の
    ラベル伝播を用いることで、(1) ダブルトラックの L+R 同時和音が (C) へ誤分類される問題と、
    (2) 1 フレーズ内でノートごとに L/C/R がふらつく不安定さの両方を防ぐ。
    """
    effective_config = config or ExpressionPanConfig()
    normalized_stem = str(stem_name).lower()
    # ボーカル・ベース・ドラムに加え、1 台の鍵盤が低音(L)〜高音(R)へ連続配置されるピアノは
    # 3 トラック分割すると音域間で不自然な CC#7 段差を生むため分割対象外とする。
    if (
        any(token in normalized_stem for token in (*effective_config.center_locked_stems, "piano"))
        or max_extra_tracks <= 0
    ):
        return []

    active_instruments = [
        (index, instrument)
        for index, instrument in enumerate(midi_object.instruments)
        if not instrument.is_drum
        and int(instrument.program) not in range(0, 6)
        and len(instrument.notes) >= int(min_notes_for_split)
    ]
    if not active_instruments:
        return []

    sample_rate = int(effective_config.sample_rate)
    n_fft = min(2048, max(128, 2 ** int(round(math.log2(max(128.0, float(sample_rate) * 0.09))))))
    hop_length = max(32, min(256, n_fft // 8))
    frames_per_second = float(sample_rate) / float(hop_length)

    # 低音バッキングの 3 次倍音が高音ソロの基本波と重なってパンが引っ張られるのを防ぎつつ、
    # 歪みギター等で基本波が弱いケースにも対応するため基本波 (1.0) + 第 2 倍音 (0.45) の狭帯域フィルタバンクを使う。
    num_bins = n_fft // 2 + 1
    frequencies = np.linspace(0.0, float(sample_rate) * 0.5, num_bins, dtype=np.float64)
    bin_spacing = float(sample_rate) / float(n_fft)
    pitch_min = 21
    pitch_max = 108
    harmonic_filterbank = np.zeros((pitch_max - pitch_min + 1, num_bins), dtype=np.float32)
    for pitch in range(pitch_min, pitch_max + 1):
        fundamental_hz = 440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0))
        row = np.zeros(num_bins, dtype=np.float64)
        for harmonic_order, harmonic_weight in ((1, 1.0), (2, 0.45)):
            center_hz = fundamental_hz * float(harmonic_order)
            if center_hz >= float(sample_rate) * 0.48:
                continue
            sigma_hz = max(center_hz * 0.025, bin_spacing * 0.9)
            row += harmonic_weight * np.exp(-0.5 * np.square((frequencies - center_hz) / sigma_hz))
        row_sum = float(row.sum())
        if row_sum > 1e-8:
            harmonic_filterbank[pitch - pitch_min] = (row / row_sum).astype(np.float32)

    # 左右パワーに加え、複素クロススペクトル実部 Re(L·conj(R)) から L/R 位相コヒーレンスを算出する。
    # 中央モノラルソロは L≒R で高コヒーレンス (+0.6〜+1.0)、左右ダブルトラックは異テイクで低コヒーレンス (〜0.2) になる。
    stereo_float = np.asarray(stereo_audio, dtype=np.float32)
    if stereo_float.ndim != 2 or int(stereo_float.shape[0]) != 2:
        raise ValueError(f"Expected stereo audio with shape [2, T], got {stereo_float.shape}")
    if int(stereo_float.shape[1]) < n_fft:
        stereo_float = np.pad(stereo_float, ((0, 0), (0, n_fft - int(stereo_float.shape[1]))))
    window = torch.hann_window(n_fft, dtype=torch.float32)
    window_energy = float(window.square().sum().item())
    stereo_spectrum = torch.stft(
        torch.from_numpy(stereo_float),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    left_real = stereo_spectrum[0].real.numpy()
    left_imag = stereo_spectrum[0].imag.numpy()
    right_real = stereo_spectrum[1].real.numpy()
    right_imag = stereo_spectrum[1].imag.numpy()
    left_power = (left_real * left_real + left_imag * left_imag) / max(window_energy, 1e-8)
    right_power = (right_real * right_real + right_imag * right_imag) / max(window_energy, 1e-8)
    cross_real_power = (left_real * right_real + left_imag * right_imag) / max(window_energy, 1e-8)

    left_pitch = harmonic_filterbank @ left_power
    right_pitch = harmonic_filterbank @ right_power
    cross_pitch = harmonic_filterbank @ cross_real_power
    num_frames = int(left_pitch.shape[1])

    def _decode_zone_labels(
        unary_scores: np.ndarray,
        neighbor_edges: list[list[tuple[int, float]]],
        *,
        allow_center: bool,
    ) -> np.ndarray:
        """旋律・和音グラフ上の平均場ラベル伝播（8 反復）で時間的に安定したゾーン列 {0:L, 1:C, 2:R} を求める。"""
        effective_unary = unary_scores.copy()
        if not allow_center:
            effective_unary[:, 1] = -1e6
        beliefs = effective_unary.copy()
        for _ in range(8):
            shifted = beliefs - beliefs.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted * 1.5)
            probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
            next_beliefs = effective_unary.copy()
            for node_index in range(effective_unary.shape[0]):
                accumulated = np.zeros(3, dtype=np.float64)
                for neighbor_index, edge_weight in neighbor_edges[node_index]:
                    accumulated += edge_weight * probabilities[neighbor_index]
                next_beliefs[node_index] = effective_unary[node_index] + 0.85 * accumulated
            if not allow_center:
                next_beliefs[:, 1] = -1e6
            beliefs = next_beliefs
        return beliefs.argmax(axis=1)

    # 1. 各トラックのノート別パン・位相コヒーレンスを計測し、グラフ平滑化で安定した L/C/R 分離候補を作る。
    split_plans: list[
        tuple[float, int, np.ndarray, np.ndarray, list[list[tuple[int, float]]], list[pretty_midi.Note], bool, int, int]
    ] = []
    for instrument_index, instrument in active_instruments:
        ordered_notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch, note.end))
        note_count = len(ordered_notes)
        raw_pans = np.empty(note_count, dtype=np.float64)
        raw_coherences = np.empty(note_count, dtype=np.float64)
        for position, note in enumerate(ordered_notes):
            start_frame = max(0, min(num_frames - 1, int(math.floor(float(note.start) * frames_per_second))))
            end_frame = min(num_frames, max(start_frame + 1, int(math.ceil(float(note.end) * frames_per_second))))
            clamped_pitch = max(pitch_min, min(pitch_max, int(note.pitch)))
            pitch_row = clamped_pitch - pitch_min
            left_energy = float(np.mean(left_pitch[pitch_row, start_frame:end_frame]))
            right_energy = float(np.mean(right_pitch[pitch_row, start_frame:end_frame]))
            cross_energy = float(np.mean(cross_pitch[pitch_row, start_frame:end_frame]))
            raw_pans[position] = float(
                _energy_pair_to_pan_cc10(
                    left_energy,
                    right_energy,
                    min_pan=0,
                    max_pan=127,
                )
            )
            raw_coherences[position] = float((2.0 * cross_energy) / max(left_energy + right_energy, 1e-12))

        starts = np.asarray([float(note.start) for note in ordered_notes], dtype=np.float64)
        pitches = np.asarray([int(note.pitch) for note in ordered_notes], dtype=np.int32)

        # ノート単位の単項スコア: L/R は振幅パン、C は中央パンかつ高位相コヒーレンスを必須条件にする。
        unary_scores = np.zeros((note_count, 3), dtype=np.float64)
        is_clear_center = np.zeros(note_count, dtype=np.bool_)
        is_clear_side = np.zeros(note_count, dtype=np.bool_)
        for position in range(note_count):
            pan_value = raw_pans[position]
            coherence_value = raw_coherences[position]
            normalized_side = (pan_value - 64.0) / 36.0
            unary_scores[position, 0] = -normalized_side
            unary_scores[position, 2] = normalized_side

            center_pan_fit = max(0.0, 1.0 - abs(pan_value - 64.0) / 26.0)
            if coherence_value >= float(min_center_coherence) and center_pan_fit > 0.0:
                unary_scores[position, 1] = 1.75 * center_pan_fit * min(
                    1.6,
                    (coherence_value - 0.32) / 0.35,
                )
                if coherence_value >= 0.65 and center_pan_fit >= 0.45:
                    is_clear_center[position] = True
            else:
                unary_scores[position, 1] = -1.35

            if (pan_value <= 34.0 or pan_value >= 94.0) and coherence_value < float(min_center_coherence):
                is_clear_side[position] = True

        # 旋律連続（同音高・近接音高の連続フレーズ）および同一和音（35ms 以内）に結合エッジを張る。
        # 明確な中央ソロとハードサイド間は常に切り離し、同時和音ではハード逆サイド同士を結合しない。
        # 一方で時間的に連続する旋律フレーズ (delta_seconds > 0.035) は単発のパン誤推定を吸収するため結合する。
        neighbor_edges: list[list[tuple[int, float]]] = [[] for _ in range(note_count)]
        for first_index in range(note_count):
            for second_index in range(first_index + 1, min(note_count, first_index + 25)):
                delta_seconds = starts[second_index] - starts[first_index]
                if delta_seconds > 0.65:
                    break
                if (is_clear_center[first_index] and is_clear_side[second_index]) or (
                    is_clear_side[first_index] and is_clear_center[second_index]
                ):
                    continue

                pitch_gap = abs(int(pitches[second_index]) - int(pitches[first_index]))
                if delta_seconds <= 0.035:
                    if (raw_pans[first_index] <= 34.0 and raw_pans[second_index] >= 94.0) or (
                        raw_pans[second_index] <= 34.0 and raw_pans[first_index] >= 94.0
                    ):
                        continue
                    edge_weight = 0.70 if pitch_gap <= 12 else 0.25
                else:
                    if pitch_gap == 0:
                        edge_weight = 2.2 * math.exp(-delta_seconds / 0.45)
                    elif pitch_gap <= 5:
                        edge_weight = 1.5 * math.exp(-delta_seconds / 0.35) * math.exp(-float(pitch_gap) / 5.0)
                    elif pitch_gap <= 9 and delta_seconds <= 0.25:
                        edge_weight = 0.60 * math.exp(-delta_seconds / 0.25)
                    else:
                        continue
                neighbor_edges[first_index].append((second_index, edge_weight))
                neighbor_edges[second_index].append((first_index, edge_weight))

        decoded_labels = _decode_zone_labels(unary_scores, neighbor_edges, allow_center=True)
        center_mask = decoded_labels == 1
        center_count = int(center_mask.sum())
        include_center = (
            center_count >= int(min_center_notes)
            and (float(center_count) / float(note_count)) >= float(min_center_ratio)
        )
        if not include_center:
            decoded_labels = _decode_zone_labels(unary_scores, neighbor_edges, allow_center=False)

        left_mask = decoded_labels == 0
        right_mask = decoded_labels == 2
        left_count = int(left_mask.sum())
        right_count = int(right_mask.sum())
        if min(left_count, right_count) < int(min_side_notes):
            continue
        if (float(min(left_count, right_count)) / float(note_count)) < float(min_side_ratio):
            continue
        if (float(left_count + right_count) / float(note_count)) < float(min_outer_ratio):
            continue

        # 元パン分布でも明確に左右へ分かれているか確認し、中央寄りの単一楽器の誤分割を防ぐ。
        raw_outer_count = int(((raw_pans <= float(left_pan_upper)) | (raw_pans >= float(right_pan_lower))).sum())
        if (float(raw_outer_count) / float(note_count)) < float(min_outer_ratio) * 0.85:
            continue

        left_median = int(np.clip(round(float(np.median(raw_pans[left_mask]))), effective_config.min_pan, 54))
        right_median = int(np.clip(round(float(np.median(raw_pans[right_mask]))), 74, effective_config.max_pan))
        pan_gap = float(right_median - left_median)
        if pan_gap < float(min_side_pan_gap):
            continue

        priority = float(note_count) * pan_gap
        split_plans.append(
            (
                priority,
                instrument_index,
                raw_pans,
                unary_scores,
                neighbor_edges,
                ordered_notes,
                include_center,
                left_median,
                right_median,
            )
        )

    if not split_plans:
        return []

    # 2. 優先度（ノート数 × パン幅）順に MIDI 16ch 予算の範囲内でサブトラックへ分割する。
    split_plans.sort(key=lambda item: -item[0])
    remaining_budget = int(max_extra_tracks)
    replacements: dict[int, list[pretty_midi.Instrument]] = {}
    split_reports: list[dict[str, Any]] = []

    for (
        _priority,
        instrument_index,
        raw_pans,
        unary_scores,
        neighbor_edges,
        ordered_notes,
        include_center,
        left_median,
        right_median,
    ) in split_plans:
        needed_extra = 2 if include_center else 1
        if needed_extra > remaining_budget:
            if remaining_budget >= 1:
                include_center = False
                needed_extra = 1
            else:
                break

        final_labels = _decode_zone_labels(unary_scores, neighbor_edges, allow_center=include_center)
        source = midi_object.instruments[instrument_index]
        base_name = str(source.name)
        left_notes: list[pretty_midi.Note] = []
        center_notes: list[pretty_midi.Note] = []
        right_notes: list[pretty_midi.Note] = []

        for note, label_value in zip(ordered_notes, final_labels.tolist()):
            if label_value == 0:
                left_notes.append(note)
            elif label_value == 1 and include_center:
                center_notes.append(note)
            else:
                right_notes.append(note)

        # 万が一どこかのバケットが空になったら安全のため分割を見送る。
        buckets: list[tuple[str, int, list[pretty_midi.Note]]] = [
            ("L", left_median, left_notes),
        ]
        if include_center and center_notes:
            center_pans = [
                pan_value
                for pan_value, label_value in zip(raw_pans.tolist(), final_labels.tolist())
                if label_value == 1
            ]
            center_median = int(np.clip(round(float(np.median(center_pans))), 54, 74))
            buckets.append(("C", center_median, center_notes))
        buckets.append(("R", right_median, right_notes))
        if any(not bucket_notes for _label, _pan, bucket_notes in buckets):
            continue

        new_subtracks: list[pretty_midi.Instrument] = []
        for label, pan_value, bucket_notes in buckets:
            subtrack = pretty_midi.Instrument(
                program=int(source.program),
                is_drum=False,
                name=f"{base_name} ({label})",
            )
            subtrack.notes = bucket_notes
            subtrack.control_changes = [
                pretty_midi.ControlChange(number=int(control.number), value=int(control.value), time=float(control.time))
                for control in source.control_changes
                if int(control.number) != 10
            ]
            subtrack.control_changes.append(pretty_midi.ControlChange(number=10, value=int(pan_value), time=0.0))
            subtrack.control_changes.sort(key=lambda control: (control.time, control.number))
            new_subtracks.append(subtrack)

        replacements[instrument_index] = new_subtracks
        remaining_budget -= len(new_subtracks) - 1
        split_reports.append(
            {
                "stem_name": stem_name,
                "source_name": base_name,
                "program_number": int(source.program),
                "subtracks": [
                    {
                        "name": subtrack.name,
                        "note_count": len(subtrack.notes),
                        "pan": next(
                            control.value for control in subtrack.control_changes if control.number == 10
                        ),
                    }
                    for subtrack in new_subtracks
                ],
            }
        )

    if replacements:
        rebuilt_instruments: list[pretty_midi.Instrument] = []
        for index, instrument in enumerate(midi_object.instruments):
            if index in replacements:
                rebuilt_instruments.extend(replacements[index])
            else:
                rebuilt_instruments.append(instrument)
        midi_object.instruments = rebuilt_instruments

    return split_reports


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
            # ステレオ分割済みのサブトラック (" (L)", " (C)", " (R)") は分割時に決定した固定パンをそのまま保つ。
            is_stereo_split_subtrack = str(instrument.name).endswith((" (L)", " (C)", " (R)"))
            existing_pan = next((int(control.value) for control in instrument.control_changes if int(control.number) == 10), None)
            if is_stereo_split_subtrack and existing_pan is not None:
                pan_events = [pretty_midi.ControlChange(number=10, value=int(existing_pan), time=0.0)]
                pan_info = {
                    "base_pan": int(existing_pan),
                    "pan_events": 1,
                    "pan_min": int(existing_pan),
                    "pan_max": int(existing_pan),
                    "center_locked": False,
                }
            else:
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
