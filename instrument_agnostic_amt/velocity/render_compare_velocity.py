"""Render & Compare（閉ループ最適化）による MIDI Velocity およびトラック別 CC#7 最適化。

選定された GM プログラムと対象 SoundFont を前提に、現在の MIDI を実際に
FluidSynth でレンダリングして元ステム音声と比較し、以下の 2 階層を分離して反復最適化する。

1. マクロ（トラック別 CC#7 = ミックスバランス）:
   ITU-R BS.1770 準拠の K-weighting（聴感ラウドネス補正）を掛けた波形上で、
   元ステムとレンダリングステムのアクティブ区間ラウドネス (LUFS 相当) を直接比較する。
   これによりサブベースの物理エネルギー過大評価を防ぎ、ボーカル・ドラム・ベース・
   ギター・ピアノ・その他が「耳で聴いた原曲の音量バランス」に一致するよう CC#7 を決める。

2. ミクロ（ノート別 Velocity = フレーズ抑揚・強弱）:
   トラック内では平均音量を中央（Velocity 85〜95 付近）に保ちつつ、楽器の性質に応じた
   計測法でノートごとの相対強弱を求める。
   - 単音系 (vocals / bass / 同時発音が少ないトラック):
     ビブラートや母音変化に影響されない「ノート区間の K-weighted 広帯域ラウドネス」を使用。
   - 和音系 (piano / guitar / other の和音トラック):
     安定した「広帯域フレームラウドネス（フレーズ抑揚）」に、「同時和音内のピッチ別相対差」
     を穏やかに合成して、時間方向のガタつきを抑えつつ和音内バランスを再現する。
   - ドラム (drums):
     GM ドラムピッチごとの 5 帯域（Low / Low-Mid / Mid / High / Full）で同時打撃を分離計測。

大容量の SoundFont ロード回数を抑えるため、各イテレーションで全ステムを 1 本の
結合 MIDI に直列パックし、FluidSynth を 1 イテレーションあたり 1 回だけ起動する。
"""

from __future__ import annotations

import copy
import math
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import scipy.signal
import soundfile as sf
import torch
import torchaudio.functional as audio_functional

from ..soundfont import (
    FluidSynthRenderOptions,
    RENDER_COMPARE_SPEED_PRESETS,
    default_parallel_render_workers,
    render_midi_batch_to_stereo_arrays,
    render_midi_to_stereo_array,
    render_midi_with_fluidsynth,
    resolve_signal_processing_device,
    write_packed_smf_type0,
)

# ドラム専用チャンネル (9) を除いた 15 本のメロディチャンネル（MIDI 16ch 制約準拠）。
_MELODIC_MIDI_CHANNELS = tuple(channel for channel in range(16) if channel != 9)
_PACKED_TICKS_PER_BEAT = 480
_PACKED_TEMPO_MICROSECONDS = 500_000  # 120 BPM -> 1 秒 = 960 tick
_PACKED_TICKS_PER_SECOND = float(_PACKED_TICKS_PER_BEAT) * 1_000_000.0 / float(_PACKED_TEMPO_MICROSECONDS)

# GM ドラムピッチ -> 担当周波数帯域名（同時打撃のキック・スネア・ハイハットを分離計測するため）。
_DRUM_LOW_PITCHES = frozenset({35, 36, 41, 43})
_DRUM_LOW_MID_PITCHES = frozenset({45, 47, 48, 50, 60, 61, 62, 63, 64, 65, 66})
_DRUM_MID_PITCHES = frozenset({37, 38, 39, 40, 56, 75, 76, 77})
_DRUM_HIGH_PITCHES = frozenset({42, 44, 46, 49, 51, 52, 53, 54, 55, 57, 59, 69, 70, 80, 81, 82})

_PackedMidiRenderer = Callable[[Path, Path, int], None]


@dataclass(frozen=True)
class RenderCompareVelocityConfig:
    """閉ループ・ベロシティ＆CC#7 最適化の設定。"""

    soundfont_path: Path
    fluidsynth_executable: Path | None = None
    sample_rate: int = 22_050
    num_iterations: int = 3
    stem_gap_seconds: float = 1.0
    render_gain: float = 0.5
    disable_reverb: bool = False
    disable_chorus: bool = False
    use_input_velocity_as_init: bool = False
    # GM SoundFont (SF2) は Velocity が低いと LPF が閉じてこもった音になるため、
    # 標準的なポップス・ロックの抜けを保てる 104 付近をトラック中央の基準にする。
    initial_velocity: int = 104
    initial_volume_cc7: int = 105
    target_reference_volume_cc7: int = 112
    min_volume_cc7: int = 36
    max_volume_cc7: int = 127
    min_velocity: int = 48
    max_velocity: int = 127
    vocal_min_velocity: int = 72
    bass_min_velocity: int = 70
    drum_min_velocity: int = 44
    velocity_curve_range_db: float = 36.0
    volume_cc7_curve_range_db: float = 40.0
    velocity_damping: float = 0.70
    volume_cc7_damping: float = 0.85
    # トラック内の相対ダイナミクスをどれくらい忠実に Velocity へ乗せるか (0.0=平坦, 1.0=生誤差そのまま)。
    # 分離アーティファクトやわずかなタイミングズレによる過度な乱高下を抑えるため 0.70 を既定にする。
    dynamics_compression: float = 0.70
    vocal_dynamics_compression: float = 0.55
    chord_voicing_strength: float = 0.50
    optimize_volume_cc7: bool = True
    max_step_error_db: float = 14.0
    max_within_stem_track_offset_db: float = 14.0
    local_smoothing_strength: float = 0.30
    vocal_smoothing_strength: float = 0.48
    # 分離器のマスク減衰と GM マスキングを補償するため、伴奏が大きい区間ほどボーカルを持ち上げ、
    # 伴奏が静かな独唱区間では突出を抑える伴奏連動補償係数 (0.0 で無効)。
    vocal_accompaniment_compensation: float = 0.35
    # パック MIDI 末尾に全ステム同時再生スロットを追加し、元フルミックスとの時間別ラウドネス誤差を
    # フィードバックして Aメロ〜サビのフルミックス整合性を取る (0.0 で無効)。
    full_mix_feedback_weight: float = 0.45
    full_mix_window_seconds: float = 0.25
    # ステレオ分離ステムからの左右パン (CC#10) 抽出、およびダブルギター等の (L)/(C)/(R) 自動分割。
    # 持続系楽器の動的強弱 (CC#11) は SoundFont のアタックと干渉しやすいため既定では無効（オプトイン）。
    optimize_pan_cc10: bool = True
    split_stereo_pan_tracks: bool = True
    optimize_expression_cc11: bool = False
    max_melodic_instruments: int = 15
    evaluate_final: bool = True
    parallel_render_workers: int | None = None

    @classmethod
    def for_speed_preset(
        cls,
        soundfont_path: Path | str,
        *,
        speed: str = "balanced",
        fluidsynth_executable: Path | str | None = None,
        max_melodic_instruments: int = 15,
        parallel_render_workers: int | None = None,
    ) -> "RenderCompareVelocityConfig":
        """速度プリセット (`fast` / `balanced` / `quality`) からベロシティ最適化設定を構築する。"""
        normalized_speed = str(speed).strip().lower()
        if normalized_speed not in RENDER_COMPARE_SPEED_PRESETS:
            raise ValueError(f"speed must be one of {RENDER_COMPARE_SPEED_PRESETS}, got {speed!r}")
        resolved_executable = Path(fluidsynth_executable) if fluidsynth_executable is not None else None
        if normalized_speed == "fast":
            return cls(
                soundfont_path=Path(soundfont_path),
                fluidsynth_executable=resolved_executable,
                sample_rate=16_000,
                num_iterations=2,
                disable_reverb=True,
                disable_chorus=True,
                evaluate_final=False,
                full_mix_feedback_weight=0.0,
                max_melodic_instruments=int(max_melodic_instruments),
                parallel_render_workers=parallel_render_workers,
            )
        if normalized_speed == "balanced":
            # 16 kHz 並列 + 3 回反復 + リバーブ/コーラス維持 + 最終評価パス省略により、
            # quality とほぼ同等の最終 MAE (差 ~0.08 dB) を保ちながら約 2.5 倍高速化する。
            return cls(
                soundfont_path=Path(soundfont_path),
                fluidsynth_executable=resolved_executable,
                sample_rate=16_000,
                num_iterations=3,
                disable_reverb=False,
                disable_chorus=False,
                evaluate_final=False,
                max_melodic_instruments=int(max_melodic_instruments),
                parallel_render_workers=parallel_render_workers,
            )
        return cls(
            soundfont_path=Path(soundfont_path),
            fluidsynth_executable=resolved_executable,
            sample_rate=22_050,
            num_iterations=3,
            disable_reverb=False,
            disable_chorus=False,
            evaluate_final=True,
            max_melodic_instruments=int(max_melodic_instruments),
            parallel_render_workers=parallel_render_workers,
        )

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if self.stem_gap_seconds < 0.0:
            raise ValueError("stem_gap_seconds must be nonnegative")
        if self.render_gain <= 0.0:
            raise ValueError("render_gain must be positive")
        if not 1 <= self.min_velocity <= self.max_velocity <= 127:
            raise ValueError("min_velocity and max_velocity must satisfy 1 <= min <= max <= 127")
        if not 1 <= self.min_volume_cc7 <= self.max_volume_cc7 <= 127:
            raise ValueError("min_volume_cc7 and max_volume_cc7 must satisfy 1 <= min <= max <= 127")
        if not self.min_volume_cc7 <= self.target_reference_volume_cc7 <= self.max_volume_cc7:
            raise ValueError("target_reference_volume_cc7 must lie within [min_volume_cc7, max_volume_cc7]")
        if self.velocity_curve_range_db <= 0.0 or self.volume_cc7_curve_range_db <= 0.0:
            raise ValueError("curve range in dB must be positive")
        if not 0.0 < self.velocity_damping <= 1.0 or not 0.0 < self.volume_cc7_damping <= 1.0:
            raise ValueError("damping factors must lie within (0, 1]")
        if not 0.0 <= self.local_smoothing_strength <= 1.0 or not 0.0 <= self.vocal_smoothing_strength <= 1.0:
            raise ValueError("smoothing strengths must lie within [0, 1]")
        if self.vocal_accompaniment_compensation < 0.0 or self.full_mix_feedback_weight < 0.0:
            raise ValueError("compensation and full_mix_feedback_weight must be nonnegative")
        if self.full_mix_window_seconds <= 0.0:
            raise ValueError("full_mix_window_seconds must be positive")


@dataclass
class _TrackState:
    """最適化対象の 1 トラック分の状態（MIDI トラック参照と現在の CC#7 / CC#10 / 空間ゾーン）。"""

    track_key: int
    stem_name: str
    instrument_index: int
    program_number: int
    is_drum: bool
    instrument_name: str
    instrument: pretty_midi.Instrument
    volume_cc7: float
    is_monophonic: bool
    pan_cc10: int = 64
    spatial_zone: str = "mono"  # "mono" | "left" | "center" | "right"


@dataclass
class _NoteState:
    """最適化対象の 1 ノート分の状態。"""

    note_id: int
    track_key: int
    stem_name: str
    note: pretty_midi.Note
    start_seconds: float
    end_seconds: float
    pitch: int
    is_drum: bool
    velocity: float


@dataclass(frozen=True)
class _StemSlot:
    stem_name: str
    slot_start_seconds: float
    slot_duration_seconds: float


@dataclass(frozen=True)
class _StemAcousticMeasurement:
    """1 ステムの K-weighted マクロ音量と、各ノートのミクロ音量をまとめた計測結果。"""

    stem_active_lufs_db: float
    track_active_lufs_db: dict[int, float]
    note_levels_db: dict[int, float]


def _load_stereo_audio(audio_path: Path, *, target_sample_rate: int) -> np.ndarray:
    """音声を指定サンプルレートの 2 チャンネル [2, T] float32 ステレオ波形として読み込む。"""
    waveform_numpy, source_sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if waveform_numpy.shape[1] == 1:
        stereo_numpy = np.repeat(waveform_numpy.T, 2, axis=0)
    else:
        stereo_numpy = waveform_numpy[:, :2].T.copy()
    if int(source_sample_rate) != int(target_sample_rate):
        resampled = audio_functional.resample(
            torch.from_numpy(stereo_numpy),
            int(source_sample_rate),
            int(target_sample_rate),
        )
        return resampled.numpy().astype(np.float32, copy=False)
    return stereo_numpy.astype(np.float32, copy=False)


def _load_mono_audio(audio_path: Path, *, target_sample_rate: int) -> np.ndarray:
    """音声を指定サンプルレートの 1 次元 float32 モノラル波形として読み込む。"""
    return _load_stereo_audio(audio_path, target_sample_rate=target_sample_rate).mean(axis=0).astype(np.float32, copy=False)


_K_WEIGHTING_SOS_CACHE: dict[int, np.ndarray] = {}


def _get_k_weighting_sos(sample_rate: int) -> np.ndarray:
    """ITU-R BS.1770 K-weighting の 2 段 SOS フィルタ係数をサンプルレートごとにキャッシュする。"""
    cached = _K_WEIGHTING_SOS_CACHE.get(sample_rate)
    if cached is not None:
        return cached

    nyquist = float(sample_rate) * 0.5
    sections: list[np.ndarray] = []

    # 1. 高域通過フィルタ (RLB 相当: 100 Hz 2 次 Butterworth)
    cutoff_hz = min(100.0, nyquist * 0.4)
    if cutoff_hz > 5.0:
        sections.append(scipy.signal.butter(2, cutoff_hz / nyquist, btype="high", output="sos"))

    # 2. 高域シェルビングフィルタ (+4 dB @ 1.5 kHz, RBJ Audio EQ Cookbook)
    shelf_hz = min(1500.0, nyquist * 0.45)
    if shelf_hz > cutoff_hz * 1.5 and nyquist > 2000.0:
        gain_db = 4.0
        amplitude = 10.0 ** (gain_db / 40.0)
        omega = 2.0 * math.pi * shelf_hz / float(sample_rate)
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)
        alpha = sin_omega / (2.0 * 0.70710678)
        two_sqrt_a_alpha = 2.0 * math.sqrt(amplitude) * alpha

        b0 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega + two_sqrt_a_alpha)
        b1 = -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cos_omega)
        b2 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega - two_sqrt_a_alpha)
        a0 = (amplitude + 1.0) - (amplitude - 1.0) * cos_omega + two_sqrt_a_alpha
        a1 = 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cos_omega)
        a2 = (amplitude + 1.0) - (amplitude - 1.0) * cos_omega - two_sqrt_a_alpha
        sections.append(np.asarray([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64))

    sos = np.vstack(sections) if sections else np.zeros((0, 6), dtype=np.float64)
    _K_WEIGHTING_SOS_CACHE[sample_rate] = sos
    return sos


def _apply_k_weighting(mono_audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
    """ITU-R BS.1770 に基づく聴感ラウドネス補正 (K-weighting) を適用する。

    1. 2 次ハイパス (~100 Hz): サブベースの物理エネルギーが聴感以上に CC#7 を押し上げるのを防ぐ。
    2. 高域シェルビング (~1.5 kHz, +4 dB): 人間の耳が敏感なボーカル・ギター・ブラス帯域を正しく評価する。
    """
    signal = np.asarray(mono_audio, dtype=np.float32)
    if signal.size < 16:
        return signal
    sos = _get_k_weighting_sos(int(sample_rate))
    if sos.shape[0] == 0:
        return signal
    return scipy.signal.sosfilt(sos.astype(np.float32), signal).astype(np.float32, copy=False)


def _batch_segment_percentile_1d(
    series: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    quantile_fraction: float,
    *,
    row_indices_2d: np.ndarray | None = None,
) -> np.ndarray:
    """複数ノート区間 `[start_frames[i], end_frames[i])` の線形補間パーセンタイルを一括ベクトル計算する。

    各ノート区間長は最大数十フレーム程度なので `[N, max_length]` 配列へパディング（無効枠は `+inf`）して
    単一の C レベル `np.sort(..., axis=1)` で処理し、単発 `_fast_percentile_1d` とビット同一の結果を得る。
    """
    note_count = int(start_frames.size)
    if note_count == 0:
        return np.empty(0, dtype=np.float64)

    lengths = np.maximum(1, end_frames - start_frames)
    max_length = int(lengths.max())
    max_valid_frame = int(series.shape[-1]) - 1
    offsets = np.arange(max_length, dtype=np.int64)[None, :]
    frame_indices = np.minimum(start_frames[:, None] + offsets, max_valid_frame)

    if row_indices_2d is None:
        gathered = series[frame_indices].astype(np.float64, copy=True)
    else:
        gathered = series[row_indices_2d[:, None], frame_indices].astype(np.float64, copy=True)

    valid_mask = offsets < lengths[:, None]
    gathered[~valid_mask] = np.inf
    gathered.sort(axis=1)

    positions = float(quantile_fraction) * (lengths - 1).astype(np.float64)
    lower_indices = positions.astype(np.int64)
    upper_indices = np.minimum(lengths - 1, lower_indices + 1)
    weights = positions - lower_indices.astype(np.float64)
    note_rows = np.arange(note_count, dtype=np.int64)
    return gathered[note_rows, lower_indices] * (1.0 - weights) + gathered[note_rows, upper_indices] * weights


def _fast_percentile_1d(values: np.ndarray, quantile_fraction: float) -> float:
    """小配列に対する高速 1 次元線形補間パーセンタイル（`np.percentile` の呼び出しオーバーヘッドを回避）。"""
    length = int(values.size)
    if length == 0:
        return 1e-12
    if length == 1:
        return float(values[0])
    ordered = np.sort(values)
    position = float(quantile_fraction) * float(length - 1)
    lower_index = int(position)
    upper_index = min(length - 1, lower_index + 1)
    weight = position - float(lower_index)
    return float(ordered[lower_index]) * (1.0 - weight) + float(ordered[upper_index]) * weight


def _render_packed_midi_with_fluidsynth(
    midi_path: Path,
    wav_path: Path,
    sample_rate: int,
    *,
    config: RenderCompareVelocityConfig,
) -> None:
    """パック済み MIDI を FluidSynth で一括レンダリングする（外部プロセス境界）。"""
    render_midi_with_fluidsynth(
        midi_path,
        wav_path,
        options=FluidSynthRenderOptions(
            soundfont_path=Path(config.soundfont_path),
            sample_rate=int(sample_rate),
            fluidsynth_executable=config.fluidsynth_executable,
            render_gain=float(config.render_gain),
            disable_reverb=bool(config.disable_reverb),
            disable_chorus=bool(config.disable_chorus),
            polyphony=96,
        ),
    )


def _build_harmonic_pitch_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    pitch_min: int = 21,
    pitch_max: int = 108,
) -> tuple[np.ndarray, np.ndarray]:
    """K-weighted STFT パワーから各 MIDI ピッチの倍音エネルギー [88, num_bins] と
    ドラム用 5 帯域フィルタバンク [5, num_bins] を事前計算する。

    注意: 以前は行和で正規化していたため、帯域幅の狭い低音と広い中高音でエネルギー密度が
    大きくズレるバイアスがあった。ここでは倍音ごとのピークが 1.0 となる積分和（ビンの総和）で
    集計し、同一和音内のピッチ間比較でも周波数帯域バイアスを抑える。
    """
    num_bins = n_fft // 2 + 1
    frequencies = np.linspace(0.0, float(sample_rate) * 0.5, num_bins, dtype=np.float64)
    nyquist = float(sample_rate) * 0.5
    bin_spacing = float(sample_rate) / float(n_fft)

    num_pitches = pitch_max - pitch_min + 1
    pitch_filterbank = np.zeros((num_pitches, num_bins), dtype=np.float32)
    harmonic_weights = ((1, 1.0), (2, 0.55), (3, 0.30), (4, 0.18))

    for pitch_offset in range(num_pitches):
        midi_pitch = pitch_min + pitch_offset
        fundamental_hz = 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))
        row = np.zeros(num_bins, dtype=np.float64)
        weight_total = 0.0
        for harmonic_number, harmonic_weight in harmonic_weights:
            center_hz = fundamental_hz * float(harmonic_number)
            if center_hz >= nyquist * 0.96:
                continue
            # ±0.6 半音程度の幅を持たせ、わずかなデチューンやビブラートを吸収する。
            sigma_hz = max(center_hz * 0.035, bin_spacing * 1.2)
            gaussian = np.exp(-0.5 * np.square((frequencies - center_hz) / sigma_hz))
            gaussian[gaussian < 1e-3] = 0.0
            row += harmonic_weight * gaussian
            weight_total += harmonic_weight
        if weight_total > 1e-8:
            # ビン数ではなく倍音重み合計で割ることで、総帯域エネルギーとして比較できる。
            pitch_filterbank[pitch_offset] = (row / weight_total).astype(np.float32)

    band_ranges_hz = (
        (30.0, 180.0),
        (100.0, 600.0),
        (180.0, 2500.0),
        (2500.0, min(10_000.0, nyquist * 0.96)),
        (60.0, min(8_000.0, nyquist * 0.96)),
    )
    drum_filterbank = np.zeros((len(band_ranges_hz), num_bins), dtype=np.float32)
    for band_index, (low_hz, high_hz) in enumerate(band_ranges_hz):
        effective_low = min(low_hz, nyquist * 0.8)
        effective_high = max(effective_low + bin_spacing, min(high_hz, nyquist))
        mask = (frequencies >= effective_low) & (frequencies <= effective_high)
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(frequencies - 0.5 * (effective_low + effective_high))))
            mask[nearest] = True
        drum_filterbank[band_index] = mask.astype(np.float32)

    return pitch_filterbank, drum_filterbank


def _drum_band_index(pitch: int) -> int:
    if pitch in _DRUM_LOW_PITCHES:
        return 0
    if pitch in _DRUM_LOW_MID_PITCHES:
        return 1
    if pitch in _DRUM_MID_PITCHES:
        return 2
    if pitch in _DRUM_HIGH_PITCHES:
        return 3
    return 4


def _is_mostly_monophonic(track_notes: list[_NoteState]) -> bool:
    """トラック内の同時発音率から、単音系（ボーカル・ベース・リード）か和音系かを判定する。"""
    if len(track_notes) <= 1:
        return True
    ordered = sorted(track_notes, key=lambda item: (item.start_seconds, item.end_seconds))
    overlap_count = 0
    for index in range(1, len(ordered)):
        # 50 ms 以上の明確な重なりがあるノートを和音とみなす（レガートの微小重なりは無視）。
        if ordered[index].start_seconds < ordered[index - 1].end_seconds - 0.05:
            overlap_count += 1
    return (float(overlap_count) / float(len(ordered))) < 0.15


def _measure_stem_acoustics(
    mono_audio: np.ndarray,
    stem_tracks: list[_TrackState],
    stem_notes: list[_NoteState],
    notes_by_track: dict[int, list[_NoteState]],
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    pitch_filterbank: np.ndarray,
    drum_filterbank: np.ndarray,
    chord_voicing_strength: float,
    stereo_audio: np.ndarray | None = None,
    pitch_min: int = 21,
    pitch_max: int = 108,
) -> _StemAcousticMeasurement:
    """1 ステムの K-weighted マクロ音量 (LUFS 相当) と各ノートのミクロ音量を計測する。"""
    if not stem_notes:
        return _StemAcousticMeasurement(stem_active_lufs_db=-90.0, track_active_lufs_db={}, note_levels_db={})

    # 1. K-weighting を掛けた波形でフレーム別広帯域パワーと STFT パワーを算出する（CUDA 利用可能時は GPU 計算）。
    signal_device = resolve_signal_processing_device()
    has_drum = any(item.is_drum for item in stem_notes)
    has_melodic = any(not item.is_drum for item in stem_notes)
    needed_zones = {track_state.spatial_zone for track_state in stem_tracks if track_state.spatial_zone != "mono"}
    has_spatial = bool(
        needed_zones and stereo_audio is not None and stereo_audio.ndim == 2 and int(stereo_audio.shape[0]) == 2
    )

    # K-weighting は線形フィルタなので、ステレオ空間計測を行うステムでは L/R を先にフィルタし、
    # mono = 0.5 * (L + R) の K-weighted 波形を流用して 3 回目の sosfilt を省略する。
    left_k_precomputed: np.ndarray | None = None
    right_k_precomputed: np.ndarray | None = None
    if has_spatial and stereo_audio is not None and int(stereo_audio.shape[1]) == int(mono_audio.size):
        left_k_precomputed = _apply_k_weighting(stereo_audio[0], sample_rate=sample_rate)
        right_k_precomputed = _apply_k_weighting(stereo_audio[1], sample_rate=sample_rate)
        k_weighted = (0.5 * (left_k_precomputed + right_k_precomputed)).astype(np.float32, copy=False)
    else:
        k_weighted = _apply_k_weighting(mono_audio, sample_rate=sample_rate)
    if k_weighted.size < n_fft:
        k_weighted = np.pad(k_weighted, (0, n_fft - k_weighted.size))

    with torch.inference_mode():
        window = torch.hann_window(n_fft, dtype=torch.float32, device=signal_device)
        window_energy = float(window.square().sum().item())
        k_wave_tensor = torch.from_numpy(np.ascontiguousarray(k_weighted, dtype=np.float32)).to(signal_device)
        spectrum = torch.stft(
            k_wave_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        # K-weighted STFT パワー（パーセバル正規化により時間領域の二乗平均パワーと一致する）。
        k_power_tensor = (spectrum.real.square() + spectrum.imag.square()) / max(window_energy, 1e-8)
        num_frames = int(k_power_tensor.shape[1])
        frames_per_second = float(sample_rate) / float(hop_length)

        # 広帯域 K-weighted フレームパワー（片側スペクトルなので DC/Nyquist 以外を 2 倍して時間領域と一致）。
        broadband_weights = torch.full((k_power_tensor.shape[0],), 2.0, dtype=torch.float32, device=signal_device)
        broadband_weights[0] = 1.0
        if n_fft % 2 == 0:
            broadband_weights[-1] = 1.0
        broadband_frame_power = ((broadband_weights @ k_power_tensor) / float(n_fft)).cpu().numpy()

        # ドラムは低域キック（60〜100 Hz）の迫力も正確に測るため、生波形の STFT パワーも用意する。
        drum_energy: np.ndarray | None = None
        if has_drum:
            raw_wave = mono_audio.astype(np.float32, copy=False)
            if raw_wave.size < n_fft:
                raw_wave = np.pad(raw_wave, (0, n_fft - raw_wave.size))
            raw_tensor = torch.from_numpy(np.ascontiguousarray(raw_wave, dtype=np.float32)).to(signal_device)
            raw_spectrum = torch.stft(
                raw_tensor,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True,
            )
            raw_power_tensor = (raw_spectrum.real.square() + raw_spectrum.imag.square()) / max(window_energy, 1e-8)
            drum_fb_tensor = torch.from_numpy(np.ascontiguousarray(drum_filterbank, dtype=np.float32)).to(signal_device)
            drum_energy = (drum_fb_tensor @ raw_power_tensor).cpu().numpy()

        pitch_fb_tensor = (
            torch.from_numpy(np.ascontiguousarray(pitch_filterbank, dtype=np.float32)).to(signal_device)
            if (has_melodic or has_spatial)
            else None
        )
        pitch_energy = (pitch_fb_tensor @ k_power_tensor).cpu().numpy() if (has_melodic and pitch_fb_tensor is not None) else None

        # ステレオ分割サブトラック ("left" / "center" / "right") が存在する場合、
        # 元ステムとレンダリングの両方に同一の L/C/R 空間分離マスクを適用し、
        # 左右バッキングと中央ソロの音量・ダイナミクスを独立して計測できるようにする。
        spatial_broadband_by_zone: dict[str, np.ndarray] = {}
        spatial_pitch_by_zone: dict[str, np.ndarray] = {}
        if has_spatial and stereo_audio is not None and pitch_fb_tensor is not None:
            left_k = (
                left_k_precomputed
                if left_k_precomputed is not None
                else _apply_k_weighting(stereo_audio[0], sample_rate=sample_rate)
            )
            right_k = (
                right_k_precomputed
                if right_k_precomputed is not None
                else _apply_k_weighting(stereo_audio[1], sample_rate=sample_rate)
            )
            if left_k.size < n_fft:
                pad_len = n_fft - left_k.size
                left_k = np.pad(left_k, (0, pad_len))
                right_k = np.pad(right_k, (0, pad_len))
            stereo_tensor = torch.from_numpy(np.ascontiguousarray(np.stack((left_k, right_k), axis=0), dtype=np.float32)).to(
                signal_device
            )
            stereo_spectrum = torch.stft(
                stereo_tensor,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True,
            )
            left_real = stereo_spectrum[0].real
            left_imag = stereo_spectrum[0].imag
            right_real = stereo_spectrum[1].real
            right_imag = stereo_spectrum[1].imag
            inv_win_energy = 1.0 / max(window_energy, 1e-8)
            left_power = (left_real.square() + left_imag.square()) * inv_win_energy
            right_power = (right_real.square() + right_imag.square()) * inv_win_energy
            center_power = torch.clamp_min(left_real * right_real + left_imag * right_imag, 0.0) * inv_win_energy

            zone_power_maps: dict[str, torch.Tensor] = {}
            if "left" in needed_zones:
                zone_power_maps["left"] = torch.clamp_min(left_power - center_power, 0.0)
            if "right" in needed_zones:
                zone_power_maps["right"] = torch.clamp_min(right_power - center_power, 0.0)
            if "center" in needed_zones:
                zone_power_maps["center"] = center_power
            for zone_name, zone_power in zone_power_maps.items():
                spatial_broadband_by_zone[zone_name] = ((broadband_weights @ zone_power) / float(n_fft)).cpu().numpy()
                spatial_pitch_by_zone[zone_name] = (pitch_fb_tensor @ zone_power).cpu().numpy()

    # 2. マクロ計測：ステム全体およびトラックごとのアクティブ区間 K-weighted LUFS (dB)。
    stem_active_mask = np.zeros(num_frames, dtype=np.bool_)
    track_active_masks: dict[int, np.ndarray] = {}
    for track_state in stem_tracks:
        mask = np.zeros(num_frames, dtype=np.bool_)
        for note_state in notes_by_track.get(track_state.track_key, []):
            start_frame = max(0, min(num_frames - 1, int(math.floor(note_state.start_seconds * frames_per_second))))
            end_frame = min(num_frames, max(start_frame + 1, int(math.ceil(note_state.end_seconds * frames_per_second))))
            mask[start_frame:end_frame] = True
            stem_active_mask[start_frame:end_frame] = True
        track_active_masks[track_state.track_key] = mask

    if np.any(stem_active_mask):
        active_powers = broadband_frame_power[stem_active_mask]
        # ITU-R BS.1770 の相対ゲート相当：上位有音部のエネルギー平均（極端な無音フレームを除外）。
        gate_floor = float(np.percentile(active_powers, 90)) * (10.0 ** (-20.0 / 10.0))
        gated = active_powers[active_powers >= gate_floor]
        stem_mean_power = float(np.mean(gated)) if gated.size > 0 else float(np.mean(active_powers))
    else:
        stem_mean_power = float(np.mean(broadband_frame_power))
    stem_active_lufs_db = float(10.0 * math.log10(max(stem_mean_power, 1e-12)))

    # 同一ステム内に複数トラックがある場合、他トラックと重ならない単独発音フレームからトラック別 LUFS を測る。
    # ステレオ空間分割されたサブトラック (L/C/R) は対応する空間分離パワー上でアクティブ全区間から直接計測する。
    track_count_per_frame = np.zeros(num_frames, dtype=np.int32)
    for mask in track_active_masks.values():
        track_count_per_frame += mask.astype(np.int32)

    track_active_lufs_db: dict[int, float] = {}
    for track_state in stem_tracks:
        mask = track_active_masks[track_state.track_key]
        if track_state.spatial_zone in spatial_broadband_by_zone and int(mask.sum()) >= 4:
            zone_series = spatial_broadband_by_zone[track_state.spatial_zone]
            common_mask = mask[: zone_series.size]
            zone_powers = zone_series[common_mask]
            zone_gate = float(np.percentile(zone_powers, 90)) * (10.0 ** (-20.0 / 10.0))
            zone_gated = zone_powers[zone_powers >= zone_gate]
            mean_zone_power = float(np.mean(zone_gated)) if zone_gated.size > 0 else float(np.mean(zone_powers))
            track_active_lufs_db[track_state.track_key] = float(10.0 * math.log10(max(mean_zone_power, 1e-12)))
            continue

        solo_mask = mask & (track_count_per_frame == 1)
        if int(solo_mask.sum()) >= 8:
            powers = broadband_frame_power[solo_mask]
            track_active_lufs_db[track_state.track_key] = float(10.0 * math.log10(max(float(np.mean(powers)), 1e-12)))
        elif int(mask.sum()) > 0:
            # 常に他トラックと重なっているトラック（例: コーラス）は、そのトラックのピッチエネルギー比で推定する。
            track_notes = notes_by_track.get(track_state.track_key, [])
            if not track_state.is_drum and pitch_energy is not None and track_notes:
                pitch_samples: list[float] = []
                for note_state in track_notes:
                    clamped_pitch = max(pitch_min, min(pitch_max, int(note_state.pitch)))
                    start_frame = max(0, min(num_frames - 1, int(math.floor(note_state.start_seconds * frames_per_second))))
                    end_frame = min(num_frames, max(start_frame + 1, int(math.ceil(note_state.end_seconds * frames_per_second))))
                    pitch_samples.append(float(np.mean(pitch_energy[clamped_pitch - pitch_min, start_frame:end_frame])))
                track_active_lufs_db[track_state.track_key] = float(
                    10.0 * math.log10(max(float(np.mean(pitch_samples)), 1e-12))
                )
            else:
                track_active_lufs_db[track_state.track_key] = stem_active_lufs_db
        else:
            track_active_lufs_db[track_state.track_key] = stem_active_lufs_db

    # 3. ミクロ計測：トラック内の各ノートの強弱 (dB)。
    note_levels_db: dict[int, float] = {}
    track_lookup = {track_state.track_key: track_state for track_state in stem_tracks}

    # 同時和音（オンセット差 40 ms 以内）をグループ化し、和音内でのピッチ間相対差だけを安全に取り出す。
    for track_state in stem_tracks:
        track_notes = notes_by_track.get(track_state.track_key, [])
        if not track_notes:
            continue

        if track_state.is_drum:
            assert drum_energy is not None
            drum_starts = np.asarray([item.start_seconds for item in track_notes], dtype=np.float64)
            drum_bands = np.asarray([_drum_band_index(item.pitch) for item in track_notes], dtype=np.int64)
            start_frames = np.clip(np.floor(drum_starts * frames_per_second).astype(np.int64), 0, num_frames - 1)
            end_frames = np.minimum(
                num_frames,
                np.maximum(start_frames + 1, np.ceil((drum_starts + 0.075) * frames_per_second).astype(np.int64)),
            )
            onset_powers = _batch_segment_percentile_1d(
                drum_energy,
                start_frames,
                end_frames,
                0.85,
                row_indices_2d=drum_bands,
            )
            onset_dbs = 10.0 * np.log10(np.maximum(onset_powers, 1e-12))
            for note_state, level_db in zip(track_notes, onset_dbs.tolist()):
                note_levels_db[note_state.note_id] = float(level_db)
            continue

        # ステレオ分割サブトラックは対応する空間分離パワー列を使う。
        active_broadband = spatial_broadband_by_zone.get(track_state.spatial_zone, broadband_frame_power)
        active_pitch_energy = spatial_pitch_by_zone.get(track_state.spatial_zone, pitch_energy)
        active_num_frames = int(active_broadband.size)

        # 3-a. 単音系（vocals / bass / 単音リード）:
        #      K-weighted 広帯域パワーをノート区間全体（最大 0.35 秒）で一括ベクトル計測する。
        #      ビブラート・しゃくり・母音（フォルマント）変化でノートごとにガタつかない。
        note_starts = np.asarray([item.start_seconds for item in track_notes], dtype=np.float64)
        note_ends = np.asarray([item.end_seconds for item in track_notes], dtype=np.float64)
        max_measure_seconds = 0.35 if track_state.is_monophonic else 0.22
        measure_durations = np.minimum(np.maximum(0.04, note_ends - note_starts), max_measure_seconds)
        start_frames = np.clip(np.floor(note_starts * frames_per_second).astype(np.int64), 0, active_num_frames - 1)
        end_frames = np.minimum(
            active_num_frames,
            np.maximum(
                start_frames + 1,
                np.ceil((note_starts + measure_durations) * frames_per_second).astype(np.int64),
            ),
        )

        # 上位 75 パーセンタイルを使い、オンセット微小ズレ (20〜40ms) による凹みを吸収する。
        broadband_powers = _batch_segment_percentile_1d(active_broadband, start_frames, end_frames, 0.75)
        broadband_dbs = 10.0 * np.log10(np.maximum(broadband_powers, 1e-12))
        broadband_db_by_note = {
            note_state.note_id: float(level_db)
            for note_state, level_db in zip(track_notes, broadband_dbs.tolist())
        }

        pitch_db_by_note: dict[int, float] = {}
        if active_pitch_energy is not None:
            pitch_rows = np.clip(
                np.asarray([int(item.pitch) for item in track_notes], dtype=np.int64),
                pitch_min,
                pitch_max,
            ) - pitch_min
            pitch_powers = _batch_segment_percentile_1d(
                active_pitch_energy,
                start_frames,
                end_frames,
                0.75,
                row_indices_2d=pitch_rows,
            )
            pitch_dbs = 10.0 * np.log10(np.maximum(pitch_powers, 1e-12))
            pitch_db_by_note = {
                note_state.note_id: float(level_db)
                for note_state, level_db in zip(track_notes, pitch_dbs.tolist())
            }

        if track_state.is_monophonic and len(stem_tracks) == 1:
            for note_state in track_notes:
                note_levels_db[note_state.note_id] = broadband_db_by_note[note_state.note_id]
            continue

        # 3-b. 和音系、または同ステム内に複数トラックが共存するケース:
        #      時間方向のベース強弱は安定した広帯域パワーを使い、同時発音ノート間（45ms 以内）
        #      の相対バランスだけをピッチ倍音エネルギーの差分で付与する。
        ordered_notes = sorted(track_notes, key=lambda item: (item.start_seconds, item.pitch))
        chord_groups: list[list[_NoteState]] = []
        for note_state in ordered_notes:
            if not chord_groups or (note_state.start_seconds - chord_groups[-1][0].start_seconds) > 0.045:
                chord_groups.append([note_state])
            else:
                chord_groups[-1].append(note_state)

        for group in chord_groups:
            if len(group) == 1 or not pitch_db_by_note:
                single = group[0]
                if len(stem_tracks) > 1 and not track_lookup[single.track_key].is_monophonic:
                    note_levels_db[single.note_id] = (
                        0.65 * broadband_db_by_note[single.note_id] + 0.35 * pitch_db_by_note[single.note_id]
                    )
                else:
                    note_levels_db[single.note_id] = broadband_db_by_note[single.note_id]
            else:
                group_pitch_dbs = np.asarray([pitch_db_by_note[item.note_id] for item in group], dtype=np.float64)
                group_pitch_mean = float(group_pitch_dbs.mean())
                for item, pitch_db in zip(group, group_pitch_dbs.tolist()):
                    # 和音内での相対差（±6 dB に制限）だけを広帯域レベルへ加算する。
                    voicing_offset = float(np.clip((pitch_db - group_pitch_mean) * chord_voicing_strength, -6.0, 6.0))
                    note_levels_db[item.note_id] = broadband_db_by_note[item.note_id] + voicing_offset

    return _StemAcousticMeasurement(
        stem_active_lufs_db=stem_active_lufs_db,
        track_active_lufs_db=track_active_lufs_db,
        note_levels_db=note_levels_db,
    )


def _min_velocity_for_track(track_state: _TrackState, config: RenderCompareVelocityConfig) -> int:
    if track_state.is_drum:
        return int(config.drum_min_velocity)
    if "vocal" in track_state.stem_name:
        return int(config.vocal_min_velocity)
    if "bass" in track_state.stem_name:
        return int(config.bass_min_velocity)
    return int(config.min_velocity)


def _measure_k_weighted_envelope_db(
    mono_audio: np.ndarray,
    *,
    sample_rate: int,
    window_seconds: float,
) -> np.ndarray:
    """K-weighted 波形の固定窓 (window_seconds) ごとのラウドネス推移 (dB) を返す。"""
    k_weighted = _apply_k_weighting(mono_audio, sample_rate=sample_rate)
    window_samples = max(1, int(round(float(sample_rate) * float(window_seconds))))
    frame_count = max(1, int(math.ceil(float(k_weighted.size) / float(window_samples))))
    padded_length = frame_count * window_samples
    if k_weighted.size < padded_length:
        k_weighted = np.pad(k_weighted, (0, padded_length - k_weighted.size))
    framed = k_weighted[:padded_length].reshape(frame_count, window_samples)
    power = np.mean(np.square(framed, dtype=np.float64), axis=1)
    return (10.0 * np.log10(np.maximum(power, 1e-12))).astype(np.float64)


def _build_packed_stems_midi(
    stem_order: list[str],
    stem_durations: dict[str, float],
    tracks_by_stem: dict[str, list[_TrackState]],
    notes_by_track: dict[int, list[_NoteState]],
    *,
    output_midi_path: Path,
    config: RenderCompareVelocityConfig,
    include_full_mix_slot: bool = False,
) -> tuple[list[_StemSlot], _StemSlot | None]:
    """全ステムの現在 MIDI（Velocity + トラック別 CC#7）を 1 本の SMF Type-0 MIDI に高速直列パックする。"""
    scheduled_events: list[tuple[int, int, bytes]] = []
    slots: list[_StemSlot] = []
    cursor_seconds = 0.0

    def _append_slot(slot_name: str, slot_duration: float, slot_tracks: list[_TrackState]) -> _StemSlot:
        nonlocal cursor_seconds
        slot_start = cursor_seconds
        slot_end = slot_start + slot_duration
        start_tick = int(round(slot_start * _PACKED_TICKS_PER_SECOND))
        end_tick = int(round(slot_end * _PACKED_TICKS_PER_SECOND))

        melodic_tracks = [item for item in slot_tracks if not item.is_drum]
        drum_tracks = [item for item in slot_tracks if item.is_drum]
        melodic_tracks.sort(key=lambda item: -len(notes_by_track.get(item.track_key, [])))

        used_channels: set[int] = set()
        channel_assignments: list[tuple[_TrackState, int]] = []
        for melodic_index, track_state in enumerate(melodic_tracks):
            channel = _MELODIC_MIDI_CHANNELS[min(melodic_index, len(_MELODIC_MIDI_CHANNELS) - 1)]
            channel_assignments.append((track_state, channel))
            used_channels.add(channel)
        for track_state in drum_tracks:
            channel_assignments.append((track_state, 9))
            used_channels.add(9)

        for track_state, channel in channel_assignments:
            track_notes = notes_by_track.get(track_state.track_key, [])
            if not track_notes:
                continue
            volume_value = max(config.min_volume_cc7, min(config.max_volume_cc7, int(round(track_state.volume_cc7))))
            pan_value = max(0, min(127, int(track_state.pan_cc10)))
            track_min_velocity = _min_velocity_for_track(track_state, config)
            track_max_velocity = int(config.max_velocity)
            if not track_state.is_drum:
                scheduled_events.append(
                    (start_tick, 1, bytes((0xC0 | channel, max(0, min(127, int(track_state.program_number))))))
                )
            cc_status = 0xB0 | channel
            scheduled_events.append((start_tick, 2, bytes((cc_status, 7, volume_value))))
            scheduled_events.append((start_tick, 2, bytes((cc_status, 10, pan_value))))
            scheduled_events.append((start_tick, 2, bytes((cc_status, 11, 127))))
            scheduled_events.append((start_tick, 2, bytes((cc_status, 64, 0))))

            note_on_status = 0x90 | channel
            note_off_status = 0x80 | channel
            for note_state in track_notes:
                note_on_tick = int(round((slot_start + note_state.start_seconds) * _PACKED_TICKS_PER_SECOND))
                note_off_tick = max(
                    note_on_tick + 1,
                    min(end_tick, int(round((slot_start + note_state.end_seconds) * _PACKED_TICKS_PER_SECOND))),
                )
                velocity_value = max(track_min_velocity, min(track_max_velocity, int(round(note_state.velocity))))
                clamped_pitch = max(0, min(127, int(note_state.pitch)))
                scheduled_events.append((note_on_tick, 4, bytes((note_on_status, clamped_pitch, velocity_value))))
                scheduled_events.append((note_off_tick, 3, bytes((note_off_status, clamped_pitch, 0))))

        cleanup_tick = end_tick + int(round(min(0.08, config.stem_gap_seconds * 0.25) * _PACKED_TICKS_PER_SECOND))
        for channel in sorted(used_channels):
            cc_status = 0xB0 | channel
            scheduled_events.append((cleanup_tick, 5, bytes((cc_status, 123, 0))))
            scheduled_events.append((cleanup_tick, 6, bytes((cc_status, 120, 0))))

        cursor_seconds = slot_end + float(config.stem_gap_seconds)
        return _StemSlot(
            stem_name=slot_name,
            slot_start_seconds=float(slot_start),
            slot_duration_seconds=float(slot_duration),
        )

    for stem_name in stem_order:
        slots.append(_append_slot(stem_name, float(stem_durations[stem_name]), tracks_by_stem.get(stem_name, [])))

    full_mix_slot: _StemSlot | None = None
    if include_full_mix_slot and stem_order:
        full_mix_duration = max(float(stem_durations[name]) for name in stem_order)
        all_tracks = [item for name in stem_order for item in tracks_by_stem.get(name, [])]
        full_mix_slot = _append_slot("__full_mix__", full_mix_duration, all_tracks)

    # 末尾スロット終了後にわずかな余白を持たせて切り詰めを防ぐ（ダミー CC ではなく write_packed_smf_type0 で EOT を配置）。
    final_tick = int(round((cursor_seconds + 0.2) * _PACKED_TICKS_PER_SECOND))
    scheduled_events.append((final_tick, 7, b"\xb0\x7b\x00"))
    write_packed_smf_type0(
        output_midi_path,
        scheduled_events,
        ticks_per_beat=_PACKED_TICKS_PER_BEAT,
        tempo_microseconds=_PACKED_TEMPO_MICROSECONDS,
    )
    return slots, full_mix_slot


def _smooth_track_velocities(
    track_state: _TrackState,
    track_notes: list[_NoteState],
    *,
    config: RenderCompareVelocityConfig,
) -> None:
    """分離アーティファクトや微小ズレによるノート間のガタつきを抑えるフレーズ内平滑化。

    ボーカルはフレーズの一貫性が重要なため強めに平滑化し、ドラムはアクセントを残すため極弱にする。
    """
    if len(track_notes) < 3:
        return
    if track_state.is_drum:
        effective_strength = float(config.local_smoothing_strength) * 0.25
    elif "vocal" in track_state.stem_name:
        effective_strength = float(config.vocal_smoothing_strength)
    else:
        effective_strength = float(config.local_smoothing_strength)

    if effective_strength <= 0.0:
        return

    ordered = sorted(track_notes, key=lambda item: (item.start_seconds, item.pitch))
    raw_velocities = np.asarray([item.velocity for item in ordered], dtype=np.float64)
    starts = np.asarray([item.start_seconds for item in ordered], dtype=np.float64)
    smoothed = raw_velocities.copy()
    track_min_velocity = _min_velocity_for_track(track_state, config)

    # ボーカルはフレーズ全体（前後 0.8 秒・最大 3 ノート）で滑らかに繋ぎ、他は 0.45 秒以内で平滑化する。
    # 非ボーカル・非ドラムでは音高差 7 半音以内の同音域ノートを優先することで、
    # 同時発音する高音ソロ／リードが低音バッキング和音の中央値へ引きずられるのを防ぐ。
    radius_notes = 3 if "vocal" in track_state.stem_name else 2
    radius_seconds = 0.80 if "vocal" in track_state.stem_name else 0.45
    filter_by_pitch = (not track_state.is_drum) and ("vocal" not in track_state.stem_name)
    pitches = np.asarray([int(item.pitch) for item in ordered], dtype=np.int32)

    for index in range(len(ordered)):
        left = max(0, index - radius_notes)
        right = min(len(ordered), index + radius_notes + 1)
        neighbors = [
            raw_velocities[neighbor]
            for neighbor in range(left, right)
            if abs(starts[neighbor] - starts[index]) <= radius_seconds
            and ((not filter_by_pitch) or abs(int(pitches[neighbor]) - int(pitches[index])) <= 7)
        ]
        if len(neighbors) >= 3:
            local_median = float(np.median(neighbors))
            smoothed[index] = (1.0 - effective_strength) * raw_velocities[index] + effective_strength * local_median

    for note_state, value in zip(ordered, smoothed.tolist()):
        note_state.velocity = float(np.clip(value, track_min_velocity, config.max_velocity))


def _set_track_volume_controls(
    instrument: pretty_midi.Instrument,
    *,
    volume_cc7: int,
    preserve_expression_cc11: bool = False,
) -> None:
    """トラックの CC#7 (Volume) を更新する。

    preserve_expression_cc11=True かつ既に CC#11 が設定済みの場合はそれを保持し、
    未設定のときだけ CC#11=127 を補う。
    """
    has_existing_cc11 = any(int(control.number) == 11 for control in instrument.control_changes)
    keep_existing_cc11 = preserve_expression_cc11 and has_existing_cc11
    excluded = (7,) if keep_existing_cc11 else (7, 11)
    instrument.control_changes = [
        control for control in instrument.control_changes if int(control.number) not in excluded
    ]
    instrument.control_changes.append(pretty_midi.ControlChange(number=7, value=int(volume_cc7), time=0.0))
    if not keep_existing_cc11:
        instrument.control_changes.append(pretty_midi.ControlChange(number=11, value=127, time=0.0))
    instrument.control_changes.sort(key=lambda control: (control.time, control.number))


def _merge_optimized_stem_midis(
    loaded_midis: Mapping[str, pretty_midi.PrettyMIDI],
    *,
    max_melodic_instruments: int,
) -> pretty_midi.PrettyMIDI:
    """最適化済みステム MIDI 群を 16ch 制約（最大 max_melodic_instruments + ドラム）でマージする。"""
    first_midi = next(iter(loaded_midis.values()))
    merged = copy.deepcopy(first_midi)

    grouped_instruments: dict[tuple[int, bool, str], pretty_midi.Instrument] = {}
    cc7_weighted_sum: dict[tuple[int, bool, str], float] = {}
    cc7_weight_total: dict[tuple[int, bool, str], float] = {}

    for midi_object in loaded_midis.values():
        for source in midi_object.instruments:
            if not source.notes:
                continue
            key = (int(source.program), bool(source.is_drum), str(source.name))
            target = grouped_instruments.get(key)
            if target is None:
                target = pretty_midi.Instrument(program=key[0], is_drum=key[1], name=key[2])
                grouped_instruments[key] = target
                cc7_weighted_sum[key] = 0.0
                cc7_weight_total[key] = 0.0
            target.notes.extend(copy.deepcopy(source.notes))
            target.pitch_bends.extend(copy.deepcopy(source.pitch_bends))
            # CC#10 (Pan) や抽出済みの CC#11 (Expression) はそのまま引き継ぐ。
            target.control_changes.extend(
                copy.deepcopy(control) for control in source.control_changes if int(control.number) != 7
            )
            source_cc7 = next((int(control.value) for control in source.control_changes if int(control.number) == 7), 100)
            note_weight = float(len(source.notes))
            cc7_weighted_sum[key] += float(source_cc7) * note_weight
            cc7_weight_total[key] += note_weight

    for key, instrument in grouped_instruments.items():
        averaged_cc7 = int(np.clip(round(cc7_weighted_sum[key] / max(cc7_weight_total[key], 1.0)), 1, 127))
        _set_track_volume_controls(instrument, volume_cc7=averaged_cc7, preserve_expression_cc11=True)

    melodic = [instrument for instrument in grouped_instruments.values() if not instrument.is_drum]
    drums = [instrument for instrument in grouped_instruments.values() if instrument.is_drum]
    melodic.sort(key=lambda instrument: (-len(instrument.notes), int(instrument.program), str(instrument.name)))
    drums.sort(key=lambda instrument: (-len(instrument.notes), int(instrument.program), str(instrument.name)))

    limit = int(max_melodic_instruments)
    if limit > 0 and len(melodic) > limit:
        kept = melodic[: max(0, limit - 1)]
        overflow = melodic[max(0, limit - 1) :]
        merged_overflow = pretty_midi.Instrument(
            program=int(overflow[0].program),
            is_drum=False,
            name="Other / Merged",
        )
        overflow_cc7_sum = 0.0
        overflow_weight = 0.0
        for instrument in overflow:
            merged_overflow.notes.extend(instrument.notes)
            merged_overflow.pitch_bends.extend(instrument.pitch_bends)
            source_cc7 = next((int(control.value) for control in instrument.control_changes if int(control.number) == 7), 100)
            overflow_cc7_sum += float(source_cc7) * float(len(instrument.notes))
            overflow_weight += float(len(instrument.notes))
        _set_track_volume_controls(
            merged_overflow,
            volume_cc7=int(np.clip(round(overflow_cc7_sum / max(overflow_weight, 1.0)), 1, 127)),
        )
        melodic = [*kept, merged_overflow]

    final_instruments = [*melodic, *drums]
    for instrument in final_instruments:
        # 16ch 制約によるトラック統合 (Other / Merged 等) や複数ステム合流で同ピッチノートが重なり
        # FluidSynth 等で発音消失するのを防ぐため、先行ノートの終端を後続ノート開始の直前で切る。
        notes_by_pitch: dict[int, list[pretty_midi.Note]] = {}
        for note in instrument.notes:
            notes_by_pitch.setdefault(int(note.pitch), []).append(note)
        for pitch_notes in notes_by_pitch.values():
            pitch_notes.sort(key=lambda item: (item.start, item.end))
            for index in range(1, len(pitch_notes)):
                previous = pitch_notes[index - 1]
                current = pitch_notes[index]
                if float(current.start) <= float(previous.start) + 0.005:
                    current.start = float(previous.start) + 0.005
                    current.end = max(float(current.end), float(current.start) + 0.015)
                if float(previous.end) > float(current.start) - 0.005:
                    previous.end = max(
                        float(previous.start) + min(0.010, (float(current.start) - float(previous.start)) * 0.7),
                        float(current.start) - 0.005,
                    )
        instrument.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
        instrument.control_changes.sort(key=lambda control: (control.time, control.number))
        instrument.pitch_bends.sort(key=lambda bend: bend.time)
    merged.instruments = final_instruments
    return merged


def optimize_velocity_and_mix_by_render_compare(
    stem_midis: Mapping[str, Path | str],
    stem_audios: Mapping[str, Path | str],
    *,
    config: RenderCompareVelocityConfig,
    original_mix_audio: Path | str | None = None,
    output_midi_path: Path | str | None = None,
    output_stem_dir: Path | str | None = None,
    packed_midi_renderer: _PackedMidiRenderer | None = None,
    preloaded_stereo_waveforms: Mapping[str, np.ndarray] | None = None,
    preloaded_full_mix_mono: np.ndarray | None = None,
) -> dict[str, Any]:
    """全ステムの MIDI Velocity とトラック別 CC#7 を Render & Compare で閉ループ最適化する。"""
    sample_rate = int(config.sample_rate)
    resolved_midis = {
        str(name).lower(): Path(path)
        for name, path in stem_midis.items()
        if Path(path).is_file()
    }
    resolved_audios = {
        str(name).lower(): Path(path)
        for name, path in stem_audios.items()
        if Path(path).is_file()
    }
    preloaded_stereo_normalized = (
        {str(name).lower(): np.asarray(wave, dtype=np.float32) for name, wave in preloaded_stereo_waveforms.items()}
        if preloaded_stereo_waveforms is not None
        else {}
    )
    active_stems = sorted(set(resolved_midis) & set(resolved_audios))
    if not active_stems:
        raise ValueError("No matching stem MIDI and audio pairs were found.")

    # 1. 元ステム音声（ステレオ + モノラル）・MIDI を読み込み、同一ステム内の重複トラックを統合する。
    loaded_midis: dict[str, pretty_midi.PrettyMIDI] = {}
    target_stereo_waveforms: dict[str, np.ndarray] = {}
    target_waveforms: dict[str, np.ndarray] = {}
    stem_durations: dict[str, float] = {}
    tracks: list[_TrackState] = []
    tracks_by_stem: dict[str, list[_TrackState]] = {}
    notes: list[_NoteState] = []
    notes_by_track: dict[int, list[_NoteState]] = {}
    notes_by_stem: dict[str, list[_NoteState]] = {}

    for stem_name in active_stems:
        midi_object = pretty_midi.PrettyMIDI(str(resolved_midis[stem_name]))
        consolidated_by_key: dict[tuple[int, bool, str], pretty_midi.Instrument] = {}
        for instrument in midi_object.instruments:
            if not instrument.notes:
                continue
            key = (int(instrument.program), bool(instrument.is_drum), str(instrument.name))
            merged_instrument = consolidated_by_key.get(key)
            if merged_instrument is None:
                merged_instrument = pretty_midi.Instrument(program=key[0], is_drum=key[1], name=key[2])
                consolidated_by_key[key] = merged_instrument
            merged_instrument.notes.extend(instrument.notes)
            merged_instrument.pitch_bends.extend(instrument.pitch_bends)
            merged_instrument.control_changes.extend(instrument.control_changes)
        consolidated_instruments = list(consolidated_by_key.values())
        for instrument in consolidated_instruments:
            instrument.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
            instrument.pitch_bends.sort(key=lambda bend: bend.time)
            instrument.control_changes.sort(key=lambda control: control.time)
        midi_object.instruments = consolidated_instruments

        if stem_name in preloaded_stereo_normalized:
            stereo_audio = preloaded_stereo_normalized[stem_name]
            if stereo_audio.ndim == 1:
                stereo_audio = np.repeat(stereo_audio[None, :], 2, axis=0)
        else:
            stereo_audio = _load_stereo_audio(resolved_audios[stem_name], target_sample_rate=sample_rate)
        mono_audio = stereo_audio.mean(axis=0).astype(np.float32, copy=False)
        loaded_midis[stem_name] = midi_object
        target_stereo_waveforms[stem_name] = stereo_audio
        target_waveforms[stem_name] = mono_audio
        stem_durations[stem_name] = max(0.25, float(mono_audio.size) / float(sample_rate))
        tracks_by_stem[stem_name] = []
        notes_by_stem[stem_name] = []

    # 1.5 ステレオ左右パンに明確に分かれたダブルトラック（ダブルギター＋中央ソロ等）を
    #     MIDI 16ch 予算の範囲内で (L)/(C)/(R) サブトラックへ自動分割する。
    stereo_split_reports: list[dict[str, Any]] = []
    if config.split_stereo_pan_tracks and config.optimize_pan_cc10:
        from .expression_pan import ExpressionPanConfig, split_stereo_pan_tracks_in_stem_midi

        major_melodic_count = sum(
            1
            for midi_object in loaded_midis.values()
            for instrument in midi_object.instruments
            if not instrument.is_drum and len(instrument.notes) >= 30
        )
        # 16ch 枠（最大 max_melodic_instruments 本）を圧迫しない範囲で優先ステムから分割予算を配分する。
        remaining_split_budget = max(2, min(4, int(config.max_melodic_instruments) - major_melodic_count))
        split_config = ExpressionPanConfig(sample_rate=sample_rate)
        # guitar -> other -> piano の順でダブルトラック効果の大きいステムから優先分割する。
        preferred_order = sorted(
            loaded_midis,
            key=lambda name: (0 if "guitar" in name else (1 if "other" in name else 2), name),
        )
        for stem_name in preferred_order:
            if remaining_split_budget <= 0:
                break
            reports_for_stem = split_stereo_pan_tracks_in_stem_midi(
                loaded_midis[stem_name],
                target_stereo_waveforms[stem_name],
                stem_name=stem_name,
                config=split_config,
                max_extra_tracks=remaining_split_budget,
            )
            for item in reports_for_stem:
                remaining_split_budget -= max(0, len(item["subtracks"]) - 1)
                stereo_split_reports.append(item)

    for stem_name in active_stems:
        midi_object = loaded_midis[stem_name]
        for instrument_index, instrument in enumerate(midi_object.instruments):
            if not instrument.notes:
                continue
            track_key = len(tracks)
            instrument_name_str = str(instrument.name)
            if instrument_name_str.endswith(" (L)"):
                spatial_zone = "left"
            elif instrument_name_str.endswith(" (C)"):
                spatial_zone = "center"
            elif instrument_name_str.endswith(" (R)"):
                spatial_zone = "right"
            else:
                spatial_zone = "mono"
            initial_pan = next(
                (int(control.value) for control in instrument.control_changes if int(control.number) == 10),
                64,
            )
            track_state = _TrackState(
                track_key=track_key,
                stem_name=stem_name,
                instrument_index=instrument_index,
                program_number=int(instrument.program),
                is_drum=bool(instrument.is_drum),
                instrument_name=instrument_name_str,
                instrument=instrument,
                volume_cc7=float(config.initial_volume_cc7),
                is_monophonic=False,
                pan_cc10=int(np.clip(initial_pan, 0, 127)),
                spatial_zone=spatial_zone,
            )
            tracks.append(track_state)
            tracks_by_stem[stem_name].append(track_state)
            notes_by_track[track_key] = []
            track_min_velocity = _min_velocity_for_track(track_state, config)

            for note in instrument.notes:
                initial_velocity = (
                    float(np.clip(int(note.velocity), track_min_velocity, config.max_velocity))
                    if config.use_input_velocity_as_init and int(note.velocity) > 0
                    else float(config.initial_velocity)
                )
                note_state = _NoteState(
                    note_id=len(notes),
                    track_key=track_key,
                    stem_name=stem_name,
                    note=note,
                    start_seconds=float(max(0.0, note.start)),
                    end_seconds=float(max(note.start + 0.02, note.end)),
                    pitch=int(note.pitch),
                    is_drum=bool(instrument.is_drum),
                    velocity=initial_velocity,
                )
                notes.append(note_state)
                notes_by_track[track_key].append(note_state)
                notes_by_stem[stem_name].append(note_state)

            track_state.is_monophonic = (
                (not track_state.is_drum)
                and (
                    "vocal" in stem_name
                    or "bass" in stem_name
                    or _is_mostly_monophonic(notes_by_track[track_key])
                )
            )

    stems_with_notes = [stem_name for stem_name in active_stems if notes_by_stem.get(stem_name)]
    if not stems_with_notes or not notes:
        raise ValueError("No notes found in matching stem MIDI files.")

    # 2. 元ステム側の K-weighted マクロ音量 (LUFS 相当) とノート別ミクロ音量を計測する。
    n_fft = min(4096, max(128, 2 ** int(round(math.log2(max(128.0, float(sample_rate) * 0.18))))))
    hop_length = max(32, min(256, n_fft // 8))
    pitch_filterbank, drum_filterbank = _build_harmonic_pitch_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
    )

    target_by_stem: dict[str, _StemAcousticMeasurement] = {}
    for stem_name in stems_with_notes:
        target_by_stem[stem_name] = _measure_stem_acoustics(
            target_waveforms[stem_name],
            tracks_by_stem[stem_name],
            notes_by_stem[stem_name],
            notes_by_track,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            pitch_filterbank=pitch_filterbank,
            drum_filterbank=drum_filterbank,
            chord_voicing_strength=float(config.chord_voicing_strength),
            stereo_audio=target_stereo_waveforms[stem_name],
        )

    # 2.5 元フルミックスの目標ラウドネス推移と、ボーカル用の伴奏ラウドネス推移を事前計算する。
    full_mix_enabled = len(stems_with_notes) >= 2 and float(config.full_mix_feedback_weight) > 0.0
    max_stem_samples = max(int(target_waveforms[name].size) for name in stems_with_notes)
    if preloaded_full_mix_mono is not None:
        target_full_mix_wave = np.asarray(preloaded_full_mix_mono, dtype=np.float32)
    elif original_mix_audio is not None and Path(original_mix_audio).is_file():
        target_full_mix_wave = _load_mono_audio(Path(original_mix_audio), target_sample_rate=sample_rate)
    else:
        target_full_mix_wave = np.zeros(max_stem_samples, dtype=np.float32)
        for name in stems_with_notes:
            wave = target_waveforms[name]
            target_full_mix_wave[: wave.size] += wave
    target_full_mix_db = _measure_k_weighted_envelope_db(
        target_full_mix_wave,
        sample_rate=sample_rate,
        window_seconds=float(config.full_mix_window_seconds),
    )

    # ボーカルの分離減衰（伴奏が大きいサビほど vocals.wav が削られる現象）を補償するため、
    # 伴奏全ステムの合計 K-weighted ラウドネス推移から各ボーカルノートへの補償量 (dB) を算出する。
    vocal_compensation_db_by_note: dict[int, float] = {}
    non_vocal_stems = [name for name in stems_with_notes if "vocal" not in name]
    if non_vocal_stems and float(config.vocal_accompaniment_compensation) > 0.0:
        accompaniment_wave = np.zeros(max_stem_samples, dtype=np.float32)
        for name in non_vocal_stems:
            wave = target_waveforms[name]
            accompaniment_wave[: wave.size] += wave
        accompaniment_db = _measure_k_weighted_envelope_db(
            accompaniment_wave,
            sample_rate=sample_rate,
            window_seconds=float(config.full_mix_window_seconds),
        )
        vocal_notes = [item for item in notes if "vocal" in item.stem_name]
        if vocal_notes:
            vocal_acc_samples = np.asarray(
                [
                    accompaniment_db[
                        min(
                            int(accompaniment_db.size) - 1,
                            max(0, int(item.start_seconds / float(config.full_mix_window_seconds))),
                        )
                    ]
                    for item in vocal_notes
                ],
                dtype=np.float64,
            )
            acc_median_db = float(np.median(vocal_acc_samples))
            for item, acc_value in zip(vocal_notes, vocal_acc_samples.tolist()):
                # 伴奏が中央値より大きい区間（サビ）では +方向、静かな独唱区間では -方向へ穏やかに補償。
                compensation = float(
                    np.clip(
                        (acc_value - acc_median_db) * float(config.vocal_accompaniment_compensation),
                        -3.5,
                        4.5,
                    )
                )
                vocal_compensation_db_by_note[item.note_id] = compensation

    # 3. 閉ループ反復最適化（本番 FluidSynth 実行時はステム＋フルミックスを並列プロセスでレンダリング）。
    iteration_history: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tsumugi_rc_velocity_") as temp_directory:
        temp_root = Path(temp_directory)
        total_passes = int(config.num_iterations) + (1 if config.evaluate_final else 0)
        parallel_job_count = len(stems_with_notes) + (1 if full_mix_enabled else 0)
        parallel_workers = (
            default_parallel_render_workers(
                Path(config.soundfont_path),
                parallel_job_count,
                max_workers=config.parallel_render_workers,
            )
            if packed_midi_renderer is None
            else 1
        )

        for iteration in range(total_passes):
            is_eval_only_pass = iteration >= int(config.num_iterations)
            rendered_by_stem: dict[str, _StemAcousticMeasurement] = {}
            rendered_full_mix_segment: np.ndarray | None = None

            render_options = FluidSynthRenderOptions(
                soundfont_path=Path(config.soundfont_path),
                sample_rate=sample_rate,
                fluidsynth_executable=config.fluidsynth_executable,
                render_gain=float(config.render_gain),
                disable_reverb=bool(config.disable_reverb),
                disable_chorus=bool(config.disable_chorus),
                polyphony=96,
            )

            if parallel_workers <= 1 or parallel_job_count <= 1:
                packed_midi_path = temp_root / f"iteration_{iteration}_packed.mid"
                slots, full_mix_slot = _build_packed_stems_midi(
                    stems_with_notes,
                    stem_durations,
                    tracks_by_stem,
                    notes_by_track,
                    output_midi_path=packed_midi_path,
                    config=config,
                    include_full_mix_slot=full_mix_enabled,
                )
                if packed_midi_renderer is not None:
                    packed_wav_path = temp_root / f"iteration_{iteration}_packed.wav"
                    packed_midi_renderer(packed_midi_path, packed_wav_path, sample_rate)
                    rendered_full, rendered_sample_rate = sf.read(str(packed_wav_path), dtype="float32", always_2d=True)
                    if int(rendered_sample_rate) != sample_rate:
                        raise ValueError(f"Rendered sample rate mismatch: expected {sample_rate}, got {rendered_sample_rate}")
                    rendered_mono = rendered_full.mean(axis=1).astype(np.float32, copy=False)
                    rendered_stereo_full = (
                        rendered_full[:, :2].T.copy()
                        if rendered_full.shape[1] >= 2
                        else np.repeat(rendered_full.T, 2, axis=0)
                    )
                else:
                    rendered_stereo_full = render_midi_to_stereo_array(packed_midi_path, options=render_options)
                    rendered_mono = rendered_stereo_full.mean(axis=0).astype(np.float32, copy=False)
                total_rendered_samples = int(rendered_mono.size)

                for slot in slots:
                    expected_samples = int(target_waveforms[slot.stem_name].size)
                    start_sample = max(0, int(round(slot.slot_start_seconds * sample_rate)))
                    end_sample = min(total_rendered_samples, start_sample + expected_samples)
                    segment = rendered_mono[start_sample:end_sample]
                    stereo_segment = rendered_stereo_full[:, start_sample:end_sample]
                    if segment.size < expected_samples:
                        pad_width = expected_samples - segment.size
                        segment = np.pad(segment, (0, pad_width))
                        stereo_segment = np.pad(stereo_segment, ((0, 0), (0, pad_width)))
                    rendered_by_stem[slot.stem_name] = _measure_stem_acoustics(
                        segment,
                        tracks_by_stem[slot.stem_name],
                        notes_by_stem[slot.stem_name],
                        notes_by_track,
                        sample_rate=sample_rate,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        pitch_filterbank=pitch_filterbank,
                        drum_filterbank=drum_filterbank,
                        chord_voicing_strength=float(config.chord_voicing_strength),
                        stereo_audio=stereo_segment,
                    )

                if full_mix_slot is not None:
                    fm_start = max(0, int(round(full_mix_slot.slot_start_seconds * sample_rate)))
                    fm_expected = int(target_full_mix_wave.size)
                    fm_segment = rendered_mono[fm_start : min(total_rendered_samples, fm_start + fm_expected)]
                    if fm_segment.size < fm_expected:
                        fm_segment = np.pad(fm_segment, (0, fm_expected - fm_segment.size))
                    rendered_full_mix_segment = fm_segment
            else:
                batch_midi_paths: list[Path] = []
                for stem_index, stem_name in enumerate(stems_with_notes):
                    stem_midi_path = temp_root / f"iter_{iteration}_stem_{stem_index}.mid"
                    _build_packed_stems_midi(
                        [stem_name],
                        stem_durations,
                        tracks_by_stem,
                        notes_by_track,
                        output_midi_path=stem_midi_path,
                        config=config,
                        include_full_mix_slot=False,
                    )
                    batch_midi_paths.append(stem_midi_path)

                if full_mix_enabled:
                    full_mix_midi_path = temp_root / f"iter_{iteration}_full_mix.mid"
                    merged_tracks_by_stem = {"__full_mix__": tracks}
                    merged_durations = {"__full_mix__": max(stem_durations.values())}
                    _build_packed_stems_midi(
                        ["__full_mix__"],
                        merged_durations,
                        merged_tracks_by_stem,
                        notes_by_track,
                        output_midi_path=full_mix_midi_path,
                        config=config,
                        include_full_mix_slot=False,
                    )
                    batch_midi_paths.append(full_mix_midi_path)

                rendered_stereo_batch = render_midi_batch_to_stereo_arrays(
                    batch_midi_paths,
                    options=render_options,
                    max_workers=parallel_workers,
                )

                for stem_index, stem_name in enumerate(stems_with_notes):
                    rendered_stereo_stem = rendered_stereo_batch[stem_index]
                    expected_samples = int(target_waveforms[stem_name].size)
                    stereo_segment = rendered_stereo_stem[:, :expected_samples]
                    segment = stereo_segment.mean(axis=0).astype(np.float32, copy=False)
                    if segment.size < expected_samples:
                        pad_width = expected_samples - segment.size
                        segment = np.pad(segment, (0, pad_width))
                        stereo_segment = np.pad(stereo_segment, ((0, 0), (0, pad_width)))
                    rendered_by_stem[stem_name] = _measure_stem_acoustics(
                        segment,
                        tracks_by_stem[stem_name],
                        notes_by_stem[stem_name],
                        notes_by_track,
                        sample_rate=sample_rate,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        pitch_filterbank=pitch_filterbank,
                        drum_filterbank=drum_filterbank,
                        chord_voicing_strength=float(config.chord_voicing_strength),
                        stereo_audio=stereo_segment,
                    )

                if full_mix_enabled:
                    rendered_stereo_mix = rendered_stereo_batch[len(stems_with_notes)]
                    fm_expected = int(target_full_mix_wave.size)
                    fm_segment = rendered_stereo_mix.mean(axis=0).astype(np.float32, copy=False)[:fm_expected]
                    if fm_segment.size < fm_expected:
                        fm_segment = np.pad(fm_segment, (0, fm_expected - fm_segment.size))
                    rendered_full_mix_segment = fm_segment

            # 3-a-0. フルミックススロットの整合性を時間窓ごとに計測する。
            full_mix_error_curve_db: np.ndarray | None = None
            full_mix_metrics: dict[str, float] = {}
            if rendered_full_mix_segment is not None:
                rendered_full_mix_db = _measure_k_weighted_envelope_db(
                    rendered_full_mix_segment,
                    sample_rate=sample_rate,
                    window_seconds=float(config.full_mix_window_seconds),
                )
                common_frames = min(int(target_full_mix_db.size), int(rendered_full_mix_db.size))
                target_fm = target_full_mix_db[:common_frames]
                rendered_fm = rendered_full_mix_db[:common_frames]
                active_fm = (target_fm > float(np.percentile(target_fm, 90)) - 25.0) & (
                    rendered_fm > float(np.percentile(rendered_fm, 90)) - 35.0
                )
                if int(active_fm.sum()) >= 4:
                    fm_diff = target_fm - rendered_fm
                    fm_offset = float(np.median(fm_diff[active_fm]))
                    full_mix_error_curve_db = fm_diff - fm_offset
                    active_errors = full_mix_error_curve_db[active_fm]
                    correlation = (
                        float(np.corrcoef(target_fm[active_fm], rendered_fm[active_fm])[0, 1])
                        if float(np.std(target_fm[active_fm])) > 1e-6 and float(np.std(rendered_fm[active_fm])) > 1e-6
                        else 1.0
                    )
                    full_mix_metrics = {
                        "full_mix_mae_db": float(np.mean(np.abs(active_errors))),
                        "full_mix_corr": correlation,
                        "target_full_mix_std_db": float(np.std(target_fm[active_fm])),
                        "rendered_full_mix_std_db": float(np.std(rendered_fm[active_fm])),
                    }

            # 3-a. マクロ（ステム間 K-weighted LUFS 誤差）とミクロ（トラック内相対ノート誤差）を算出する。
            stem_lufs_diffs = {
                stem_name: target_by_stem[stem_name].stem_active_lufs_db - rendered_by_stem[stem_name].stem_active_lufs_db
                for stem_name in stems_with_notes
            }
            global_lufs_offset_db = float(np.median(list(stem_lufs_diffs.values())))
            stem_mix_error_db = {
                stem_name: float(diff_db - global_lufs_offset_db)
                for stem_name, diff_db in stem_lufs_diffs.items()
            }

            # 各トラック内でのノート相対誤差（トラック中央値を 0 dB に基準化 = マクロ音量と直交化）。
            # ボーカルには伴奏連動の分離減衰補償を加え、全トラックにフルミックス時間誤差を穏やかに反映する。
            within_track_note_errors_db: dict[int, float] = {}
            for track_state in tracks:
                track_notes = notes_by_track.get(track_state.track_key, [])
                if not track_notes:
                    continue
                target_note_map = target_by_stem[track_state.stem_name].note_levels_db
                rendered_note_map = rendered_by_stem[track_state.stem_name].note_levels_db
                raw_track_diffs = np.asarray(
                    [
                        (
                            target_note_map[item.note_id]
                            + vocal_compensation_db_by_note.get(item.note_id, 0.0)
                        )
                        - rendered_note_map[item.note_id]
                        for item in track_notes
                    ],
                    dtype=np.float64,
                )
                track_median = float(np.median(raw_track_diffs))
                for item, raw_diff in zip(track_notes, raw_track_diffs.tolist()):
                    note_relative_error = float(raw_diff - track_median)
                    if full_mix_error_curve_db is not None and float(config.full_mix_feedback_weight) > 0.0:
                        frame_index = min(
                            int(full_mix_error_curve_db.size) - 1,
                            max(0, int(item.start_seconds / float(config.full_mix_window_seconds))),
                        )
                        fm_feedback = float(np.clip(full_mix_error_curve_db[frame_index], -6.0, 6.0))
                        # ボーカルは伴奏が凹んでいる区間で過剰に持ち上がらないようフルミックス影響を控えめにする。
                        fm_weight = float(config.full_mix_feedback_weight) * (
                            0.35 if "vocal" in track_state.stem_name else 1.0
                        )
                        note_relative_error += fm_weight * fm_feedback
                    within_track_note_errors_db[item.note_id] = note_relative_error

            stem_mix_mae_db = float(np.mean([abs(value) for value in stem_mix_error_db.values()]))
            note_dynamics_mae_db = float(np.mean([abs(value) for value in within_track_note_errors_db.values()]))
            overall_mae_db = float(
                np.mean(
                    [
                        abs(stem_mix_error_db[item.stem_name] + within_track_note_errors_db[item.note_id])
                        for item in notes
                    ]
                )
            )
            iteration_entry: dict[str, Any] = {
                "iteration": int(iteration),
                "global_lufs_offset_db": float(global_lufs_offset_db),
                "stem_mix_mae_db": stem_mix_mae_db,
                "note_dynamics_mae_db": note_dynamics_mae_db,
                "overall_mae_db": overall_mae_db,
                "stem_mix_error_db": stem_mix_error_db,
                "target_stem_lufs_db": {
                    stem_name: float(target_by_stem[stem_name].stem_active_lufs_db)
                    for stem_name in stems_with_notes
                },
                "rendered_stem_lufs_db": {
                    stem_name: float(rendered_by_stem[stem_name].stem_active_lufs_db)
                    for stem_name in stems_with_notes
                },
            }
            iteration_entry.update(full_mix_metrics)
            iteration_history.append(iteration_entry)

            if is_eval_only_pass:
                break

            # 3-b. マクロ更新：K-weighted LUFS 誤差から各トラックの CC#7 を直接合わせる。
            #      極端に静かな 1 パッチ（例: Slap Bass 2）の最大値に合わせて全体が小さくならないよう、
            #      主要トラックの 85 パーセンタイルを target_reference_volume_cc7 へ揃え、
            #      127 で天井に当たったトラックの不足分は下の Velocity 目標中心引き上げで補う。
            cc7_headroom_deficit_db: dict[int, float] = {track_state.track_key: 0.0 for track_state in tracks}
            if config.optimize_volume_cc7:
                proposed_cc7: dict[int, float] = {}
                for track_state in tracks:
                    track_notes = notes_by_track.get(track_state.track_key, [])
                    if not track_notes:
                        proposed_cc7[track_state.track_key] = track_state.volume_cc7
                        continue

                    base_stem_error = stem_mix_error_db[track_state.stem_name]
                    if len(tracks_by_stem[track_state.stem_name]) > 1:
                        target_track_lufs = target_by_stem[track_state.stem_name].track_active_lufs_db[track_state.track_key]
                        rendered_track_lufs = rendered_by_stem[track_state.stem_name].track_active_lufs_db[track_state.track_key]
                        track_raw_error = (target_track_lufs - rendered_track_lufs) - global_lufs_offset_db
                        shrinkage = float(len(track_notes)) / (float(len(track_notes)) + 16.0)
                        within_offset = float(
                            np.clip(
                                shrinkage * (track_raw_error - base_stem_error),
                                -config.max_within_stem_track_offset_db,
                                config.max_within_stem_track_offset_db,
                            )
                        )
                        desired_error_db = base_stem_error + within_offset
                    else:
                        desired_error_db = base_stem_error

                    clamped_error = float(np.clip(desired_error_db, -config.max_step_error_db, config.max_step_error_db))
                    ratio = 10.0 ** (float(config.volume_cc7_damping) * clamped_error / float(config.volume_cc7_curve_range_db))
                    proposed_cc7[track_state.track_key] = track_state.volume_cc7 * ratio

                # ノート数 30 以上の主要トラック群の 85 パーセンタイル（無ければ全トラックの 85 パーセンタイル）を基準にする。
                major_proposed = [
                    proposed_cc7[item.track_key]
                    for item in tracks
                    if len(notes_by_track.get(item.track_key, [])) >= 30
                ]
                reference_pool = major_proposed or [
                    proposed_cc7[item.track_key] for item in tracks if notes_by_track.get(item.track_key)
                ]
                reference_cc7 = (
                    float(np.percentile(reference_pool, 85))
                    if reference_pool
                    else float(config.target_reference_volume_cc7)
                )
                normalization_scale = float(config.target_reference_volume_cc7) / max(reference_cc7, 1e-6)
                for track_state in tracks:
                    unclamped_cc7 = proposed_cc7[track_state.track_key] * normalization_scale
                    clamped_cc7 = float(np.clip(unclamped_cc7, config.min_volume_cc7, config.max_volume_cc7))
                    track_state.volume_cc7 = clamped_cc7
                    # CC#7 が 127 天井に当たって足りなくなった dB 分を記録し、Velocity 側で補えるようにする。
                    if unclamped_cc7 > clamped_cc7 + 1e-3:
                        cc7_headroom_deficit_db[track_state.track_key] = float(
                            config.volume_cc7_curve_range_db * math.log10(unclamped_cc7 / clamped_cc7)
                        )

            # 3-c. ミクロ更新：トラック内相対誤差だけを使って Velocity を更新する。
            #      各トラックの中央値 Velocity は initial_velocity (既定 104: LPF が開いて抜ける領域)
            #      に保ち、CC#7 が 127 で頭打ちになった静かなパッチだけ目標中心を最大 122 まで引き上げる。
            for track_state in tracks:
                track_notes = notes_by_track.get(track_state.track_key, [])
                if not track_notes:
                    continue
                track_min_velocity = _min_velocity_for_track(track_state, config)
                compression = (
                    float(config.vocal_dynamics_compression)
                    if "vocal" in track_state.stem_name
                    else float(config.dynamics_compression)
                )

                unclipped_velocities: list[float] = []
                for note_state in track_notes:
                    relative_error_db = within_track_note_errors_db[note_state.note_id] * compression
                    clamped_error = float(np.clip(relative_error_db, -config.max_step_error_db, config.max_step_error_db))
                    velocity_ratio = 10.0 ** (
                        float(config.velocity_damping) * clamped_error / float(config.velocity_curve_range_db)
                    )
                    unclipped_velocity = note_state.velocity * velocity_ratio
                    unclipped_velocities.append(unclipped_velocity)
                    note_state.velocity = float(
                        np.clip(unclipped_velocity, track_min_velocity, config.max_velocity)
                    )

                _smooth_track_velocities(track_state, track_notes, config=config)

                if config.optimize_volume_cc7 and len(track_notes) >= 3:
                    deficit_db = cc7_headroom_deficit_db[track_state.track_key]
                    # ギターソロやサビのピークが Velocity 127 天井へ張り付いている場合、CC#7 に余裕があれば
                    # トラック中央の目標 Velocity を少し下げつつ CC#7 を同 dB 引き上げてピークの余裕を確保する。
                    peak_clip_db = 0.0
                    if deficit_db <= 1e-3 and track_state.volume_cc7 < float(config.max_volume_cc7) - 1.0:
                        upper_unclipped = float(np.percentile(unclipped_velocities, 95))
                        if upper_unclipped > float(config.max_velocity) * 1.03:
                            raw_clip_db = float(config.velocity_curve_range_db) * math.log10(
                                upper_unclipped / float(config.max_velocity)
                            )
                            available_cc7_headroom_db = float(config.volume_cc7_curve_range_db) * math.log10(
                                float(config.max_volume_cc7) / max(track_state.volume_cc7, 1.0)
                            )
                            peak_clip_db = min(raw_clip_db, available_cc7_headroom_db, 5.5)

                    base_center = float(config.initial_velocity) * (
                        10.0 ** ((deficit_db - peak_clip_db) / float(config.velocity_curve_range_db))
                    )
                    target_center_velocity = float(
                        np.clip(
                            base_center,
                            max(track_min_velocity, 84),
                            min(122, int(config.max_velocity)),
                        )
                    )
                    median_velocity = float(np.median([item.velocity for item in track_notes]))
                    if median_velocity > 1.0:
                        recenter_ratio = target_center_velocity / median_velocity
                        blended_ratio = recenter_ratio ** 0.65
                        for note_state in track_notes:
                            note_state.velocity = float(
                                np.clip(note_state.velocity * blended_ratio, track_min_velocity, config.max_velocity)
                            )
                        # Velocity を下げた実適用 dB 分だけ CC#7 を同量引き上げ、トラック全体の基準ラウドネスを厳密に保つ。
                        if peak_clip_db > 0.2:
                            new_median_velocity = float(np.median([item.velocity for item in track_notes]))
                            applied_velocity_drop_db = float(config.velocity_curve_range_db) * math.log10(
                                max(median_velocity, 1.0) / max(new_median_velocity, 1.0)
                            )
                            if applied_velocity_drop_db > 0.0:
                                track_state.volume_cc7 = float(
                                    np.clip(
                                        track_state.volume_cc7
                                        * (10.0 ** (applied_velocity_drop_db / float(config.volume_cc7_curve_range_db))),
                                        config.min_volume_cc7,
                                        config.max_volume_cc7,
                                    )
                                )

    # 4. 最適化結果を PrettyMIDI オブジェクトへ書き戻す。
    track_reports: list[dict[str, Any]] = []
    for track_state in tracks:
        track_notes = notes_by_track.get(track_state.track_key, [])
        if not track_notes:
            continue
        final_cc7 = int(np.clip(round(track_state.volume_cc7), config.min_volume_cc7, config.max_volume_cc7))
        track_min_velocity = _min_velocity_for_track(track_state, config)
        _set_track_volume_controls(track_state.instrument, volume_cc7=final_cc7)
        final_velocities: list[int] = []
        for note_state in track_notes:
            final_velocity = int(np.clip(round(note_state.velocity), track_min_velocity, config.max_velocity))
            note_state.note.velocity = final_velocity
            final_velocities.append(final_velocity)
        velocity_array = np.asarray(final_velocities, dtype=np.float64)
        track_reports.append(
            {
                "stem_name": track_state.stem_name,
                "instrument_name": track_state.instrument_name,
                "program_number": int(track_state.program_number),
                "is_drum": bool(track_state.is_drum),
                "is_monophonic": bool(track_state.is_monophonic),
                "note_count": int(len(track_notes)),
                "volume_cc7": int(final_cc7),
                "velocity_mean": float(velocity_array.mean()),
                "velocity_std": float(velocity_array.std()),
                "velocity_min": int(velocity_array.min()),
                "velocity_max": int(velocity_array.max()),
            }
        )

    # 4.5 ステレオ分離ステムから左右パン (CC#10) と持続系楽器の動的強弱 (CC#11) を抽出・適用する。
    pan_expression_by_track: dict[tuple[str, int, bool, str], dict[str, Any]] = {}
    if config.optimize_pan_cc10 or config.optimize_expression_cc11:
        from .expression_pan import ExpressionPanConfig, apply_pan_and_expression_to_stem_midi

        expression_pan_config = ExpressionPanConfig(
            sample_rate=sample_rate,
            enable_pan_cc10=bool(config.optimize_pan_cc10),
            enable_expression_cc11=bool(config.optimize_expression_cc11),
            max_velocity=int(config.max_velocity),
        )
        for stem_name in stems_with_notes:
            stem_reports = apply_pan_and_expression_to_stem_midi(
                loaded_midis[stem_name],
                target_stereo_waveforms[stem_name],
                stem_name=stem_name,
                config=expression_pan_config,
            )
            for item in stem_reports:
                key = (
                    str(item["stem_name"]),
                    int(item["program_number"]),
                    bool(item["is_drum"]),
                    str(item["instrument_name"]),
                )
                pan_expression_by_track[key] = item

        for track_report in track_reports:
            key = (
                str(track_report["stem_name"]),
                int(track_report["program_number"]),
                bool(track_report["is_drum"]),
                str(track_report["instrument_name"]),
            )
            extra = pan_expression_by_track.get(key)
            if extra is not None:
                for field_name in (
                    "base_pan",
                    "pan_events",
                    "pan_min",
                    "pan_max",
                    "expression_applied",
                    "expression_events",
                    "modulated_phrases",
                ):
                    if field_name in extra:
                        track_report[field_name] = extra[field_name]

    written_stem_paths: dict[str, str] = {}
    if output_stem_dir is not None:
        stem_output_directory = Path(output_stem_dir)
        stem_output_directory.mkdir(parents=True, exist_ok=True)
        for stem_name, midi_object in loaded_midis.items():
            destination = stem_output_directory / f"{resolved_midis[stem_name].stem}_rc_velocity.mid"
            midi_object.write(str(destination))
            written_stem_paths[stem_name] = str(destination.resolve())

    merged_output_resolved: str | None = None
    if output_midi_path is not None:
        merged_midi = _merge_optimized_stem_midis(
            loaded_midis,
            max_melodic_instruments=int(config.max_melodic_instruments),
        )
        destination_path = Path(output_midi_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        merged_midi.write(str(destination_path))
        merged_output_resolved = str(destination_path.resolve())

    return {
        "stems": stems_with_notes,
        "note_count": int(len(notes)),
        "track_count": int(len(track_reports)),
        "stereo_splits": stereo_split_reports,
        "iterations": iteration_history,
        "tracks": track_reports,
        "output_midi_path": merged_output_resolved,
        "output_stem_paths": written_stem_paths,
    }
