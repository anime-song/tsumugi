"""ビート情報と Render & Compare による音声参照 MIDI タイミング量子化。

分離ステムから採譜した MIDI は、分離ノイズや低音・持続音の緩やかな立ち上がりによって
系統的なタイミングバイアスとノートごとのランダムジッターを含む。
一方で全ノートを機械的に 16 分音符へ強制量子化すると、3 連符・速いフィル・アルペジオ・
意図的なレイドバック・ビート誤検出区間を壊し、かえって元音源から遠ざかる。

本モジュールでは以下の 4 段階で「元音源に近づく量子化だけを採用する」閉ループ処理を行う。
1. ビート位相アラインメント:
   原曲（またはドラム等打撃系ステム）のオンセット包絡線に対してビート時刻全体を
   サブフレーム精度で微調整し、ビートトラッカー固有の数〜十数 ms の位相ズレを取り除く。
2. 音楽的保護ルール付きの候補量子化バリアント生成:
   original / soft_16th / tight_16th / triplet_12th / fine_32nd の候補を生成する。
   トラックごとの系統バイアスを差し引いた位相で最近傍グリッドを判定し誤スナップを防ぎ、
   同時和音揃え・昇降アルペジオ保護・同ピッチ衝突回避を行う。
3. 1 パック MIDI × FluidSynth による小節単位 Render & Compare 選択:
   全ステム × 全候補を 1 本の結合 MIDI に直列パックして FluidSynth を 1 回だけ起動し、
   各ステム・各小節ごとに元ステム音声とのマルチバンド・オンセット距離＋正規化ログメル距離を比較。
   Viterbi 連続性制約と劣化防止ガードのもとで最適候補を小節単位に選択する。
4. フルミックス検証およびテンポマップ埋め込み:
   量子化前後を全ステム同時再生で比較して元音源への接近量を定量レポートし、
   必要に応じて位相アラインメント済みビート・拍子情報をテンポマップとして書き込む。
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import soundfile as sf
import torch
import torchaudio.functional as audio_functional

from ..beat_chord.sheetsage import sheetsage_beats_to_beat_grid
from ..beat_chord.tempo_map_export import MeterSegmentSpec, export_tempo_mapped_midi
from ..soundfont import (
    FluidSynthRenderOptions,
    RENDER_COMPARE_SPEED_PRESETS,
    default_parallel_render_workers,
    is_libfluidsynth_available,
    render_midi_batch_to_stereo_arrays,
    render_midi_to_stereo_array,
    render_midi_with_fluidsynth,
    resolve_signal_processing_device,
    write_packed_smf_type0,
)
from ..velocity.render_compare_velocity import _merge_optimized_stem_midis

# ドラム専用チャンネル (9) を除いた 15 本のメロディチャンネル（MIDI 16ch 制約準拠）。
_MELODIC_MIDI_CHANNELS = tuple(channel for channel in range(16) if channel != 9)
_PACKED_TICKS_PER_BEAT = 480
_PACKED_TEMPO_MICROSECONDS = 500_000  # 120 BPM -> 1 秒 = 960 tick
_PACKED_TICKS_PER_SECOND = float(_PACKED_TICKS_PER_BEAT) * 1_000_000.0 / float(_PACKED_TEMPO_MICROSECONDS)

_PackedMidiRenderer = Callable[[Path, Path, int], None]


@dataclass(frozen=True)
class BeatGridInput:
    """量子化とテンポマップ書き出しに使うビート・ダウンビート・拍子情報。"""

    beat_times: tuple[float, ...]
    downbeat_times: tuple[float, ...] = ()
    meter_segments: tuple[MeterSegmentSpec, ...] = ()

    def shifted(self, shift_seconds: float) -> "BeatGridInput":
        """ビート・ダウンビート・小節区間全体を一律に shift_seconds だけ平行移動する。"""
        delta = float(shift_seconds)
        shifted_beats = tuple(max(0.0, float(time) + delta) for time in self.beat_times)
        shifted_downbeats = tuple(max(0.0, float(time) + delta) for time in self.downbeat_times)
        shifted_segments = tuple(
            MeterSegmentSpec(
                start_seconds=max(0.0, float(segment.start_seconds) + delta),
                end_seconds=max(0.0, float(segment.end_seconds) + delta),
                numerator=int(segment.numerator),
                denominator=int(segment.denominator),
                bar_count=int(segment.bar_count),
                score=float(segment.score),
                gap_filled_beats=int(segment.gap_filled_beats),
            )
            for segment in self.meter_segments
            if float(segment.end_seconds) + delta > max(0.0, float(segment.start_seconds) + delta) + 1e-4
        )
        return BeatGridInput(
            beat_times=shifted_beats,
            downbeat_times=shifted_downbeats,
            meter_segments=shifted_segments,
        )


@dataclass(frozen=True)
class _QuantizeCandidateSpec:
    """1 つの量子化候補バリアントの定義。"""

    name: str
    subdivisions_per_beat: int  # 0 は量子化なし (original)
    strength: float  # 0.0..1.0 の引き寄せ強度
    prior_cost: float  # 音響距離に加算する事前コスト（負ほど優先）
    vocal_prior_cost: float  # ボーカルステム用の事前コスト


DEFAULT_QUANTIZE_CANDIDATES: tuple[_QuantizeCandidateSpec, ...] = (
    _QuantizeCandidateSpec(
        name="original",
        subdivisions_per_beat=0,
        strength=0.0,
        prior_cost=0.0,
        vocal_prior_cost=0.0,
    ),
    _QuantizeCandidateSpec(
        name="soft_16th",
        subdivisions_per_beat=4,
        strength=0.55,
        prior_cost=-0.004,
        vocal_prior_cost=-0.002,
    ),
    _QuantizeCandidateSpec(
        name="tight_16th",
        subdivisions_per_beat=4,
        strength=1.0,
        prior_cost=-0.006,
        vocal_prior_cost=0.006,
    ),
    _QuantizeCandidateSpec(
        name="triplet_12th",
        subdivisions_per_beat=3,
        strength=1.0,
        prior_cost=0.004,
        vocal_prior_cost=0.010,
    ),
    _QuantizeCandidateSpec(
        name="fine_32nd",
        subdivisions_per_beat=8,
        strength=1.0,
        prior_cost=0.006,
        vocal_prior_cost=0.012,
    ),
)


@dataclass(frozen=True)
class RenderCompareQuantizeConfig:
    """Render & Compare 型ビート量子化の設定。"""

    soundfont_path: Path
    fluidsynth_executable: Path | None = None
    # オンセット照合は 16 kHz / リバーブ・コーラス無効にするとアタックが鮮明かつ高速になる。
    sample_rate: int = 16_000
    slot_gap_seconds: float = 0.6
    render_gain: float = 0.5
    disable_reverb: bool = True
    disable_chorus: bool = True
    # ビートグリッド位相アラインメント探索範囲（秒）。
    align_beats_to_audio: bool = True
    max_beat_phase_shift_seconds: float = 0.030
    beat_phase_step_seconds: float = 0.002
    # 量子化時の音楽的保護パラメータ。
    max_snap_grid_ratio: float = 0.45
    max_snap_seconds: float = 0.055
    debias_min_seconds: float = 0.008
    debias_max_seconds: float = 0.035
    chord_cluster_seconds: float = 0.022
    arpeggio_min_gap_seconds: float = 0.012
    arpeggio_max_gap_seconds: float = 0.055
    # スライド／チョーキング／フォールなど、ピッチベンドの代わりに短い隣接音程ノートで表現された
    # 装飾列を検出し、主音（アンカー）だけ量子化して経過音の相対タイミングをそのまま保つための閾値。
    slide_max_semitones: int = 2
    slide_max_grace_duration_seconds: float = 0.115
    slide_max_onset_gap_seconds: float = 0.115
    slide_fall_max_release_gap_seconds: float = 0.035
    min_note_duration_seconds: float = 0.020
    # 同一ピッチの連続ノートが量子化で押し潰されないための最小開始間隔・リトリガー隙間。
    min_same_pitch_onset_gap_seconds: float = 0.035
    same_pitch_retrigger_gap_seconds: float = 0.008
    # 音響タイミング距離と小節別 Viterbi 選択のパラメータ。
    onset_distance_weight: float = 0.65
    mel_distance_weight: float = 0.35
    bar_boundary_margin_seconds: float = 0.035
    viterbi_transition_penalty: float = 0.008
    # 生の音響距離が original よりこの量を超えて悪化した小節は安全に original へ戻す。
    max_regression_tolerance: float = 0.005
    # 検証パス（量子化前後フルミックス比較）と出力テンポマップ書き出し。
    verify_full_mix: bool = True
    write_tempo_map: bool = True
    max_melodic_instruments: int = 15
    # True のとき、各ステムの 3 連符・32 分音符の出現頻度に応じて不要候補を省略し高速化する。
    adaptive_candidates: bool = False
    parallel_render_workers: int | None = None

    @classmethod
    def for_speed_preset(
        cls,
        soundfont_path: Path | str,
        *,
        speed: str = "balanced",
        fluidsynth_executable: Path | str | None = None,
        max_melodic_instruments: int = 15,
        write_tempo_map: bool = False,
        parallel_render_workers: int | None = None,
    ) -> "RenderCompareQuantizeConfig":
        """速度プリセット (`fast` / `balanced` / `quality`) から量子化設定を構築する。"""
        normalized_speed = str(speed).strip().lower()
        if normalized_speed not in RENDER_COMPARE_SPEED_PRESETS:
            raise ValueError(f"speed must be one of {RENDER_COMPARE_SPEED_PRESETS}, got {speed!r}")
        resolved_executable = Path(fluidsynth_executable) if fluidsynth_executable is not None else None
        if normalized_speed == "fast":
            return cls(
                soundfont_path=Path(soundfont_path),
                fluidsynth_executable=resolved_executable,
                max_melodic_instruments=int(max_melodic_instruments),
                write_tempo_map=bool(write_tempo_map),
                verify_full_mix=False,
                adaptive_candidates=True,
                parallel_render_workers=parallel_render_workers,
            )
        if normalized_speed == "balanced":
            # 並列実行では全 5 候補を評価しても時間差がごくわずかなため、精度維持のため全候補を残し検証パスのみ省略する。
            return cls(
                soundfont_path=Path(soundfont_path),
                fluidsynth_executable=resolved_executable,
                max_melodic_instruments=int(max_melodic_instruments),
                write_tempo_map=bool(write_tempo_map),
                verify_full_mix=False,
                adaptive_candidates=False,
                parallel_render_workers=parallel_render_workers,
            )
        return cls(
            soundfont_path=Path(soundfont_path),
            fluidsynth_executable=resolved_executable,
            max_melodic_instruments=int(max_melodic_instruments),
            write_tempo_map=bool(write_tempo_map),
            verify_full_mix=True,
            adaptive_candidates=False,
            parallel_render_workers=parallel_render_workers,
        )

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.slot_gap_seconds < 0.0:
            raise ValueError("slot_gap_seconds must be nonnegative")
        if self.max_beat_phase_shift_seconds < 0.0 or self.beat_phase_step_seconds <= 0.0:
            raise ValueError("beat phase alignment parameters are invalid")
        if not 0.0 < self.max_snap_grid_ratio <= 0.5:
            raise ValueError("max_snap_grid_ratio must be in (0, 0.5]")
        if self.max_snap_seconds <= 0.0:
            raise ValueError("max_snap_seconds must be positive")


@dataclass(frozen=True)
class _PackedVariantSlot:
    """直列パック MIDI 内に配置された 1 スロット（ステム × 候補、またはフルミックス）。"""

    slot_key: str
    stem_name: str
    candidate_name: str
    slot_start_seconds: float
    duration_seconds: float


def load_beat_grid_from_json(json_path: Path | str, *, duration_seconds: float | None = None) -> BeatGridInput:
    """SheetSage2-Refined の `*.beats.json` から BeatGridInput を読み込む。"""
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    resolved_duration = float(duration_seconds) if duration_seconds is not None else float(payload.get("duration", 0.0))
    if resolved_duration <= 0.0 and payload.get("beats"):
        resolved_duration = float(payload["beats"][-1]) + 1.0
    sheetsage_grid = sheetsage_beats_to_beat_grid(payload, resolved_duration)
    return BeatGridInput(
        beat_times=sheetsage_grid.beat_times,
        downbeat_times=sheetsage_grid.downbeat_times,
        meter_segments=sheetsage_grid.meter_segments,
    )


def load_beat_grid_from_midi(midi_path: Path | str) -> BeatGridInput:
    """テンポマップ付き MIDI からビート・ダウンビート・拍子区間を読み込む。"""
    midi_object = pretty_midi.PrettyMIDI(str(midi_path))
    beat_times = tuple(float(time) for time in midi_object.get_beats().tolist())
    downbeat_times = tuple(float(time) for time in midi_object.get_downbeats().tolist())
    meter_segments: list[MeterSegmentSpec] = []
    if len(downbeat_times) >= 2:
        signature_changes = sorted(midi_object.time_signature_changes, key=lambda item: float(item.time))
        for index in range(len(downbeat_times) - 1):
            start_seconds = float(downbeat_times[index])
            end_seconds = float(downbeat_times[index + 1])
            if end_seconds <= start_seconds + 1e-4:
                continue
            numerator = 4
            denominator = 4
            for signature in signature_changes:
                if float(signature.time) <= start_seconds + 1e-3:
                    numerator = int(signature.numerator)
                    denominator = int(signature.denominator)
                else:
                    break
            meter_segments.append(
                MeterSegmentSpec(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    numerator=numerator,
                    denominator=denominator,
                    bar_count=1,
                )
            )
    return BeatGridInput(
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        meter_segments=tuple(meter_segments),
    )


def _load_mono_audio(audio_path: Path, *, target_sample_rate: int) -> np.ndarray:
    """音声ファイルを target_sample_rate のモノラル float32 波形 [T] として読み込む。"""
    waveform_data, source_sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = waveform_data.mean(axis=1).astype(np.float32, copy=False)
    if int(source_sample_rate) != int(target_sample_rate):
        tensor = torch.from_numpy(mono.copy()).unsqueeze(0)
        mono = audio_functional.resample(tensor, int(source_sample_rate), int(target_sample_rate)).squeeze(0).numpy()
    return mono.astype(np.float32, copy=False)


def _compute_timing_features(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    mel_filterbank: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
) -> tuple[np.ndarray, np.ndarray, float]:
    """波形 [T] から高時間分解能の 4 帯域オンセット包絡線 [4, F] とログメル [M, F] を抽出する。

    16 kHz / hop=128 のとき 1 フレーム = 8.0 ms。
    """
    if waveform.size < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.size))
    signal_device = resolve_signal_processing_device()
    with torch.inference_mode():
        wave_tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).to(signal_device)
        window = torch.hann_window(n_fft, dtype=torch.float32, device=signal_device)
        spectrum = torch.stft(
            wave_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        power_tensor = spectrum.real.square() + spectrum.imag.square()
        log_magnitude = torch.log1p(10.0 * torch.sqrt(power_tensor.clamp_min(1e-12)))

        # 1. 半波整流スペクトルフラックスを 4 帯域へ集約し、キック・スネア・コード・シンバルのアタックを捉える。
        flux = torch.zeros_like(log_magnitude)
        if log_magnitude.shape[1] > 1:
            flux[:, 1:] = torch.clamp_min(log_magnitude[:, 1:] - log_magnitude[:, :-1], 0.0)
        frequencies = torch.linspace(
            0.0, float(sample_rate) * 0.5, int(power_tensor.shape[0]), dtype=torch.float32, device=signal_device
        )
        band_edges = (0.0, 200.0, 800.0, 3200.0, float(sample_rate) * 0.5 + 1.0)
        band_rows = []
        for band_index in range(4):
            mask = (frequencies >= band_edges[band_index]) & (frequencies < band_edges[band_index + 1])
            if bool(mask.any().item()):
                band_rows.append(flux[mask].mean(dim=0))
            else:
                band_rows.append(torch.zeros(flux.shape[1], dtype=torch.float32, device=signal_device))
        multiband_onset = torch.stack(band_rows, dim=0).cpu().numpy()

        # 2. ログメルスペクトログラム（持続音や和音チェンジの時間輪郭比較用）。
        mel_tensor = torch.from_numpy(np.ascontiguousarray(mel_filterbank, dtype=np.float32)).to(signal_device)
        log_mel = torch.log1p(10.0 * torch.clamp_min(mel_tensor @ power_tensor, 0.0)).cpu().numpy()
    frames_per_second = float(sample_rate) / float(hop_length)
    return multiband_onset, log_mel, frames_per_second


def _build_mel_filterbank(*, sample_rate: int, n_fft: int = 512, num_mels: int = 40) -> np.ndarray:
    """三角メルフィルタバンク [num_mels, n_fft // 2 + 1] を生成する。"""
    num_bins = n_fft // 2 + 1
    frequencies = np.linspace(0.0, float(sample_rate) * 0.5, num_bins, dtype=np.float64)

    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(_hz_to_mel(40.0), _hz_to_mel(min(7600.0, sample_rate * 0.48)), num_mels + 2)
    hz_points = np.asarray([_mel_to_hz(float(mel)) for mel in mel_points], dtype=np.float64)
    filterbank = np.zeros((num_mels, num_bins), dtype=np.float32)
    for mel_index in range(num_mels):
        left_hz, center_hz, right_hz = hz_points[mel_index], hz_points[mel_index + 1], hz_points[mel_index + 2]
        rising = (frequencies - left_hz) / max(center_hz - left_hz, 1e-6)
        falling = (right_hz - frequencies) / max(right_hz - center_hz, 1e-6)
        weights = np.maximum(0.0, np.minimum(rising, falling))
        weight_sum = float(weights.sum())
        if weight_sum > 0.0:
            weights /= weight_sum
        filterbank[mel_index] = weights.astype(np.float32)
    return filterbank


def align_beat_grid_to_audio(
    beat_grid: BeatGridInput,
    reference_waveform: np.ndarray,
    *,
    sample_rate: int,
    max_shift_seconds: float = 0.030,
    step_seconds: float = 0.002,
) -> tuple[BeatGridInput, float, float]:
    """参照音声のオンセット包絡線に合うようビートグリッド全体の微小位相オフセットを探索する。

    戻り値は (アラインメント済み BeatGridInput, 最適シフト秒, オンセットスコア改善率)。
    """
    if len(beat_grid.beat_times) < 4 or max_shift_seconds <= 0.0:
        return beat_grid, 0.0, 0.0

    # 因果的な前後対比（時刻 t 直後 12 ms と直前 12 ms の対数 RMS 差）を使うことで、
    # STFT の center=True 窓による数 ms の先読み位相バイアスをなくし、真のアタック開始点 t0 にピークを立てる。
    waveform = reference_waveform.astype(np.float64, copy=False)
    window_samples = max(16, int(round(0.012 * float(sample_rate))))
    if waveform.size < window_samples * 3:
        return beat_grid, 0.0, 0.0
    # 一次プリエンファシスで打撃アタック（高域過渡成分）を強調する。
    emphasized = np.empty_like(waveform)
    emphasized[0] = waveform[0]
    emphasized[1:] = waveform[1:] - 0.95 * waveform[:-1]
    squared = np.square(emphasized)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))

    # 時刻サンプル s に対し、直後窓 [s, s + W] と直前窓 [s - W, s] の対数 RMS 差を評価する。
    def _onset_contrast_at_samples(sample_indices: np.ndarray) -> np.ndarray:
        valid = (sample_indices >= window_samples) & (sample_indices + window_samples <= waveform.size)
        if not np.any(valid):
            return np.zeros(0, dtype=np.float64)
        centers = sample_indices[valid]
        after_rms = np.sqrt((cumulative[centers + window_samples] - cumulative[centers]) / float(window_samples))
        before_rms = np.sqrt((cumulative[centers] - cumulative[centers - window_samples]) / float(window_samples))
        return np.maximum(0.0, np.log1p(25.0 * after_rms) - np.log1p(25.0 * before_rms))

    beats_array = np.asarray(beat_grid.beat_times, dtype=np.float64)
    num_steps = int(round(max_shift_seconds / step_seconds))
    candidate_shifts = np.linspace(-num_steps * step_seconds, num_steps * step_seconds, 2 * num_steps + 1)
    best_shift = 0.0
    zero_score = 0.0
    best_score = -1.0

    for shift in candidate_shifts.tolist():
        sample_indices = np.rint((beats_array + shift) * float(sample_rate)).astype(np.int64)
        contrasts = _onset_contrast_at_samples(sample_indices)
        if contrasts.size == 0:
            continue
        score = float(np.mean(contrasts))
        if abs(shift) < 1e-9:
            zero_score = score
        if score > best_score:
            best_score = score
            best_shift = float(shift)

    if best_score <= 1e-12:
        return beat_grid, 0.0, 0.0
    if zero_score > 1e-12 and best_score <= zero_score * 1.015:
        return beat_grid, 0.0, 0.0

    improvement_ratio = (best_score / zero_score) - 1.0 if zero_score > 1e-12 else 1.0
    return beat_grid.shifted(best_shift), best_shift, float(improvement_ratio)


def _seconds_to_beat_position(seconds: float, beat_times: np.ndarray) -> tuple[float, float]:
    """時刻 (秒) を連続拍位置と局所 1 拍の長さ (秒) へ変換する。"""
    if beat_times.size < 2:
        return float(seconds) * 2.0, 0.5
    if seconds <= float(beat_times[0]):
        first_duration = max(1e-4, float(beat_times[1] - beat_times[0]))
        return float(seconds - beat_times[0]) / first_duration, first_duration
    if seconds >= float(beat_times[-1]):
        last_duration = max(1e-4, float(beat_times[-1] - beat_times[-2]))
        return float(beat_times.size - 1) + float(seconds - beat_times[-1]) / last_duration, last_duration

    index = int(np.searchsorted(beat_times, seconds, side="right")) - 1
    index = max(0, min(int(beat_times.size) - 2, index))
    beat_start = float(beat_times[index])
    beat_duration = max(1e-4, float(beat_times[index + 1] - beat_start))
    return float(index) + (float(seconds) - beat_start) / beat_duration, beat_duration


def _beat_position_to_seconds(beat_position: float, beat_times: np.ndarray) -> float:
    """連続拍位置を時刻 (秒) へ逆変換する。"""
    if beat_times.size < 2:
        return max(0.0, float(beat_position) * 0.5)
    if beat_position <= 0.0:
        first_duration = max(1e-4, float(beat_times[1] - beat_times[0]))
        return max(0.0, float(beat_times[0]) + beat_position * first_duration)
    last_index = int(beat_times.size) - 1
    if beat_position >= float(last_index):
        last_duration = max(1e-4, float(beat_times[-1] - beat_times[-2]))
        return max(0.0, float(beat_times[-1]) + (beat_position - float(last_index)) * last_duration)

    lower_index = int(math.floor(beat_position))
    fraction = beat_position - float(lower_index)
    beat_start = float(beat_times[lower_index])
    beat_end = float(beat_times[lower_index + 1])
    return max(0.0, beat_start + fraction * (beat_end - beat_start))


def _estimate_track_timing_bias_seconds(
    notes: Sequence[pretty_midi.Note],
    beat_times: np.ndarray,
    *,
    config: RenderCompareQuantizeConfig,
) -> float:
    """トラック全体の系統的なオンセット遅延／先行バイアス（16分音符グリッド比の中央値）を推定する。

    系統バイアスが大きい楽器（例: +28ms 遅れるベース）で隣の 16 分音符へ誤スナップするのを防ぐために使う。
    """
    if len(notes) < 8 or beat_times.size < 2:
        return 0.0
    signed_errors: list[float] = []
    for note in notes:
        beat_position, beat_duration = _seconds_to_beat_position(float(note.start), beat_times)
        nearest_sixteenth = round(beat_position * 4.0) / 4.0
        signed_errors.append((beat_position - nearest_sixteenth) * beat_duration)
    median_bias = float(np.median(signed_errors))
    if abs(median_bias) < float(config.debias_min_seconds):
        return 0.0
    return float(np.clip(median_bias, -config.debias_max_seconds, config.debias_max_seconds))


def _resolve_same_pitch_collisions(
    notes: list[pretty_midi.Note],
    original_starts_by_id: Mapping[int, float] | None,
    *,
    config: RenderCompareQuantizeConfig,
) -> None:
    """同一ピッチのノート重複・極短押し潰し・リトリガー消失を防ぐ。

    量子化によって 2 つの同ピッチノートが同一グリッドへ近づきすぎた場合は、
    後続ノートを無理に 20 ms へ押し込まず元の相対間隔を維持し、先行ノート終端と
    後続ノート開始の間にリトリガー隙間を確保する。
    """
    notes_by_pitch: dict[int, list[pretty_midi.Note]] = {}
    for note in notes:
        notes_by_pitch.setdefault(int(note.pitch), []).append(note)

    min_duration = float(config.min_note_duration_seconds)
    min_onset_gap = float(config.min_same_pitch_onset_gap_seconds)
    retrigger_gap = float(config.same_pitch_retrigger_gap_seconds)

    for pitch_notes in notes_by_pitch.values():
        if original_starts_by_id is not None:
            pitch_notes.sort(key=lambda item: (original_starts_by_id.get(id(item), float(item.start)), float(item.end)))
        else:
            pitch_notes.sort(key=lambda item: (float(item.start), float(item.end)))

        for position in range(1, len(pitch_notes)):
            previous = pitch_notes[position - 1]
            current = pitch_notes[position]
            current_duration = max(min_duration, float(current.end) - float(current.start))

            if original_starts_by_id is not None:
                original_gap = max(
                    0.0,
                    original_starts_by_id.get(id(current), float(current.start))
                    - original_starts_by_id.get(id(previous), float(previous.start)),
                )
            else:
                original_gap = max(0.0, float(current.start) - float(previous.start))

            # 元々しっかり離れていた同ピッチ連打が量子化で極端に接近／逆転した場合は、
            # 先行ノートからの元の相対間隔を保って発音潰れを防ぐ。
            required_gap = min(original_gap, max(min_onset_gap, original_gap * 0.65)) if original_gap >= min_duration else max(min_duration, original_gap)
            actual_gap = float(current.start) - float(previous.start)
            if actual_gap < required_gap - 1e-4:
                preserved_gap = max(original_gap, min_duration + retrigger_gap)
                current.start = float(previous.start) + preserved_gap
                current.end = max(float(current.start) + min_duration, float(current.start) + current_duration)

            # 先行ノートの終端が後続ノートの開始に被る（またはリトリガー隙間がない）場合は手前で切る。
            max_previous_end = float(current.start) - retrigger_gap
            if float(previous.end) > max_previous_end:
                previous.end = max(float(previous.start) + min(min_duration, max(0.008, (float(current.start) - float(previous.start)) * 0.7)), max_previous_end)


def _detect_slide_and_grace_chains(
    ordered_notes: Sequence[pretty_midi.Note],
    *,
    is_drum: bool,
    config: RenderCompareQuantizeConfig,
) -> list[list[int]]:
    """ピッチベンドの代わりに短い隣接音程ノートで表現されたスライド／チョーキング／フォール列を検出する。

    戻り値は ordered_notes 上のインデックス列のリスト（各チェーン長 >= 2）。
    チェーン内では主音（最長ノート）をアンカーにして平行移動し、経過音の相対間隔を完全保護する。
    """
    if is_drum or len(ordered_notes) < 2:
        return []

    max_semitones = int(config.slide_max_semitones)
    max_grace_duration = float(config.slide_max_grace_duration_seconds)
    max_onset_gap = float(config.slide_max_onset_gap_seconds)
    max_fall_release_gap = float(config.slide_fall_max_release_gap_seconds)

    linked_next: list[bool] = [False] * (len(ordered_notes) - 1)
    for index in range(len(ordered_notes) - 1):
        left = ordered_notes[index]
        right = ordered_notes[index + 1]
        pitch_diff = abs(int(right.pitch) - int(left.pitch))
        if not (1 <= pitch_diff <= max_semitones):
            continue
        onset_gap = float(right.start) - float(left.start)
        left_duration = float(left.end) - float(left.start)
        right_duration = float(right.end) - float(right.start)

        # 1. 頭スライド・チョーキング・ハンマリング・前打音: 先行音が短く直後に隣接音程へ繋がる。
        is_lead_in_slide = (
            left_duration <= max_grace_duration
            and 0.006 <= onset_gap <= max_onset_gap
            and float(right.start) <= float(left.end) + max_fall_release_gap
        )
        # 2. フォール・ベンドダウン（尻スライド）: 主音の終端付近から短い隣接音程へ落ちる／滑る。
        is_tail_fall_slide = (
            right_duration <= max_grace_duration
            and onset_gap >= 0.010
            and abs(float(right.start) - float(left.end)) <= max_fall_release_gap
        )
        if is_lead_in_slide or is_tail_fall_slide:
            linked_next[index] = True

    chains: list[list[int]] = []
    index = 0
    while index < len(ordered_notes) - 1:
        if not linked_next[index]:
            index += 1
            continue
        chain = [index]
        while index < len(ordered_notes) - 1 and linked_next[index]:
            index += 1
            chain.append(index)
        chains.append(chain)
        index += 1
    return chains


def _build_quantized_instrument_variant(
    source_instrument: pretty_midi.Instrument,
    *,
    beat_times: np.ndarray,
    candidate: _QuantizeCandidateSpec,
    config: RenderCompareQuantizeConfig,
) -> tuple[pretty_midi.Instrument, list[tuple[float, float]]]:
    """1 トラック分の量子化候補 Instrument と、元ノートのソート順に対応する (start, end) 列を生成する。

    元ノート列は (start, pitch, end) 昇順で固定インデックス化し、返却するタイミング列も
    必ず同一インデックス順で返す（小節合成時のノート取り違え・消失を構造上ゼロにする）。
    """
    ordered_source_notes = sorted(source_instrument.notes, key=lambda note: (note.start, note.pitch, note.end))
    cloned = pretty_midi.Instrument(
        program=int(source_instrument.program),
        is_drum=bool(source_instrument.is_drum),
        name=str(source_instrument.name),
    )
    cloned.notes = [
        pretty_midi.Note(
            velocity=int(note.velocity),
            pitch=int(note.pitch),
            start=float(note.start),
            end=float(note.end),
        )
        for note in ordered_source_notes
    ]
    cloned.control_changes = [
        pretty_midi.ControlChange(number=int(cc.number), value=int(cc.value), time=float(cc.time))
        for cc in source_instrument.control_changes
    ]
    cloned.pitch_bends = [
        pretty_midi.PitchBend(pitch=int(bend.pitch), time=float(bend.time))
        for bend in source_instrument.pitch_bends
    ]
    if candidate.subdivisions_per_beat <= 0 or candidate.strength <= 0.0 or not cloned.notes:
        timings = [(float(note.start), float(note.end)) for note in cloned.notes]
        return cloned, timings

    bias_seconds = _estimate_track_timing_bias_seconds(
        cloned.notes,
        beat_times,
        config=config,
    )
    subdivisions = float(candidate.subdivisions_per_beat)
    strength = float(candidate.strength)

    # 1. ピッチベンド代替のスライド／チョーキング／フォール列を最優先で検出し、剛体保護する。
    #    主音（最長ノート）をグリッドへスナップし、経過音は全く同じシフト量だけ平行移動する。
    slide_chains = _detect_slide_and_grace_chains(
        cloned.notes,
        is_drum=bool(cloned.is_drum),
        config=config,
    )
    protected_indices: set[int] = set()
    proposed_starts: dict[int, float] = {}

    for chain in slide_chains:
        protected_indices.update(chain)
        # チェーン内で最も長いノート（頭スライドなら着地音、フォールなら先頭主音）をアンカーにする。
        anchor_index = max(
            chain,
            key=lambda note_index: (
                float(cloned.notes[note_index].end) - float(cloned.notes[note_index].start),
                -note_index,
            ),
        )
        anchor_start = float(cloned.notes[anchor_index].start)
        snapped_anchor = _snap_time_to_grid(
            anchor_start,
            beat_times=beat_times,
            subdivisions=subdivisions,
            strength=strength,
            bias_seconds=bias_seconds,
            config=config,
        )
        shift = snapped_anchor - anchor_start
        # 先頭ノートが負時刻にならない範囲にシフト量を制限する。
        min_chain_start = min(float(cloned.notes[note_index].start) for note_index in chain)
        if min_chain_start + shift < 0.0:
            shift = -min_chain_start
        for note_index in chain:
            proposed_starts[note_index] = max(0.0, float(cloned.notes[note_index].start) + shift)

    # 2. スライド以外の近接ノートをグルーピングし、「昇降アルペジオ」と「同時和音」を判別する。
    remaining_indices = [index for index in range(len(cloned.notes)) if index not in protected_indices]
    groups: list[list[int]] = []
    for note_index in remaining_indices:
        current_note = cloned.notes[note_index]
        if not groups:
            groups.append([note_index])
            continue
        previous_note = cloned.notes[groups[-1][-1]]
        gap = float(current_note.start) - float(previous_note.start)
        if not cloned.is_drum and gap <= float(config.arpeggio_max_gap_seconds) and int(current_note.pitch) != int(previous_note.pitch):
            groups[-1].append(note_index)
        else:
            groups.append([note_index])

    for group in groups:
        starts = [float(cloned.notes[note_index].start) for note_index in group]
        pitches = [int(cloned.notes[note_index].pitch) for note_index in group]
        gaps = [right - left for left, right in zip(starts[:-1], starts[1:])]
        is_monotonic_arpeggio = (
            len(group) >= 3
            and all(float(config.arpeggio_min_gap_seconds) <= gap <= float(config.arpeggio_max_gap_seconds) for gap in gaps)
            and (
                all(right > left for left, right in zip(pitches[:-1], pitches[1:]))
                or all(right < left for left, right in zip(pitches[:-1], pitches[1:]))
            )
        )
        is_tight_chord = len(group) >= 2 and (starts[-1] - starts[0]) <= float(config.chord_cluster_seconds)

        if is_monotonic_arpeggio:
            anchor_start = starts[0]
            snapped_anchor = _snap_time_to_grid(
                anchor_start,
                beat_times=beat_times,
                subdivisions=subdivisions,
                strength=strength,
                bias_seconds=bias_seconds,
                config=config,
            )
            shift = snapped_anchor - anchor_start
            for note_index in group:
                proposed_starts[note_index] = max(0.0, float(cloned.notes[note_index].start) + shift)
        elif is_tight_chord:
            representative_start = float(np.median(starts))
            snapped_chord = _snap_time_to_grid(
                representative_start,
                beat_times=beat_times,
                subdivisions=subdivisions,
                strength=strength,
                bias_seconds=bias_seconds,
                config=config,
            )
            for note_index in group:
                proposed_starts[note_index] = snapped_chord
        else:
            for note_index in group:
                proposed_starts[note_index] = _snap_time_to_grid(
                    float(cloned.notes[note_index].start),
                    beat_times=beat_times,
                    subdivisions=subdivisions,
                    strength=strength,
                    bias_seconds=bias_seconds,
                    config=config,
                )

    # 3. 元のノート長を維持したまま開始時刻を更新し、同ピッチ衝突・極短押し潰しを安全に解消する。
    original_starts_by_id: dict[int, float] = {}
    for note_index, note in enumerate(cloned.notes):
        original_start = float(note.start)
        original_duration = max(float(config.min_note_duration_seconds), float(note.end) - original_start)
        original_starts_by_id[id(note)] = original_start
        new_start = max(0.0, float(proposed_starts[note_index]))
        note.start = new_start
        note.end = max(new_start + float(config.min_note_duration_seconds), new_start + original_duration)

    _resolve_same_pitch_collisions(cloned.notes, original_starts_by_id, config=config)

    # 重要: 元のソート済みインデックス順のまま (start, end) を記録してからレンダリング用にソートする。
    timings_in_source_order = [(float(note.start), float(note.end)) for note in cloned.notes]
    cloned.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
    return cloned, timings_in_source_order


def _snap_time_to_grid(
    original_seconds: float,
    *,
    beat_times: np.ndarray,
    subdivisions: float,
    strength: float,
    bias_seconds: float,
    config: RenderCompareQuantizeConfig,
) -> float:
    """単一のノート開始時刻を、系統バイアス考慮付きで指定分割数の拍グリッドへ引き寄せる。"""
    # 系統バイアスを引いた位相で最近傍グリッドを判定することで、隣の 16 分音符への誤スナップを防ぐ。
    debiased_seconds = max(0.0, float(original_seconds) - float(bias_seconds))
    debiased_position, beat_duration = _seconds_to_beat_position(debiased_seconds, beat_times)
    grid_step_seconds = beat_duration / subdivisions
    snapped_beat_position = round(debiased_position * subdivisions) / subdivisions
    target_seconds = _beat_position_to_seconds(snapped_beat_position, beat_times)

    # バイアスを差し引いた後のズレが許容範囲外（装飾音や別拍）なら移動しない。
    debiased_distance = abs(target_seconds - debiased_seconds)
    max_allowed_distance = min(float(config.max_snap_seconds), grid_step_seconds * float(config.max_snap_grid_ratio))
    if debiased_distance > max_allowed_distance:
        return max(0.0, float(original_seconds))

    # 実際に移動する総量も安全上限（max_snap_seconds + |bias|）以内に収める。
    total_shift = target_seconds - float(original_seconds)
    max_total_shift = float(config.max_snap_seconds) + abs(float(bias_seconds))
    if abs(total_shift) > max_total_shift:
        return max(0.0, float(original_seconds))

    return max(0.0, float(original_seconds) + float(strength) * total_shift)


def _build_bar_windows(beat_grid: BeatGridInput, duration_seconds: float) -> list[tuple[float, float]]:
    """曲全体を重複のない小節区間リスト [(start, end), ...] へ分割する。"""
    if len(beat_grid.downbeat_times) >= 2:
        boundaries = [float(time) for time in beat_grid.downbeat_times if 0.0 < float(time) < duration_seconds]
    elif len(beat_grid.beat_times) >= 5:
        boundaries = [
            float(beat_grid.beat_times[index])
            for index in range(0, len(beat_grid.beat_times), 4)
            if 0.0 < float(beat_grid.beat_times[index]) < duration_seconds
        ]
    else:
        boundaries = []

    unique_boundaries = sorted(set(boundaries))
    points = [0.0, *unique_boundaries, max(0.25, float(duration_seconds))]
    windows: list[tuple[float, float]] = []
    for start_seconds, end_seconds in zip(points[:-1], points[1:]):
        if end_seconds - start_seconds >= 0.08:
            windows.append((float(start_seconds), float(end_seconds)))
    return windows or [(0.0, max(0.25, float(duration_seconds)))]


def _assign_note_to_bar(
    note_start_seconds: float,
    bar_windows: Sequence[tuple[float, float]],
    *,
    boundary_margin_seconds: float,
) -> int:
    """ノート開始時刻が属する小節インデックスを二分探索 O(log B) で返す。

    次小節のダウンビート直前（boundary_margin_seconds 以内）のフライング音符は
    次小節の 1 拍目候補として次小節へ帰属させる。
    """
    window_count = len(bar_windows)
    # end_seconds < note_start_seconds - boundary_margin_seconds を満たす小節は
    # フライング判定・区間包含判定のどちらも確実に偽となるため、二分探索で先頭候補まで飛ばす。
    threshold = note_start_seconds - boundary_margin_seconds
    low = 0
    high = window_count
    while low < high:
        mid = (low + high) // 2
        if bar_windows[mid][1] < threshold:
            low = mid + 1
        else:
            high = mid

    for index in range(low, window_count):
        start_seconds, end_seconds = bar_windows[index]
        if index + 1 < window_count and abs(note_start_seconds - end_seconds) <= boundary_margin_seconds:
            return index + 1
        if start_seconds <= note_start_seconds < end_seconds:
            return index
    return window_count - 1


def _compute_window_timing_distance(
    target_onset: np.ndarray,
    target_mel: np.ndarray,
    rendered_onset: np.ndarray,
    rendered_mel: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    config: RenderCompareQuantizeConfig,
) -> float:
    """1 小節区間における元音声とレンダリング音声のタイミング音響距離（小さいほど一致）を計算する。"""
    target_onset_slice = target_onset[:, start_frame:end_frame].reshape(-1).astype(np.float64)
    rendered_onset_slice = rendered_onset[:, start_frame:end_frame].reshape(-1).astype(np.float64)
    target_onset_norm = float(np.linalg.norm(target_onset_slice))
    rendered_onset_norm = float(np.linalg.norm(rendered_onset_slice))
    if target_onset_norm <= 1e-8 or rendered_onset_norm <= 1e-8:
        onset_distance = 1.0
    else:
        onset_cosine = float(np.dot(target_onset_slice, rendered_onset_slice) / (target_onset_norm * rendered_onset_norm))
        onset_distance = float(np.clip(1.0 - onset_cosine, 0.0, 2.0))

    target_mel_slice = target_mel[:, start_frame:end_frame].astype(np.float64)
    rendered_mel_slice = rendered_mel[:, start_frame:end_frame].astype(np.float64)
    target_centered = (target_mel_slice - target_mel_slice.mean()).reshape(-1)
    rendered_centered = (rendered_mel_slice - rendered_mel_slice.mean()).reshape(-1)
    target_mel_norm = float(np.linalg.norm(target_centered))
    rendered_mel_norm = float(np.linalg.norm(rendered_centered))
    if target_mel_norm <= 1e-8 or rendered_mel_norm <= 1e-8:
        mel_distance = 1.0
    else:
        mel_cosine = float(np.dot(target_centered, rendered_centered) / (target_mel_norm * rendered_mel_norm))
        mel_distance = float(np.clip(1.0 - mel_cosine, 0.0, 2.0))

    total_weight = float(config.onset_distance_weight) + float(config.mel_distance_weight)
    if total_weight <= 0.0:
        return onset_distance
    return (
        float(config.onset_distance_weight) * onset_distance
        + float(config.mel_distance_weight) * mel_distance
    ) / total_weight


def _decode_viterbi_candidate_sequence(
    raw_distances: np.ndarray,
    *,
    prior_costs: Sequence[float],
    transition_penalty: float,
    max_regression_tolerance: float,
) -> list[int]:
    """小節ごとの音響距離行列 [num_bars, num_candidates] から滑らかな最適候補列を復号する。

    candidate 0 は常に original を想定し、生の音響距離が original より許容値以上悪化した
    小節は強制的に original (0) へ安全フォールバックする。
    """
    num_bars, num_candidates = raw_distances.shape
    if num_bars == 0:
        return []

    prior_array = np.asarray(prior_costs, dtype=np.float64)
    costs = raw_distances + prior_array[None, :]
    dp = np.zeros((num_bars, num_candidates), dtype=np.float64)
    backpointers = np.zeros((num_bars, num_candidates), dtype=np.int32)
    dp[0] = costs[0]

    for bar_index in range(1, num_bars):
        for candidate_index in range(num_candidates):
            transition = np.full(num_candidates, float(transition_penalty), dtype=np.float64)
            transition[candidate_index] = 0.0
            total = dp[bar_index - 1] + transition
            best_previous = int(np.argmin(total))
            dp[bar_index, candidate_index] = total[best_previous] + costs[bar_index, candidate_index]
            backpointers[bar_index, candidate_index] = best_previous

    selected = [0] * num_bars
    selected[-1] = int(np.argmin(dp[-1]))
    for bar_index in range(num_bars - 2, -1, -1):
        selected[bar_index] = int(backpointers[bar_index + 1, selected[bar_index + 1]])

    # 劣化防止ガード: original (index=0) より生距離が大きく悪化した小節は original に戻す。
    for bar_index in range(num_bars):
        chosen = selected[bar_index]
        if chosen != 0 and raw_distances[bar_index, chosen] > raw_distances[bar_index, 0] + float(max_regression_tolerance):
            selected[bar_index] = 0
    return selected


def _write_packed_variants_midi(
    slot_specs: Sequence[tuple[str, str, str, Sequence[pretty_midi.Instrument], float]],
    *,
    output_midi_path: Path,
    slot_gap_seconds: float,
) -> list[_PackedVariantSlot]:
    """複数スロット（ステム×候補やフルミックス）を 1 本の直列パック MIDI (SMF Type-0) へ高速書き出しする。"""
    events: list[tuple[int, int, bytes]] = []
    slots: list[_PackedVariantSlot] = []
    cursor_seconds = 0.0
    melodic_channel_offset = 0

    for slot_key, stem_name, candidate_name, instruments, duration_seconds in slot_specs:
        slot_start_seconds = float(cursor_seconds)
        active_instruments = [instrument for instrument in instruments if instrument.notes]
        used_channels: set[int] = set()
        local_melodic_index = 0

        for instrument in active_instruments:
            if instrument.is_drum:
                channel = 9
            else:
                channel = _MELODIC_MIDI_CHANNELS[(melodic_channel_offset + local_melodic_index) % len(_MELODIC_MIDI_CHANNELS)]
                local_melodic_index += 1
            used_channels.add(channel)

            setup_tick = max(0, int(round(slot_start_seconds * _PACKED_TICKS_PER_SECOND)))
            if not instrument.is_drum:
                events.append((setup_tick, 2, bytes((0xC0 | channel, max(0, min(127, int(instrument.program)))))))
            volume_cc7 = next((int(control.value) for control in instrument.control_changes if int(control.number) == 7), 100)
            cc_status = 0xB0 | channel
            events.append((setup_tick, 3, bytes((cc_status, 7, max(0, min(127, volume_cc7))))))
            events.append((setup_tick, 3, bytes((cc_status, 10, 64))))
            events.append((setup_tick, 3, bytes((cc_status, 11, 127))))

            note_on_status = 0x90 | channel
            note_off_status = 0x80 | channel
            for note in instrument.notes:
                start_tick = max(
                    setup_tick,
                    int(round((slot_start_seconds + max(0.0, float(note.start))) * _PACKED_TICKS_PER_SECOND)),
                )
                end_tick = max(
                    start_tick + 1,
                    int(round((slot_start_seconds + max(float(note.start) + 0.02, float(note.end))) * _PACKED_TICKS_PER_SECOND)),
                )
                velocity = max(1, min(127, int(round(note.velocity))))
                pitch = max(0, min(127, int(note.pitch)))
                events.append((start_tick, 5, bytes((note_on_status, pitch, velocity))))
                events.append((end_tick, 4, bytes((note_off_status, pitch, 0))))

        slot_end_seconds = slot_start_seconds + float(duration_seconds)
        stop_tick = int(round((slot_end_seconds + 0.04) * _PACKED_TICKS_PER_SECOND))
        for channel in sorted(used_channels):
            cc_status = 0xB0 | channel
            events.append((stop_tick, 6, bytes((cc_status, 123, 0))))
            events.append((stop_tick, 6, bytes((cc_status, 120, 0))))

        slots.append(
            _PackedVariantSlot(
                slot_key=slot_key,
                stem_name=stem_name,
                candidate_name=candidate_name,
                slot_start_seconds=slot_start_seconds,
                duration_seconds=float(duration_seconds),
            )
        )
        melodic_channel_offset = (melodic_channel_offset + max(1, local_melodic_index)) % len(_MELODIC_MIDI_CHANNELS)
        cursor_seconds = slot_end_seconds + float(slot_gap_seconds)

    write_packed_smf_type0(
        output_midi_path,
        events,
        ticks_per_beat=_PACKED_TICKS_PER_BEAT,
        tempo_microseconds=_PACKED_TEMPO_MICROSECONDS,
    )
    return slots


def _render_packed_midi_with_fluidsynth(
    midi_path: Path,
    wav_path: Path,
    sample_rate: int,
    *,
    config: RenderCompareQuantizeConfig,
) -> None:
    """FluidSynth を起動してパック MIDI を WAV へレンダリングする。"""
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
            polyphony=64,
        ),
    )


def _select_adaptive_candidates_for_stem(
    midi_object: pretty_midi.PrettyMIDI,
    beat_times: np.ndarray,
    candidates: Sequence[_QuantizeCandidateSpec],
) -> list[_QuantizeCandidateSpec]:
    """ステム内の拍内オンセット分布から、3連符 (`subdivisions=3`) と 32分音符 (`subdivisions=8`) の必要性を判定する。"""
    note_starts = [
        float(note.start)
        for instrument in midi_object.instruments
        for note in instrument.notes
    ]
    if len(note_starts) < 8 or beat_times.size < 2:
        return [candidate for candidate in candidates if candidate.subdivisions_per_beat in (0, 4)]

    triplet_hits = 0
    fine_32nd_hits = 0
    for start_seconds in note_starts:
        beat_position, _ = _seconds_to_beat_position(start_seconds, beat_times)
        fraction = beat_position - math.floor(beat_position)
        distance_to_16th = min(abs(fraction - step) for step in (0.0, 0.25, 0.5, 0.75, 1.0))
        distance_to_triplet = min(abs(fraction - (1.0 / 3.0)), abs(fraction - (2.0 / 3.0)))
        if distance_to_triplet <= 0.065 and distance_to_triplet + 0.015 < distance_to_16th:
            triplet_hits += 1
        distance_to_odd_32nd = min(abs(fraction - step) for step in (0.125, 0.375, 0.625, 0.875))
        if distance_to_odd_32nd <= 0.045 and distance_to_odd_32nd + 0.015 < distance_to_16th:
            fine_32nd_hits += 1

    total_notes = float(len(note_starts))
    needs_triplet = triplet_hits >= 6 and (float(triplet_hits) / total_notes) >= 0.10
    needs_fine_32nd = fine_32nd_hits >= 8 and (float(fine_32nd_hits) / total_notes) >= 0.12

    selected: list[_QuantizeCandidateSpec] = []
    for candidate in candidates:
        if candidate.subdivisions_per_beat == 3 and not needs_triplet:
            continue
        if candidate.subdivisions_per_beat >= 8 and not needs_fine_32nd:
            continue
        selected.append(candidate)
    return selected or list(candidates[:1])


def _assemble_stem_midi_from_bar_choices(
    original_midi: pretty_midi.PrettyMIDI,
    variant_timings_by_candidate: Mapping[str, Sequence[Sequence[tuple[float, float]]]],
    candidate_names: Sequence[str],
    selected_candidate_by_bar: Sequence[int],
    bar_windows: Sequence[tuple[float, float]],
    *,
    config: RenderCompareQuantizeConfig,
) -> tuple[pretty_midi.PrettyMIDI, int, float]:
    """小節ごとの選択結果に従って、各候補の元インデックス順タイミング列から 1 本のステム MIDI を合成する。

    各ノートのピッチ・ベロシティは元ノートから直接複製し、(start, end) だけを対応する
    元インデックスから参照するため、ソート順の入れ替わりによるノート消失・ピッチ取り違えが
    構造上発生しない。さらにスライド／装飾音チェーンはアンカー音と同じ小節候補を共有する。
    """
    assembled = copy.deepcopy(original_midi)
    moved_note_count = 0
    total_abs_shift_ms = 0.0

    for instrument_index, original_instrument in enumerate(assembled.instruments):
        if not original_instrument.notes:
            continue
        original_sorted = sorted(original_instrument.notes, key=lambda note: (note.start, note.pitch, note.end))

        # スライド／フォール列が小節境界で分断されないよう、チェーン内の全ノートはアンカー音の小節へ統一する。
        note_bar_indices = [
            _assign_note_to_bar(
                float(note.start),
                bar_windows,
                boundary_margin_seconds=float(config.bar_boundary_margin_seconds),
            )
            for note in original_sorted
        ]
        slide_chains = _detect_slide_and_grace_chains(
            original_sorted,
            is_drum=bool(original_instrument.is_drum),
            config=config,
        )
        for chain in slide_chains:
            anchor_index = max(
                chain,
                key=lambda note_index: (
                    float(original_sorted[note_index].end) - float(original_sorted[note_index].start),
                    -note_index,
                ),
            )
            anchor_bar = note_bar_indices[anchor_index]
            for note_index in chain:
                note_bar_indices[note_index] = anchor_bar

        rebuilt_notes: list[pretty_midi.Note] = []
        original_starts_by_id: dict[int, float] = {}
        for note_position, original_note in enumerate(original_sorted):
            bar_index = note_bar_indices[note_position]
            chosen_name = candidate_names[selected_candidate_by_bar[bar_index]]
            chosen_start, chosen_end = variant_timings_by_candidate[chosen_name][instrument_index][note_position]
            rebuilt = pretty_midi.Note(
                velocity=int(original_note.velocity),
                pitch=int(original_note.pitch),
                start=float(chosen_start),
                end=float(chosen_end),
            )
            original_starts_by_id[id(rebuilt)] = float(original_note.start)
            shift_ms = abs(float(chosen_start) - float(original_note.start)) * 1000.0
            if shift_ms >= 0.5:
                moved_note_count += 1
                total_abs_shift_ms += shift_ms
            rebuilt_notes.append(rebuilt)

        # 小節境界で候補が切り替わった箇所でも同ピッチの重複・極短押し潰しが起きないよう解消する。
        _resolve_same_pitch_collisions(rebuilt_notes, original_starts_by_id, config=config)
        rebuilt_notes.sort(key=lambda note: (note.start, note.pitch, note.end))
        original_instrument.notes = rebuilt_notes

    mean_shift_ms = total_abs_shift_ms / float(moved_note_count) if moved_note_count > 0 else 0.0
    return assembled, moved_note_count, mean_shift_ms


def quantize_stems_by_render_compare(
    stem_midis: Mapping[str, Path | str],
    stem_audios: Mapping[str, Path | str],
    *,
    beat_grid: BeatGridInput,
    config: RenderCompareQuantizeConfig,
    original_mix_audio: Path | str | None = None,
    output_midi_path: Path | str | None = None,
    output_stem_dir: Path | str | None = None,
    candidates: Sequence[_QuantizeCandidateSpec] = DEFAULT_QUANTIZE_CANDIDATES,
    packed_midi_renderer: _PackedMidiRenderer | None = None,
    preloaded_waveforms: Mapping[str, np.ndarray] | None = None,
    preloaded_full_mix: np.ndarray | None = None,
) -> dict[str, Any]:
    """全ステム MIDI をビート情報と Render & Compare によって小節単位で最適量子化する。"""
    sample_rate = int(config.sample_rate)
    resolved_midis = {str(name).lower(): Path(path) for name, path in stem_midis.items() if Path(path).is_file()}
    resolved_audios = {str(name).lower(): Path(path) for name, path in stem_audios.items() if Path(path).is_file()}
    preloaded_normalized = (
        {str(name).lower(): np.asarray(wave, dtype=np.float32) for name, wave in preloaded_waveforms.items()}
        if preloaded_waveforms is not None
        else {}
    )
    active_stems = sorted(set(resolved_midis) & set(resolved_audios))
    if not active_stems:
        raise ValueError("No matching stem MIDI and audio pairs were found.")
    if len(beat_grid.beat_times) < 2:
        raise ValueError("beat_grid must contain at least 2 beat times.")

    # 1. 元ステム音声と MIDI を読み込み、同一ステム内の重複トラックを統合しておく。
    original_midis: dict[str, pretty_midi.PrettyMIDI] = {}
    target_waveforms: dict[str, np.ndarray] = {}
    stem_durations: dict[str, float] = {}

    for stem_name in active_stems:
        midi_object = pretty_midi.PrettyMIDI(str(resolved_midis[stem_name]))
        consolidated: dict[tuple[int, bool, str], pretty_midi.Instrument] = {}
        for instrument in midi_object.instruments:
            if not instrument.notes:
                continue
            key = (int(instrument.program), bool(instrument.is_drum), str(instrument.name))
            merged_instrument = consolidated.get(key)
            if merged_instrument is None:
                merged_instrument = pretty_midi.Instrument(program=key[0], is_drum=key[1], name=key[2])
                consolidated[key] = merged_instrument
            merged_instrument.notes.extend(instrument.notes)
            merged_instrument.control_changes.extend(instrument.control_changes)
            merged_instrument.pitch_bends.extend(instrument.pitch_bends)
        for instrument in consolidated.values():
            instrument.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
            instrument.control_changes.sort(key=lambda control: control.time)
            instrument.pitch_bends.sort(key=lambda bend: bend.time)
        midi_object.instruments = list(consolidated.values())
        if not midi_object.instruments:
            continue

        mono_audio = (
            preloaded_normalized[stem_name]
            if stem_name in preloaded_normalized
            else _load_mono_audio(resolved_audios[stem_name], target_sample_rate=sample_rate)
        )
        original_midis[stem_name] = midi_object
        target_waveforms[stem_name] = mono_audio
        stem_durations[stem_name] = max(0.25, float(mono_audio.size) / float(sample_rate))

    stems_with_notes = sorted(original_midis)
    if not stems_with_notes:
        raise ValueError("No notes found in matching stem MIDI files.")

    max_duration_seconds = max(stem_durations.values())
    max_samples = max(int(wave.size) for wave in target_waveforms.values())
    if preloaded_full_mix is not None:
        target_full_mix = np.asarray(preloaded_full_mix, dtype=np.float32)
    elif original_mix_audio is not None and Path(original_mix_audio).is_file():
        target_full_mix = _load_mono_audio(Path(original_mix_audio), target_sample_rate=sample_rate)
    else:
        target_full_mix = np.zeros(max_samples, dtype=np.float32)
        for wave in target_waveforms.values():
            target_full_mix[: wave.size] += wave

    # 2. ビートグリッドを実音声のオンセット位相へサブフレームアラインメントする。
    #    ドラムステムがあればドラムのアタックを最優先基準にし、無ければフルミックスを使う。
    beat_phase_shift_seconds = 0.0
    beat_alignment_gain_ratio = 0.0
    aligned_beat_grid = beat_grid
    if config.align_beats_to_audio:
        alignment_reference = target_waveforms.get("drums", target_full_mix)
        aligned_beat_grid, beat_phase_shift_seconds, beat_alignment_gain_ratio = align_beat_grid_to_audio(
            beat_grid,
            alignment_reference,
            sample_rate=sample_rate,
            max_shift_seconds=float(config.max_beat_phase_shift_seconds),
            step_seconds=float(config.beat_phase_step_seconds),
        )

    beat_times_array = np.asarray(aligned_beat_grid.beat_times, dtype=np.float64)
    bar_windows = _build_bar_windows(aligned_beat_grid, max_duration_seconds)

    # 3. 各ステム × 各量子化候補の MIDI を生成する（adaptive_candidates 有効時は不要候補を省略）。
    if not candidates or candidates[0].name != "original":
        raise ValueError("First quantize candidate must be 'original'")

    candidates_by_stem: dict[str, list[_QuantizeCandidateSpec]] = {}
    variants_by_stem: dict[str, dict[str, pretty_midi.PrettyMIDI]] = {}
    variant_timings_by_stem: dict[str, dict[str, list[list[tuple[float, float]]]]] = {}
    for stem_name in stems_with_notes:
        base_midi = original_midis[stem_name]
        stem_candidates = (
            _select_adaptive_candidates_for_stem(base_midi, beat_times_array, candidates)
            if config.adaptive_candidates
            else list(candidates)
        )
        candidates_by_stem[stem_name] = stem_candidates
        variants_by_stem[stem_name] = {}
        variant_timings_by_stem[stem_name] = {}
        for candidate in stem_candidates:
            variant_midi = pretty_midi.PrettyMIDI()
            new_instruments: list[pretty_midi.Instrument] = []
            instrument_timings: list[list[tuple[float, float]]] = []
            for instrument in base_midi.instruments:
                quantized_instrument, source_order_timings = _build_quantized_instrument_variant(
                    instrument,
                    beat_times=beat_times_array,
                    candidate=candidate,
                    config=config,
                )
                new_instruments.append(quantized_instrument)
                instrument_timings.append(source_order_timings)
            variant_midi.instruments = new_instruments
            variants_by_stem[stem_name][candidate.name] = variant_midi
            variant_timings_by_stem[stem_name][candidate.name] = instrument_timings

    # 4. 全ステム × 候補をレンダリングする（本番 FluidSynth 実行時はステム単位でマルチコア並列実行）。
    mel_filterbank = _build_mel_filterbank(sample_rate=sample_rate)
    target_features_by_stem: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for stem_name in stems_with_notes:
        target_features_by_stem[stem_name] = _compute_timing_features(
            target_waveforms[stem_name],
            sample_rate=sample_rate,
            mel_filterbank=mel_filterbank,
        )

    slot_specs_by_stem: dict[str, list[tuple[str, str, str, Sequence[pretty_midi.Instrument], float]]] = {}
    for stem_name in stems_with_notes:
        duration_seconds = stem_durations[stem_name]
        slot_specs_by_stem[stem_name] = [
            (
                f"{stem_name}::{candidate.name}",
                stem_name,
                candidate.name,
                variants_by_stem[stem_name][candidate.name].instruments,
                duration_seconds,
            )
            for candidate in candidates_by_stem[stem_name]
        ]

    assembled_midis: dict[str, pretty_midi.PrettyMIDI] = {}
    stem_reports: dict[str, dict[str, Any]] = {}
    full_mix_report: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="tsumugi_rc_quantize_") as temp_directory:
        temp_root = Path(temp_directory)
        rendered_segments_by_key: dict[str, np.ndarray] = {}
        use_fine_grained_parallel = packed_midi_renderer is None and is_libfluidsynth_available(
            config.fluidsynth_executable
        )
        total_slot_count = sum(len(slot_specs_by_stem[stem_name]) for stem_name in stems_with_notes)
        parallel_job_count = total_slot_count if use_fine_grained_parallel else len(stems_with_notes)
        parallel_workers = (
            default_parallel_render_workers(
                Path(config.soundfont_path),
                parallel_job_count,
                max_workers=config.parallel_render_workers,
            )
            if packed_midi_renderer is None
            else 1
        )

        render_options = FluidSynthRenderOptions(
            soundfont_path=Path(config.soundfont_path),
            sample_rate=sample_rate,
            fluidsynth_executable=config.fluidsynth_executable,
            render_gain=float(config.render_gain),
            disable_reverb=bool(config.disable_reverb),
            disable_chorus=bool(config.disable_chorus),
            polyphony=64,
        )

        if parallel_workers <= 1 or parallel_job_count <= 1:
            all_slot_specs = [spec for stem_name in stems_with_notes for spec in slot_specs_by_stem[stem_name]]
            candidates_midi_path = temp_root / "candidates_packed.mid"
            slots = _write_packed_variants_midi(
                all_slot_specs,
                output_midi_path=candidates_midi_path,
                slot_gap_seconds=float(config.slot_gap_seconds),
            )
            if packed_midi_renderer is not None:
                candidates_wav_path = temp_root / "candidates_packed.wav"
                packed_midi_renderer(candidates_midi_path, candidates_wav_path, sample_rate)
                rendered_audio, rendered_rate = sf.read(str(candidates_wav_path), dtype="float32", always_2d=True)
                if int(rendered_rate) != sample_rate:
                    raise ValueError(f"Rendered sample rate mismatch: expected {sample_rate}, got {rendered_rate}")
                rendered_mono = rendered_audio.mean(axis=1).astype(np.float32, copy=False)
            else:
                rendered_stereo = render_midi_to_stereo_array(candidates_midi_path, options=render_options)
                rendered_mono = rendered_stereo.mean(axis=0).astype(np.float32, copy=False)
            for slot in slots:
                expected_samples = int(target_waveforms[slot.stem_name].size)
                start_sample = max(0, int(round(slot.slot_start_seconds * sample_rate)))
                segment = rendered_mono[start_sample : min(rendered_mono.size, start_sample + expected_samples)]
                if segment.size < expected_samples:
                    segment = np.pad(segment, (0, expected_samples - segment.size))
                rendered_segments_by_key[slot.slot_key] = segment
        elif use_fine_grained_parallel:
            # 常駐 libfluidsynth では SF2 再読込コストがゼロのため、(ステム × 候補) 単位で細粒度並列化する。
            # ノート数が多い重い候補（例: other の 5 候補）から先に投入する LPT スケジューリングにより、
            # 最大 4 ワーカーの範囲内でコア間の待ち偏りを抑えて高速化する。
            ordered_slot_specs = sorted(
                (spec for stem_name in stems_with_notes for spec in slot_specs_by_stem[stem_name]),
                key=lambda spec: (-sum(len(instrument.notes) for instrument in spec[3]), spec[0]),
            )
            slot_midi_paths: list[Path] = []
            slot_metadata: list[tuple[str, str]] = []
            for slot_index, slot_spec in enumerate(ordered_slot_specs):
                slot_midi_path = temp_root / f"quantize_slot_{slot_index}.mid"
                _write_packed_variants_midi(
                    [slot_spec],
                    output_midi_path=slot_midi_path,
                    slot_gap_seconds=0.0,
                )
                slot_midi_paths.append(slot_midi_path)
                slot_metadata.append((slot_spec[0], slot_spec[1]))

            rendered_stereo_list = render_midi_batch_to_stereo_arrays(
                slot_midi_paths,
                options=render_options,
                max_workers=parallel_workers,
            )
            for (slot_key, stem_name), rendered_stereo in zip(slot_metadata, rendered_stereo_list):
                rendered_mono = rendered_stereo.mean(axis=0).astype(np.float32, copy=False)
                expected_samples = int(target_waveforms[stem_name].size)
                segment = rendered_mono[:expected_samples]
                if segment.size < expected_samples:
                    segment = np.pad(segment, (0, expected_samples - segment.size))
                rendered_segments_by_key[slot_key] = segment
        else:
            stem_midi_paths: list[Path] = []
            stem_job_metadata: list[tuple[str, list[_PackedVariantSlot]]] = []
            for stem_index, stem_name in enumerate(stems_with_notes):
                stem_midi_path = temp_root / f"quantize_stem_{stem_index}.mid"
                stem_slots = _write_packed_variants_midi(
                    slot_specs_by_stem[stem_name],
                    output_midi_path=stem_midi_path,
                    slot_gap_seconds=float(config.slot_gap_seconds),
                )
                stem_midi_paths.append(stem_midi_path)
                stem_job_metadata.append((stem_name, stem_slots))

            rendered_stereo_list = render_midi_batch_to_stereo_arrays(
                stem_midi_paths,
                options=render_options,
                max_workers=parallel_workers,
            )
            for (stem_name, stem_slots), rendered_stereo in zip(stem_job_metadata, rendered_stereo_list):
                rendered_mono = rendered_stereo.mean(axis=0).astype(np.float32, copy=False)
                expected_samples = int(target_waveforms[stem_name].size)
                for slot in stem_slots:
                    start_sample = max(0, int(round(slot.slot_start_seconds * sample_rate)))
                    segment = rendered_mono[start_sample : min(rendered_mono.size, start_sample + expected_samples)]
                    if segment.size < expected_samples:
                        segment = np.pad(segment, (0, expected_samples - segment.size))
                    rendered_segments_by_key[slot.slot_key] = segment

        # 5. 各ステム・各小節ごとに音響タイミング距離を計測し、Viterbi で最適候補列を選択する。
        for stem_name in stems_with_notes:
            stem_candidates = candidates_by_stem[stem_name]
            stem_candidate_names = [candidate.name for candidate in stem_candidates]
            target_onset, target_mel, frames_per_second = target_features_by_stem[stem_name]
            num_target_frames = int(target_onset.shape[1])

            rendered_features_by_candidate: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for candidate_name in stem_candidate_names:
                segment = rendered_segments_by_key[f"{stem_name}::{candidate_name}"]
                candidate_onset, candidate_mel, _ = _compute_timing_features(
                    segment,
                    sample_rate=sample_rate,
                    mel_filterbank=mel_filterbank,
                )
                rendered_features_by_candidate[candidate_name] = (candidate_onset, candidate_mel)

            raw_distances = np.zeros((len(bar_windows), len(stem_candidates)), dtype=np.float64)
            bar_note_counts = np.zeros(len(bar_windows), dtype=np.int32)
            for instrument in original_midis[stem_name].instruments:
                for note in instrument.notes:
                    bar_index = _assign_note_to_bar(
                        float(note.start),
                        bar_windows,
                        boundary_margin_seconds=float(config.bar_boundary_margin_seconds),
                    )
                    bar_note_counts[bar_index] += 1

            for bar_index, (bar_start, bar_end) in enumerate(bar_windows):
                if bar_note_counts[bar_index] == 0:
                    continue
                window_start = max(0.0, bar_start - float(config.bar_boundary_margin_seconds))
                window_end = min(stem_durations[stem_name], bar_end + float(config.bar_boundary_margin_seconds))
                start_frame = max(0, min(num_target_frames - 2, int(math.floor(window_start * frames_per_second))))
                end_frame = min(num_target_frames, max(start_frame + 2, int(math.ceil(window_end * frames_per_second))))
                for candidate_index, candidate_name in enumerate(stem_candidate_names):
                    candidate_onset, candidate_mel = rendered_features_by_candidate[candidate_name]
                    common_end = min(end_frame, int(candidate_onset.shape[1]))
                    if common_end <= start_frame + 1:
                        raw_distances[bar_index, candidate_index] = 1.0
                    else:
                        raw_distances[bar_index, candidate_index] = _compute_window_timing_distance(
                            target_onset,
                            target_mel,
                            candidate_onset,
                            candidate_mel,
                            start_frame=start_frame,
                            end_frame=common_end,
                            config=config,
                        )

            is_vocal_stem = "vocal" in stem_name
            prior_costs = [
                float(candidate.vocal_prior_cost if is_vocal_stem else candidate.prior_cost)
                for candidate in stem_candidates
            ]
            selected_by_bar = _decode_viterbi_candidate_sequence(
                raw_distances,
                prior_costs=prior_costs,
                transition_penalty=float(config.viterbi_transition_penalty),
                max_regression_tolerance=float(config.max_regression_tolerance),
            )
            # ノートが 1 つもない空小節は original 扱いにして統計を明確にする。
            for bar_index in range(len(bar_windows)):
                if bar_note_counts[bar_index] == 0:
                    selected_by_bar[bar_index] = 0

            assembled_midi, moved_notes, mean_shift_ms = _assemble_stem_midi_from_bar_choices(
                original_midis[stem_name],
                variant_timings_by_stem[stem_name],
                stem_candidate_names,
                selected_by_bar,
                bar_windows,
                config=config,
            )
            assembled_midis[stem_name] = assembled_midi

            active_bars = [index for index in range(len(bar_windows)) if bar_note_counts[index] > 0]
            before_mean_distance = (
                float(np.mean([raw_distances[index, 0] for index in active_bars])) if active_bars else 0.0
            )
            after_mean_distance = (
                float(np.mean([raw_distances[index, selected_by_bar[index]] for index in active_bars]))
                if active_bars
                else 0.0
            )
            choice_counts = {
                candidate_name: int(sum(1 for index in active_bars if selected_by_bar[index] == candidate_index))
                for candidate_index, candidate_name in enumerate(stem_candidate_names)
            }
            total_notes = sum(len(instrument.notes) for instrument in original_midis[stem_name].instruments)
            stem_reports[stem_name] = {
                "note_count": int(total_notes),
                "moved_note_count": int(moved_notes),
                "mean_abs_shift_ms": float(mean_shift_ms),
                "active_bar_count": int(len(active_bars)),
                "before_timing_distance": before_mean_distance,
                "after_timing_distance": after_mean_distance,
                "distance_delta": float(after_mean_distance - before_mean_distance),
                "candidate_bar_counts": choice_counts,
            }

        # 6. 最終フルミックス検証（量子化前 vs 量子化後を全ステム同時再生して元フルミックスと比較）。
        if config.verify_full_mix:
            before_all_instruments = [
                instrument
                for stem_name in stems_with_notes
                for instrument in original_midis[stem_name].instruments
                if instrument.notes
            ]
            after_all_instruments = [
                instrument
                for stem_name in stems_with_notes
                for instrument in assembled_midis[stem_name].instruments
                if instrument.notes
            ]
            verify_midi_path = temp_root / "verify_full_mix.mid"
            verify_slots = _write_packed_variants_midi(
                [
                    ("before_full_mix", "__full_mix__", "original", before_all_instruments, max_duration_seconds),
                    ("after_full_mix", "__full_mix__", "quantized", after_all_instruments, max_duration_seconds),
                ],
                output_midi_path=verify_midi_path,
                slot_gap_seconds=float(config.slot_gap_seconds),
            )
            if packed_midi_renderer is not None:
                verify_wav_path = temp_root / "verify_full_mix.wav"
                packed_midi_renderer(verify_midi_path, verify_wav_path, sample_rate)
                verify_audio, _ = sf.read(str(verify_wav_path), dtype="float32", always_2d=True)
                verify_mono = verify_audio.mean(axis=1).astype(np.float32, copy=False)
            else:
                verify_stereo = render_midi_to_stereo_array(verify_midi_path, options=render_options)
                verify_mono = verify_stereo.mean(axis=0).astype(np.float32, copy=False)
            target_mix_onset, target_mix_mel, frames_per_second = _compute_timing_features(
                target_full_mix,
                sample_rate=sample_rate,
                mel_filterbank=mel_filterbank,
            )
            mix_distances: dict[str, float] = {}
            mix_onset_corrs: dict[str, float] = {}
            for slot in verify_slots:
                start_sample = max(0, int(round(slot.slot_start_seconds * sample_rate)))
                segment = verify_mono[start_sample : min(verify_mono.size, start_sample + target_full_mix.size)]
                if segment.size < target_full_mix.size:
                    segment = np.pad(segment, (0, target_full_mix.size - segment.size))
                rendered_mix_onset, rendered_mix_mel, _ = _compute_timing_features(
                    segment,
                    sample_rate=sample_rate,
                    mel_filterbank=mel_filterbank,
                )
                common_frames = min(int(target_mix_onset.shape[1]), int(rendered_mix_onset.shape[1]))
                per_bar_distances: list[float] = []
                for bar_start, bar_end in bar_windows:
                    start_frame = max(0, min(common_frames - 2, int(math.floor(bar_start * frames_per_second))))
                    end_frame = min(common_frames, max(start_frame + 2, int(math.ceil(bar_end * frames_per_second))))
                    per_bar_distances.append(
                        _compute_window_timing_distance(
                            target_mix_onset,
                            target_mix_mel,
                            rendered_mix_onset,
                            rendered_mix_mel,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            config=config,
                        )
                    )
                mix_distances[slot.slot_key] = float(np.mean(per_bar_distances))
                target_flat = target_mix_onset[:, :common_frames].reshape(-1).astype(np.float64)
                rendered_flat = rendered_mix_onset[:, :common_frames].reshape(-1).astype(np.float64)
                denom = float(np.linalg.norm(target_flat) * np.linalg.norm(rendered_flat))
                mix_onset_corrs[slot.slot_key] = float(np.dot(target_flat, rendered_flat) / denom) if denom > 1e-8 else 0.0

            full_mix_report = {
                "before_timing_distance": mix_distances["before_full_mix"],
                "after_timing_distance": mix_distances["after_full_mix"],
                "distance_delta": float(mix_distances["after_full_mix"] - mix_distances["before_full_mix"]),
                "before_onset_cosine": mix_onset_corrs["before_full_mix"],
                "after_onset_cosine": mix_onset_corrs["after_full_mix"],
            }

    # 7. ステム別 MIDI・マージ済み MIDI（任意でアラインメント済みテンポマップ付き）を書き出す。
    written_stem_paths: dict[str, str] = {}
    if output_stem_dir is not None:
        stem_output_directory = Path(output_stem_dir)
        stem_output_directory.mkdir(parents=True, exist_ok=True)
        for stem_name, midi_object in assembled_midis.items():
            destination = stem_output_directory / f"{resolved_midis[stem_name].stem}_quantized.mid"
            midi_object.write(str(destination))
            written_stem_paths[stem_name] = str(destination.resolve())

    merged_output_resolved: str | None = None
    tempo_map_written = False
    if output_midi_path is not None:
        merged_midi = _merge_optimized_stem_midis(
            assembled_midis,
            max_melodic_instruments=int(config.max_melodic_instruments),
        )
        # 16ch 制約で複数トラックが Other / Merged 等へ統合された場合も同ピッチ重複による発音消失を防ぐ。
        for instrument in merged_midi.instruments:
            if instrument.notes:
                _resolve_same_pitch_collisions(instrument.notes, None, config=config)
                instrument.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
        destination_path = Path(output_midi_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if config.write_tempo_map and aligned_beat_grid.meter_segments:
            with tempfile.TemporaryDirectory(prefix="tsumugi_tempo_export_") as tempo_temp_dir:
                raw_merged_path = Path(tempo_temp_dir) / "raw_merged.mid"
                merged_midi.write(str(raw_merged_path))
                export_tempo_mapped_midi(
                    source_midi_path=raw_merged_path,
                    output_midi_path=destination_path,
                    beat_times=aligned_beat_grid.beat_times,
                    meter_segments=aligned_beat_grid.meter_segments,
                    chord_segments=(),
                    duration_seconds=max_duration_seconds,
                )
            tempo_map_written = True
        else:
            merged_midi.write(str(destination_path))
        merged_output_resolved = str(destination_path.resolve())

    return {
        "stems": stems_with_notes,
        "bar_count": int(len(bar_windows)),
        "beat_phase_shift_ms": float(beat_phase_shift_seconds * 1000.0),
        "beat_alignment_gain_percent": float(beat_alignment_gain_ratio * 100.0),
        "tempo_map_written": tempo_map_written,
        "stems_report": stem_reports,
        "full_mix": full_mix_report,
        "output_midi_path": merged_output_resolved,
        "output_stem_paths": written_stem_paths,
    }
