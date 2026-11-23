"""Render & Compare による GM 128 プログラム直接選定。

Instrument Refinement モデルが出力した音色クラスタごとに、候補となる複数の GM
プログラムを実際に対象 SoundFont でレンダリングし、元ステム音声との聴感・音色距離
（ニューラル音色埋め込み + フレーム正規化スペクトル形状 + エンベロープ形状）が
最も小さい GM プログラムを選定する。

大容量の SoundFont 読み込みコストを抑えるため、全クラスタ × 全スニペット × 全候補を
1 本の結合 MIDI に時間軸パックし、FluidSynth を 1 ステムあたり 1 回だけ起動する。
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from ...runtime import copy_tensors_to_cpu_once
from ...soundfont import (
    FluidSynthRenderOptions,
    RENDER_COMPARE_SPEED_PRESETS,
    default_parallel_render_workers,
    render_midi_batch_to_stereo_arrays,
    render_midi_to_stereo_array,
    render_midi_with_fluidsynth,
    resolve_fluidsynth_executable,
    resolve_signal_processing_device,
    write_packed_smf_type0,
)
from ...taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_gm_program_frequency,
    get_gm_program_name,
    get_instrument_class_id,
    get_program_numbers_for_class_id,
)
from ..data.labels import inference_stem_group, stem_context_id
from ..data.midi import RefinementNoteTable
from ..modeling.model import InstrumentRefinementConfig, InstrumentRefinementModel

# ドラム専用チャンネル (9) を避けて 15 チャンネルをローテーションすることで、
# 直前スロットの残響が次の候補スロットへ混ざるのを防ぐ。
_MELODIC_MIDI_CHANNELS = tuple(channel for channel in range(16) if channel != 9)
_PACKED_TICKS_PER_BEAT = 480
_PACKED_TEMPO_MICROSECONDS = 500_000  # 120 BPM -> 1 秒 = 960 tick (約 1.04 ms 精度)
_PACKED_TICKS_PER_SECOND = float(_PACKED_TICKS_PER_BEAT) * 1_000_000.0 / float(_PACKED_TEMPO_MICROSECONDS)

_PackedMidiRenderer = Callable[[Path, Path, int], None]


@dataclass(frozen=True)
class RenderCompareConfig:
    """Render & Compare による GM プログラム選定の設定。"""

    soundfont_path: Path
    fluidsynth_executable: Path | None = None
    candidate_mode: str = "top_classes"
    top_candidate_classes: int = 5
    max_candidate_programs: int = 24
    snippet_seconds: float = 4.0
    max_snippets_per_cluster: int = 2
    slot_gap_seconds: float = 0.8
    render_gain: float = 0.5
    render_velocity: int = 96
    disable_reverb: bool = False
    disable_chorus: bool = False
    embedding_weight: float = 0.40
    spectral_weight: float = 0.35
    envelope_weight: float = 0.25
    classifier_prior_weight: float = 0.08
    program_prior_weight: float = 0.03
    min_rendered_rms_dbfs: float = -55.0
    max_relative_rms_drop_db: float = 22.0
    small_cluster_note_threshold: int = 40
    embedding_batch_size: int = 8
    top_report_programs: int = 5
    parallel_render_workers: int | None = None

    @classmethod
    def for_speed_preset(
        cls,
        soundfont_path: Path | str,
        *,
        speed: str = "balanced",
        fluidsynth_executable: Path | str | None = None,
        parallel_render_workers: int | None = None,
    ) -> "RenderCompareConfig":
        """速度と精度のバランスプリセット (`fast` / `balanced` / `quality`) から設定を構築する。"""
        normalized_speed = str(speed).strip().lower()
        if normalized_speed not in RENDER_COMPARE_SPEED_PRESETS:
            raise ValueError(f"speed must be one of {RENDER_COMPARE_SPEED_PRESETS}, got {speed!r}")
        resolved_executable = Path(fluidsynth_executable) if fluidsynth_executable is not None else None
        if normalized_speed == "fast":
            return cls(
                soundfont_path=Path(soundfont_path),
                fluidsynth_executable=resolved_executable,
                top_candidate_classes=3,
                max_candidate_programs=10,
                snippet_seconds=2.5,
                max_snippets_per_cluster=1,
                slot_gap_seconds=0.5,
                disable_reverb=True,
                disable_chorus=True,
                embedding_batch_size=16,
                parallel_render_workers=parallel_render_workers,
            )
        if normalized_speed == "balanced":
            return cls(
                soundfont_path=Path(soundfont_path),
                fluidsynth_executable=resolved_executable,
                top_candidate_classes=4,
                max_candidate_programs=16,
                snippet_seconds=3.0,
                max_snippets_per_cluster=2,
                slot_gap_seconds=0.6,
                disable_reverb=True,
                disable_chorus=True,
                embedding_batch_size=16,
                parallel_render_workers=parallel_render_workers,
            )
        return cls(
            soundfont_path=Path(soundfont_path),
            fluidsynth_executable=resolved_executable,
            embedding_batch_size=16,
            parallel_render_workers=parallel_render_workers,
        )

    def __post_init__(self) -> None:
        if self.candidate_mode not in {"top_classes", "stem_all"}:
            raise ValueError("candidate_mode must be 'top_classes' or 'stem_all'")
        if self.top_candidate_classes <= 0 or self.max_candidate_programs <= 0:
            raise ValueError("top_candidate_classes and max_candidate_programs must be positive")
        if self.snippet_seconds <= 0.0 or self.max_snippets_per_cluster <= 0:
            raise ValueError("snippet_seconds and max_snippets_per_cluster must be positive")
        if self.slot_gap_seconds < 0.0:
            raise ValueError("slot_gap_seconds must be nonnegative")
        if self.render_gain <= 0.0:
            raise ValueError("render_gain must be positive")
        if not 1 <= self.render_velocity <= 127:
            raise ValueError("render_velocity must be in 1..127")
        if (
            self.embedding_weight < 0.0
            or self.spectral_weight < 0.0
            or self.envelope_weight < 0.0
            or self.classifier_prior_weight < 0.0
            or self.program_prior_weight < 0.0
        ):
            raise ValueError("distance weights must be nonnegative")
        if self.max_relative_rms_drop_db < 0.0 or self.small_cluster_note_threshold < 0:
            raise ValueError("max_relative_rms_drop_db and small_cluster_note_threshold must be nonnegative")
        if self.embedding_batch_size <= 0 or self.top_report_programs <= 0:
            raise ValueError("embedding_batch_size and top_report_programs must be positive")


@dataclass(frozen=True)
class _CandidateProgram:
    program_number: int
    class_id: int
    prior_probability: float


@dataclass(frozen=True)
class _ClusterSnippet:
    cluster_id: int
    snippet_index: int
    start_seconds: float
    duration_seconds: float
    note_indices: np.ndarray
    relative_start_seconds: np.ndarray
    relative_end_seconds: np.ndarray
    pitches: np.ndarray
    target_audio: np.ndarray  # [2, sample_count] float32
    purity: float


@dataclass(frozen=True)
class _RenderSlot:
    cluster_id: int
    snippet_index: int
    program_number: int
    slot_start_seconds: float
    slot_duration_seconds: float
    channel: int


def _render_packed_midi_with_fluidsynth(
    midi_path: Path,
    wav_path: Path,
    sample_rate: int,
    *,
    config: RenderCompareConfig,
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
            polyphony=64,
        ),
    )


def _collect_candidate_programs(
    cluster_logits: np.ndarray,
    *,
    allowed_class_ids: tuple[int, ...],
    config: RenderCompareConfig,
    note_count: int | None = None,
) -> list[_CandidateProgram]:
    """クラスタロジット上位クラスから候補 GM プログラム一覧を展開する。"""
    allowed_array = np.asarray(allowed_class_ids, dtype=np.int64)
    selected_logits = np.asarray(cluster_logits, dtype=np.float64)[allowed_array]
    selected_logits = selected_logits - float(selected_logits.max())
    probabilities = np.exp(selected_logits)
    probabilities /= float(probabilities.sum().clip(min=1e-12))

    # 小クラスタでは候補クラス数と最大プログラム数を少し絞って全体のレンダリング時間を抑える。
    is_small_cluster = note_count is not None and int(note_count) < int(config.small_cluster_note_threshold)
    class_limit = (
        min(int(config.top_candidate_classes), 3)
        if is_small_cluster
        else int(config.top_candidate_classes)
    )
    program_limit = (
        min(int(config.max_candidate_programs), 14)
        if is_small_cluster
        else int(config.max_candidate_programs)
    )

    ranked_order = np.argsort(-probabilities)
    if config.candidate_mode == "top_classes":
        ranked_order = ranked_order[:class_limit]

    candidates: list[_CandidateProgram] = []
    seen_programs: set[int] = set()
    for order_index in ranked_order.tolist():
        class_id = int(allowed_array[order_index])
        class_probability = float(probabilities[order_index])
        for program_number in get_program_numbers_for_class_id(class_id):
            if program_number in seen_programs:
                continue
            seen_programs.add(program_number)
            candidates.append(
                _CandidateProgram(
                    program_number=int(program_number),
                    class_id=class_id,
                    prior_probability=class_probability,
                )
            )
            if len(candidates) >= program_limit:
                return candidates
    return candidates


def _select_cluster_snippets(
    notes: RefinementNoteTable,
    waveform_numpy: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    cluster_id: int,
    sample_rate: int,
    config: RenderCompareConfig,
) -> list[_ClusterSnippet]:
    """他クラスタとの重なりが少なく、音量が十分な代表区間を最大 N 箇所選ぶ。"""
    total_samples = int(waveform_numpy.shape[1])
    total_seconds = float(total_samples) / float(sample_rate)
    member_indices = np.flatnonzero(cluster_ids == cluster_id)
    if member_indices.size == 0 or total_samples == 0:
        return []

    other_indices = np.flatnonzero(cluster_ids != cluster_id)
    member_starts = notes.start_seconds[member_indices].astype(np.float64)
    member_ends = np.maximum(notes.end_seconds[member_indices].astype(np.float64), member_starts + 0.03)
    other_starts = notes.start_seconds[other_indices].astype(np.float64)
    other_ends = np.maximum(notes.end_seconds[other_indices].astype(np.float64), other_starts + 0.03)

    snippet_seconds = min(float(config.snippet_seconds), max(0.25, total_seconds))
    cluster_first = float(member_starts.min())
    cluster_last = float(member_ends.max())
    cluster_span = cluster_last - cluster_first

    # クラスタ全体が 1 スニペットに収まる場合はそのまま 1 区間を候補にする。
    if cluster_span <= snippet_seconds:
        candidate_starts = [max(0.0, min(cluster_first - 0.05, max(0.0, total_seconds - snippet_seconds)))]
        effective_duration = min(snippet_seconds, max(0.25, cluster_span + 0.25), total_seconds)
    else:
        # ノート開始時刻と一定間隔グリッドから候補窓を列挙する（上限を設けて高速化）。
        step_seconds = max(0.5, snippet_seconds * 0.5)
        grid_starts = np.arange(
            max(0.0, cluster_first - 0.05),
            max(0.0, min(cluster_last, total_seconds) - snippet_seconds * 0.25) + 1e-6,
            step_seconds,
            dtype=np.float64,
        )
        note_aligned_starts = np.maximum(0.0, member_starts - 0.05)
        if note_aligned_starts.size > 48:
            stride = int(math.ceil(note_aligned_starts.size / 48.0))
            note_aligned_starts = note_aligned_starts[::stride]
        raw_starts = np.unique(np.round(np.concatenate((grid_starts, note_aligned_starts)), 2))
        max_start = max(0.0, total_seconds - snippet_seconds)
        candidate_starts = sorted({float(min(max(0.0, value), max_start)) for value in raw_starts.tolist()})
        effective_duration = snippet_seconds

    scored_candidates: list[tuple[float, float, float, np.ndarray]] = []
    for window_start in candidate_starts:
        window_end = min(total_seconds, window_start + effective_duration)
        actual_duration = window_end - window_start
        if actual_duration <= 0.1:
            continue

        # 窓内で開始するクラスタノートを評価対象にする（途中開始ノートはオンセット比較が崩れるため除外）。
        in_window_mask = (member_starts >= window_start - 1e-4) & (member_starts < window_end - 0.04)
        local_positions = np.flatnonzero(in_window_mask)
        if local_positions.size == 0:
            continue

        selected_starts = np.clip(member_starts[local_positions], window_start, window_end)
        selected_ends = np.clip(member_ends[local_positions], window_start, window_end)
        cluster_active_duration = float(np.maximum(0.0, selected_ends - selected_starts).sum())
        if cluster_active_duration <= 0.02:
            continue

        # 他クラスタのノートと時間的に重なっている量から純度を計算する。
        if other_indices.size > 0:
            other_overlap = np.maximum(
                0.0,
                np.minimum(other_ends, window_end) - np.maximum(other_starts, window_start),
            )
            other_active_duration = float(other_overlap.sum())
        else:
            other_active_duration = 0.0
        purity = cluster_active_duration / (cluster_active_duration + other_active_duration + 1e-6)

        start_sample = max(0, int(round(window_start * sample_rate)))
        end_sample = min(total_samples, start_sample + max(1, int(round(actual_duration * sample_rate))))
        segment = waveform_numpy[:, start_sample:end_sample]
        rms_value = float(np.sqrt(np.mean(np.square(segment), dtype=np.float64) + 1e-12))
        rms_dbfs = 20.0 * math.log10(max(rms_value, 1e-8))
        energy_factor = 1.0 if rms_dbfs > -55.0 else 0.05

        selection_score = (
            (purity**2)
            * math.log1p(float(local_positions.size))
            * min(cluster_active_duration, actual_duration)
            * energy_factor
        )
        scored_candidates.append((selection_score, window_start, purity, member_indices[local_positions]))

    if not scored_candidates:
        return []

    # 高スコア順に、時間的に重複しないスニペットを貪欲に選ぶ。
    # ノート数が少ないクラスタは 1 スニペットに抑えてレンダリング総量を削減する。
    snippet_limit = (
        1
        if int(member_indices.size) < int(config.small_cluster_note_threshold)
        else int(config.max_snippets_per_cluster)
    )
    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen_windows: list[tuple[float, float, np.ndarray]] = []
    for _score, window_start, purity, snippet_note_indices in scored_candidates:
        window_end = min(total_seconds, window_start + effective_duration)
        if any(
            not (window_end <= existing_start + 0.25 or window_start >= existing_start + effective_duration - 0.25)
            for existing_start, _existing_purity, _existing_notes in chosen_windows
        ):
            continue
        chosen_windows.append((window_start, purity, snippet_note_indices))
        if len(chosen_windows) >= snippet_limit:
            break

    snippets: list[_ClusterSnippet] = []
    for snippet_index, (window_start, purity, snippet_note_indices) in enumerate(chosen_windows):
        window_end = min(total_seconds, window_start + effective_duration)
        actual_duration = window_end - window_start
        start_sample = max(0, int(round(window_start * sample_rate)))
        target_sample_count = max(1, int(round(actual_duration * sample_rate)))
        target_slice = waveform_numpy[:, start_sample : start_sample + target_sample_count]
        if target_slice.shape[1] < target_sample_count:
            target_slice = np.pad(target_slice, ((0, 0), (0, target_sample_count - target_slice.shape[1])))

        relative_starts = np.clip(
            notes.start_seconds[snippet_note_indices].astype(np.float64) - window_start,
            0.0,
            max(0.0, actual_duration - 0.02),
        )
        relative_ends = np.clip(
            notes.end_seconds[snippet_note_indices].astype(np.float64) - window_start,
            relative_starts + 0.02,
            actual_duration,
        )
        snippets.append(
            _ClusterSnippet(
                cluster_id=int(cluster_id),
                snippet_index=int(snippet_index),
                start_seconds=float(window_start),
                duration_seconds=float(actual_duration),
                note_indices=snippet_note_indices.astype(np.int64),
                relative_start_seconds=relative_starts.astype(np.float32),
                relative_end_seconds=relative_ends.astype(np.float32),
                pitches=notes.pitch[snippet_note_indices].astype(np.int64),
                target_audio=target_slice.astype(np.float32, copy=True),
                purity=float(purity),
            )
        )
    return snippets


def _build_packed_candidate_midi(
    snippets_by_cluster: dict[int, list[_ClusterSnippet]],
    candidates_by_cluster: dict[int, list[_CandidateProgram]],
    *,
    output_midi_path: Path,
    config: RenderCompareConfig,
    minimum_candidates_per_cluster: int = 2,
) -> list[_RenderSlot]:
    """全クラスタ × 全スニペット × 全候補を 1 本の SMF Type-0 MIDI へ高速パックして書き出す。"""
    scheduled_events: list[tuple[int, int, bytes]] = []
    slots: list[_RenderSlot] = []
    cursor_seconds = 0.0
    slot_counter = 0
    velocity = max(1, min(127, int(config.render_velocity)))

    for cluster_id in sorted(snippets_by_cluster):
        snippets = snippets_by_cluster[cluster_id]
        candidates = candidates_by_cluster.get(cluster_id, [])
        if not snippets or len(candidates) < int(minimum_candidates_per_cluster):
            continue
        for snippet in snippets:
            for candidate in candidates:
                channel = _MELODIC_MIDI_CHANNELS[slot_counter % len(_MELODIC_MIDI_CHANNELS)]
                slot_counter += 1
                slot_start = cursor_seconds
                slot_end = slot_start + snippet.duration_seconds
                start_tick = int(round(slot_start * _PACKED_TICKS_PER_SECOND))
                end_tick = int(round(slot_end * _PACKED_TICKS_PER_SECOND))

                # スロット先頭で音色・音量・ペダルを初期化する。
                cc_status = 0xB0 | channel
                scheduled_events.append((start_tick, 1, bytes((0xC0 | channel, max(0, min(127, int(candidate.program_number)))))))
                scheduled_events.append((start_tick, 2, bytes((cc_status, 7, 127))))
                scheduled_events.append((start_tick, 2, bytes((cc_status, 11, 127))))
                scheduled_events.append((start_tick, 2, bytes((cc_status, 64, 0))))

                note_on_status = 0x90 | channel
                note_off_status = 0x80 | channel
                for note_start, note_end, pitch in zip(
                    snippet.relative_start_seconds.tolist(),
                    snippet.relative_end_seconds.tolist(),
                    snippet.pitches.tolist(),
                ):
                    note_on_tick = int(round((slot_start + float(note_start)) * _PACKED_TICKS_PER_SECOND))
                    note_off_tick = max(
                        note_on_tick + 1,
                        min(end_tick, int(round((slot_start + float(note_end)) * _PACKED_TICKS_PER_SECOND))),
                    )
                    clamped_pitch = max(0, min(127, int(pitch)))
                    scheduled_events.append((note_on_tick, 4, bytes((note_on_status, clamped_pitch, velocity))))
                    scheduled_events.append((note_off_tick, 3, bytes((note_off_status, clamped_pitch, 0))))

                # スロット終了直後に全発音を停止し、次スロットへの残響混入を抑える。
                cleanup_tick = end_tick + int(round(min(0.05, config.slot_gap_seconds * 0.25) * _PACKED_TICKS_PER_SECOND))
                scheduled_events.append((cleanup_tick, 5, bytes((cc_status, 123, 0))))
                scheduled_events.append((cleanup_tick, 6, bytes((cc_status, 120, 0))))

                slots.append(
                    _RenderSlot(
                        cluster_id=int(cluster_id),
                        snippet_index=int(snippet.snippet_index),
                        program_number=int(candidate.program_number),
                        slot_start_seconds=float(slot_start),
                        slot_duration_seconds=float(snippet.duration_seconds),
                        channel=int(channel),
                    )
                )
                cursor_seconds = slot_end + float(config.slot_gap_seconds)

    write_packed_smf_type0(
        output_midi_path,
        scheduled_events,
        ticks_per_beat=_PACKED_TICKS_PER_BEAT,
        tempo_microseconds=_PACKED_TEMPO_MICROSECONDS,
    )
    return slots


def _build_mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    num_mels: int,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> np.ndarray:
    """外部ライブラリ非依存の三角メルフィルタバンク [num_mels, n_fft // 2 + 1] を作る。"""
    num_bins = n_fft // 2 + 1
    frequencies = np.linspace(0.0, float(sample_rate) * 0.5, num_bins, dtype=np.float64)

    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(min_frequency_hz), hz_to_mel(max_frequency_hz), num_mels + 2)
    hz_points = mel_to_hz(mel_points)
    filterbank = np.zeros((num_mels, num_bins), dtype=np.float32)

    for mel_index in range(num_mels):
        left = hz_points[mel_index]
        center = hz_points[mel_index + 1]
        right = hz_points[mel_index + 2]
        rising = (frequencies - left) / max(center - left, 1e-6)
        falling = (right - frequencies) / max(right - center, 1e-6)
        triangle = np.maximum(0.0, np.minimum(rising, falling))
        area = float(triangle.sum())
        if area > 1e-8:
            filterbank[mel_index] = (triangle / area).astype(np.float32)
    return filterbank


def _compute_log_mel_and_frame_db(
    stereo_audio: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    mel_filterbank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """ステレオ波形から対数メルスペクトログラム [num_mels, num_frames] とフレーム音量 dB を求める。"""
    mono = stereo_audio.mean(axis=0).astype(np.float32, copy=False)
    if mono.size < n_fft:
        mono = np.pad(mono, (0, n_fft - mono.size))
    waveform_tensor = torch.from_numpy(mono)
    window = torch.hann_window(n_fft, dtype=torch.float32)
    spectrum = torch.stft(
        waveform_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power = (spectrum.real.square() + spectrum.imag.square()).numpy()
    mel_power = mel_filterbank @ power
    log_mel_db = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
    frame_power = power.mean(axis=0)
    frame_db = 10.0 * np.log10(np.maximum(frame_power, 1e-10))
    return log_mel_db.astype(np.float32), frame_db.astype(np.float32)


def _compute_spectral_and_envelope_distances(
    target_log_mel: np.ndarray,
    target_frame_db: np.ndarray,
    rendered_log_mel: np.ndarray,
    rendered_frame_db: np.ndarray,
    snippet: _ClusterSnippet,
    *,
    sample_rate: int,
    hop_length: int,
) -> tuple[float, float]:
    """フレーム正規化スペクトル形状距離と、ノート単位エンベロープ形状距離を計算する。

    ベロシティが未最適化な状態でも正しく音色だけを比較できるよう、フレームごとの平均
    エネルギーを差し引いたスペクトル形状と、ノートごとのピーク正規化減衰カーブで測る。
    """
    num_frames = min(int(target_log_mel.shape[1]), int(rendered_log_mel.shape[1]))
    if num_frames <= 0:
        return 1.0, 1.0

    target_mel = target_log_mel[:, :num_frames]
    rendered_mel = rendered_log_mel[:, :num_frames]
    target_db = target_frame_db[:num_frames]
    rendered_db = rendered_frame_db[:num_frames]

    frames_per_second = float(sample_rate) / float(hop_length)

    # 1. このクラスタのノートが発音中かつ元ステムが有音なフレームだけをマスクする。
    note_active_mask = np.zeros(num_frames, dtype=np.bool_)
    for note_start, note_end in zip(
        snippet.relative_start_seconds.tolist(),
        snippet.relative_end_seconds.tolist(),
    ):
        start_frame = max(0, int(math.floor(float(note_start) * frames_per_second)))
        end_frame = min(num_frames, max(start_frame + 1, int(math.ceil(float(note_end) * frames_per_second))))
        note_active_mask[start_frame:end_frame] = True

    target_floor_db = float(np.percentile(target_db, 85)) - 28.0
    rendered_floor_db = float(np.percentile(rendered_db, 85)) - 28.0
    active_mask = note_active_mask & (target_db >= target_floor_db) & (rendered_db >= rendered_floor_db)
    if int(active_mask.sum()) < 2:
        active_mask = note_active_mask
    if int(active_mask.sum()) == 0:
        active_mask = np.ones(num_frames, dtype=np.bool_)

    # 2. フレーム正規化（音量不変）スペクトル形状距離 + 長期平均スペクトル距離。
    #    1 フレームあたり dB 差をおおよそ [0, 1] 付近のスケールに収めるため 12 dB で割る。
    target_active = target_mel[:, active_mask]
    rendered_active = rendered_mel[:, active_mask]
    target_centered = target_active - target_active.mean(axis=0, keepdims=True)
    rendered_centered = rendered_active - rendered_active.mean(axis=0, keepdims=True)
    frame_spectral_distance = float(np.mean(np.abs(target_centered - rendered_centered)) / 12.0)

    target_long_term = target_centered.mean(axis=1)
    rendered_long_term = rendered_centered.mean(axis=1)
    long_term_distance = float(np.mean(np.abs(target_long_term - rendered_long_term)) / 12.0)
    spectral_distance = 0.55 * frame_spectral_distance + 0.45 * long_term_distance

    # 3. ノートごとのアタック〜減衰エンベロープ形状距離 + 持続部の振幅変調（トレモロ等）距離。
    #    各ノートのオンセット起点から 4 区間（アタック直後〜持続部）の相対 dB と、
    #    持続区間内のフレーム間 dB 変動量（トレモロ奏法や細かい振動の有無）を比較する。
    segment_edges_seconds = (0.0, 0.035, 0.080, 0.180, 0.350)
    envelope_errors: list[float] = []
    modulation_errors: list[float] = []
    for note_start, note_end in zip(
        snippet.relative_start_seconds.tolist(),
        snippet.relative_end_seconds.tolist(),
    ):
        note_duration = float(note_end) - float(note_start)
        if note_duration < 0.06:
            continue
        target_profile: list[float] = []
        rendered_profile: list[float] = []
        for edge_start, edge_end in zip(segment_edges_seconds[:-1], segment_edges_seconds[1:]):
            if edge_start >= note_duration:
                break
            window_start_frame = max(0, int(math.floor((float(note_start) + edge_start) * frames_per_second)))
            window_end_frame = min(
                num_frames,
                max(
                    window_start_frame + 1,
                    int(math.ceil((float(note_start) + min(edge_end, note_duration)) * frames_per_second)),
                ),
            )
            if window_start_frame >= num_frames:
                break
            target_profile.append(float(target_db[window_start_frame:window_end_frame].mean()))
            rendered_profile.append(float(rendered_db[window_start_frame:window_end_frame].mean()))

        if len(target_profile) >= 2:
            target_vector = np.asarray(target_profile, dtype=np.float64)
            rendered_vector = np.asarray(rendered_profile, dtype=np.float64)
            # ピーク基準（最大値を 0 dB）にして、絶対音量ではなく減衰の速さだけを比較する。
            target_vector = np.clip(target_vector - target_vector.max(), -30.0, 0.0)
            rendered_vector = np.clip(rendered_vector - rendered_vector.max(), -30.0, 0.0)
            envelope_errors.append(float(np.mean(np.abs(target_vector - rendered_vector)) / 12.0))

        # 持続部 (開始 60ms 以降) が十分あるノートで、フレーム間の音量揺れ（トレモロ等）を比較する。
        if note_duration >= 0.18:
            sustain_start = max(0, int(math.floor((float(note_start) + 0.06) * frames_per_second)))
            sustain_end = min(num_frames, int(math.ceil(float(note_end) * frames_per_second)))
            if sustain_end - sustain_start >= 4:
                target_flux = float(np.mean(np.abs(np.diff(target_db[sustain_start:sustain_end]))))
                rendered_flux = float(np.mean(np.abs(np.diff(rendered_db[sustain_start:sustain_end]))))
                modulation_errors.append(abs(target_flux - rendered_flux) / 4.0)

    if envelope_errors and modulation_errors:
        envelope_distance = 0.70 * float(np.mean(envelope_errors)) + 0.30 * float(np.mean(modulation_errors))
    elif envelope_errors:
        envelope_distance = float(np.mean(envelope_errors))
    else:
        envelope_distance = spectral_distance
    return spectral_distance, envelope_distance


@torch.inference_mode()
def _compute_rendered_slot_embeddings(
    model: InstrumentRefinementModel,
    model_config: InstrumentRefinementConfig,
    slots: list[_RenderSlot],
    slot_waveforms: list[np.ndarray],
    snippets_by_key: dict[tuple[int, int], _ClusterSnippet],
    *,
    stem_name: str | None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    batch_size: int,
) -> dict[tuple[int, int, int], np.ndarray]:
    """レンダリング済み候補スロットを既存 refinement モデルへバッチ投入し、音色埋め込み重心を得る。"""
    if not slots:
        return {}

    context_id = stem_context_id(inference_stem_group(stem_name))
    embeddings_by_slot: dict[tuple[int, int, int], np.ndarray] = {}
    # 並列レンダリング由来のチャンク順のままバッチ化すると、長短・疎密の異なるスニペットが混ざって
    # パディング比率が増大し GPU 計算量が増える。波形長とノート数で並べて同一スニペットの候補群を
    # 同一バッチに集めることで、パディングの無駄を最小化する。
    ordered_pairs = sorted(
        zip(slots, slot_waveforms),
        key=lambda pair: (
            int(pair[1].shape[1]),
            int(snippets_by_key[(pair[0].cluster_id, pair[0].snippet_index)].pitches.size),
        ),
    )
    ordered_slots = [pair[0] for pair in ordered_pairs]
    ordered_waveforms = [pair[1] for pair in ordered_pairs]

    # バッチごとに GPU 上で正規化重心ベクトル [B, D] を計算して溜め、全バッチ終了後に 1 回だけ CPU へ転送する。
    collected_slots: list[_RenderSlot] = []
    collected_centroids_on_device: list[torch.Tensor] = []

    for batch_offset in range(0, len(ordered_slots), int(batch_size)):
        batch_slots = ordered_slots[batch_offset : batch_offset + int(batch_size)]
        batch_waveforms = ordered_waveforms[batch_offset : batch_offset + int(batch_size)]

        # RecursiveCQT の最下段オクターブ STFT で reflect padding が入力長を超えないよう、
        # 短いスニペットでも最低 2.0 秒分まで 0 埋めし、実長は valid_audio_frames で伝える。
        min_backbone_samples = int(round(2.0 * float(model_config.sample_rate)))
        max_samples = max(min_backbone_samples, max(int(waveform.shape[1]) for waveform in batch_waveforms))
        padded_audio: list[torch.Tensor] = []
        valid_audio_frames: list[int] = []
        note_starts: list[torch.Tensor] = []
        note_ends: list[torch.Tensor] = []
        note_pitches: list[torch.Tensor] = []
        note_counts: list[int] = []

        for slot, waveform in zip(batch_slots, batch_waveforms):
            snippet = snippets_by_key[(slot.cluster_id, slot.snippet_index)]
            sample_count = int(waveform.shape[1])
            tensor_audio = torch.from_numpy(waveform.astype(np.float32, copy=False))
            if sample_count < max_samples:
                tensor_audio = torch.nn.functional.pad(tensor_audio, (0, max_samples - sample_count))
            padded_audio.append(tensor_audio)
            valid_audio_frames.append(sample_count)
            note_starts.append(torch.from_numpy(snippet.relative_start_seconds))
            note_ends.append(torch.from_numpy(snippet.relative_end_seconds))
            note_pitches.append(torch.from_numpy(snippet.pitches))
            note_counts.append(int(snippet.pitches.size))

        max_notes = max(note_counts)
        note_start_batch = torch.nn.utils.rnn.pad_sequence(note_starts, batch_first=True)
        note_end_batch = torch.nn.utils.rnn.pad_sequence(note_ends, batch_first=True)
        note_pitch_batch = torch.nn.utils.rnn.pad_sequence(note_pitches, batch_first=True)
        note_mask = torch.zeros((len(batch_slots), max_notes), dtype=torch.bool)
        for index, count in enumerate(note_counts):
            note_mask[index, :count] = True

        note_mask_device = note_mask.to(device)
        window_seconds = float(max_samples) / float(model_config.sample_rate)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            outputs = model(
                torch.stack(padded_audio).to(device),
                valid_audio_frames=torch.tensor(valid_audio_frames, device=device),
                note_start_seconds=note_start_batch.to(device),
                note_end_seconds=note_end_batch.to(device),
                note_pitch=note_pitch_batch.to(device),
                note_prior_class=torch.full(
                    (len(batch_slots), max_notes),
                    -1,
                    dtype=torch.long,
                    device=device,
                ),
                note_confidence=torch.ones((len(batch_slots), max_notes), device=device),
                note_mask=note_mask_device,
                stem_context_id=torch.full((len(batch_slots),), context_id, dtype=torch.long, device=device),
                window_seconds=window_seconds,
            )
        note_embeddings_dev = outputs["note_embedding"].float()
        mask_weights = note_mask_device.unsqueeze(-1).to(dtype=note_embeddings_dev.dtype)
        denom = mask_weights.sum(dim=1).clamp_min(1.0)
        centroids = (note_embeddings_dev * mask_weights).sum(dim=1) / denom
        norms = torch.linalg.vector_norm(centroids, dim=-1, keepdim=True)
        centroids = torch.where(norms > 1e-8, centroids / norms.clamp_min(1e-8), centroids)
        collected_centroids_on_device.append(centroids.detach())
        collected_slots.extend(batch_slots)
        del outputs

    if collected_centroids_on_device:
        (all_centroids_cpu,) = copy_tensors_to_cpu_once((torch.cat(collected_centroids_on_device, dim=0),))
        all_centroids_numpy = all_centroids_cpu.numpy().astype(np.float64, copy=False)
        for slot, centroid in zip(collected_slots, all_centroids_numpy):
            embeddings_by_slot[(slot.cluster_id, slot.snippet_index, slot.program_number)] = centroid

    return embeddings_by_slot


def select_gm_programs_by_render_compare(
    *,
    model: InstrumentRefinementModel,
    model_config: InstrumentRefinementConfig,
    waveform: torch.Tensor,
    notes: RefinementNoteTable,
    cluster_ids: np.ndarray,
    note_logits: np.ndarray,
    note_embeddings: np.ndarray,
    global_logits: np.ndarray,
    clusters: list[dict[str, Any]],
    allowed_class_ids: tuple[int, ...],
    global_logit_weight: float,
    stem_name: str | None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    render_compare_config: RenderCompareConfig,
    packed_midi_renderer: _PackedMidiRenderer | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """クラスタごとに Render & Compare で最適な GM プログラムを選定する。

    Returns:
        (updated_class_ids, selected_program_numbers, updated_clusters)
    """
    sample_rate = int(model_config.sample_rate)
    waveform_numpy = waveform.detach().cpu().numpy().astype(np.float32, copy=False)

    # 1. 各クラスタの候補 GM プログラムと代表スニペットを準備する。
    candidates_by_cluster: dict[int, list[_CandidateProgram]] = {}
    snippets_by_cluster: dict[int, list[_ClusterSnippet]] = {}
    snippets_by_key: dict[tuple[int, int], _ClusterSnippet] = {}

    for cluster_entry in clusters:
        cluster_id = int(cluster_entry["cluster_id"])
        member_indices = np.flatnonzero(cluster_ids == cluster_id)
        if member_indices.size == 0:
            continue
        cluster_logits = note_logits[member_indices].mean(axis=0) + float(global_logit_weight) * global_logits
        candidates = _collect_candidate_programs(
            cluster_logits,
            allowed_class_ids=allowed_class_ids,
            config=render_compare_config,
            note_count=int(member_indices.size),
        )
        candidates_by_cluster[cluster_id] = candidates
        if len(candidates) <= 1:
            continue
        snippets = _select_cluster_snippets(
            notes,
            waveform_numpy,
            cluster_ids,
            cluster_id=cluster_id,
            sample_rate=sample_rate,
            config=render_compare_config,
        )
        snippets_by_cluster[cluster_id] = snippets
        for snippet in snippets:
            snippets_by_key[(cluster_id, snippet.snippet_index)] = snippet

    # 2. 全候補を FluidSynth でレンダリングする。
    #    本番 FluidSynth 実行かつスロット数が多い場合はワーカー数分の小パック MIDI に分割して並列実行する。
    rendered_audio_by_slot: dict[tuple[int, int, int], np.ndarray] = {}
    rendered_slots: list[_RenderSlot] = []
    if any(snippets_by_cluster.values()):
        with tempfile.TemporaryDirectory(prefix="tsumugi_render_compare_") as temp_directory:
            temp_root = Path(temp_directory)

            def _slice_rendered_stereo_into_slots(
                rendered_stereo: np.ndarray,
                batch_slots: list[_RenderSlot],
            ) -> None:
                total_rendered_samples = int(rendered_stereo.shape[1])
                for slot in batch_slots:
                    snippet = snippets_by_key[(slot.cluster_id, slot.snippet_index)]
                    expected_samples = int(snippet.target_audio.shape[1])
                    start_sample = max(0, int(round(slot.slot_start_seconds * sample_rate)))
                    slot_slice = rendered_stereo[:, start_sample : min(total_rendered_samples, start_sample + expected_samples)]
                    if slot_slice.shape[1] < expected_samples:
                        slot_slice = np.pad(slot_slice, ((0, 0), (0, expected_samples - slot_slice.shape[1])))
                    rendered_audio_by_slot[(slot.cluster_id, slot.snippet_index, slot.program_number)] = slot_slice.astype(
                        np.float32,
                        copy=True,
                    )

            render_options = FluidSynthRenderOptions(
                soundfont_path=Path(render_compare_config.soundfont_path),
                sample_rate=int(sample_rate),
                fluidsynth_executable=render_compare_config.fluidsynth_executable,
                render_gain=float(render_compare_config.render_gain),
                disable_reverb=bool(render_compare_config.disable_reverb),
                disable_chorus=bool(render_compare_config.disable_chorus),
                polyphony=64,
            )

            total_candidate_slots = sum(
                len(snippets_by_cluster[cid]) * len(candidates_by_cluster.get(cid, []))
                for cid in snippets_by_cluster
                if len(candidates_by_cluster.get(cid, [])) > 1
            )
            parallel_workers = (
                default_parallel_render_workers(
                    Path(render_compare_config.soundfont_path),
                    max(1, total_candidate_slots // 6),
                    max_workers=render_compare_config.parallel_render_workers,
                )
                if packed_midi_renderer is None and total_candidate_slots >= 12
                else 1
            )

            if parallel_workers <= 1:
                packed_midi_path = temp_root / "candidates_packed.mid"
                rendered_slots = _build_packed_candidate_midi(
                    snippets_by_cluster,
                    candidates_by_cluster,
                    output_midi_path=packed_midi_path,
                    config=render_compare_config,
                )
                if rendered_slots:
                    if packed_midi_renderer is not None:
                        packed_wav_path = temp_root / "candidates_packed.wav"
                        packed_midi_renderer(packed_midi_path, packed_wav_path, sample_rate)
                        rendered_full, rendered_sample_rate = sf.read(
                            str(packed_wav_path),
                            dtype="float32",
                            always_2d=True,
                        )
                        if int(rendered_sample_rate) != sample_rate:
                            raise ValueError(
                                f"Rendered sample rate mismatch: expected {sample_rate}, got {rendered_sample_rate}"
                            )
                        rendered_stereo = (
                            np.repeat(rendered_full.T, 2, axis=0)
                            if rendered_full.shape[1] == 1
                            else rendered_full[:, :2].T
                        )
                    else:
                        rendered_stereo = render_midi_to_stereo_array(
                            packed_midi_path,
                            options=render_options,
                        )
                    _slice_rendered_stereo_into_slots(rendered_stereo, rendered_slots)
            else:
                # 候補をワーカー数へ均等にラウンドロビン分割し、常駐 FluidSynth プールで並列合成する。
                chunk_candidates: list[dict[int, list[_CandidateProgram]]] = [
                    {} for _ in range(parallel_workers)
                ]
                chunk_snippets: list[dict[int, list[_ClusterSnippet]]] = [
                    {} for _ in range(parallel_workers)
                ]
                round_robin_index = 0
                for cluster_id in sorted(snippets_by_cluster):
                    snippets = snippets_by_cluster[cluster_id]
                    candidates = candidates_by_cluster.get(cluster_id, [])
                    if not snippets or len(candidates) <= 1:
                        continue
                    for candidate in candidates:
                        bucket_index = round_robin_index % parallel_workers
                        chunk_candidates[bucket_index].setdefault(cluster_id, []).append(candidate)
                        chunk_snippets[bucket_index][cluster_id] = snippets
                        round_robin_index += 1

                chunk_midi_paths: list[Path] = []
                batch_slot_lists: list[list[_RenderSlot]] = []
                for chunk_index in range(parallel_workers):
                    if not chunk_candidates[chunk_index]:
                        continue
                    # 1 候補だけになったクラスタも除外されないよう、候補数チェックを一時バイパスして書き出す。
                    chunk_midi_path = temp_root / f"candidates_chunk_{chunk_index}.mid"
                    chunk_slots = _build_packed_candidate_midi(
                        chunk_snippets[chunk_index],
                        chunk_candidates[chunk_index],
                        output_midi_path=chunk_midi_path,
                        config=render_compare_config,
                        minimum_candidates_per_cluster=1,
                    )
                    if chunk_slots:
                        chunk_midi_paths.append(chunk_midi_path)
                        batch_slot_lists.append(chunk_slots)
                        rendered_slots.extend(chunk_slots)

                rendered_chunk_waveforms = render_midi_batch_to_stereo_arrays(
                    chunk_midi_paths,
                    options=render_options,
                    max_workers=parallel_workers,
                )
                for chunk_stereo, chunk_slots in zip(rendered_chunk_waveforms, batch_slot_lists):
                    _slice_rendered_stereo_into_slots(chunk_stereo, chunk_slots)

    # 3. レンダリング済み候補のニューラル音色埋め込みをバッチ計算する（重み > 0 のとき）。
    rendered_embeddings: dict[tuple[int, int, int], np.ndarray] = {}
    if rendered_slots and render_compare_config.embedding_weight > 0.0:
        slot_waveforms = [
            rendered_audio_by_slot[(slot.cluster_id, slot.snippet_index, slot.program_number)]
            for slot in rendered_slots
        ]
        rendered_embeddings = _compute_rendered_slot_embeddings(
            model,
            model_config,
            rendered_slots,
            slot_waveforms,
            snippets_by_key,
            stem_name=stem_name,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            batch_size=render_compare_config.embedding_batch_size,
        )

    # 4. 元ステム側およびレンダリング候補側の対数メル特徴を GPU 上で同一長バッチ STFT により一括計算する。
    max_frequency_hz = min(8000.0, float(sample_rate) * 0.45)
    min_frequency_hz = min(40.0, max_frequency_hz * 0.25)
    n_fft = min(1024, max(32, 2 ** int(math.floor(math.log2(max(32.0, float(sample_rate) * 0.05))))))
    hop_length = max(8, n_fft // 4)
    num_mels = min(64, max(12, n_fft // 4))
    mel_filterbank = _build_mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        num_mels=num_mels,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
    )

    signal_device = resolve_signal_processing_device()
    hann_window_tensor = torch.hann_window(n_fft, dtype=torch.float32, device=signal_device)
    mel_filterbank_tensor = torch.from_numpy(mel_filterbank).to(device=signal_device, dtype=torch.float32)

    # 同一スニペットに属する全候補は同一サンプル長を持つため、サンプル長ごとにまとめてバッチ STFT する。
    # 末尾ゼロ埋めによる反射境界の変化を起こさず、単発 STFT と数値同一のまま GPU 並列化できる。
    grouped_mono_waveforms: dict[int, list[tuple[tuple[Any, ...], np.ndarray]]] = {}
    for key, snippet in snippets_by_key.items():
        mono = snippet.target_audio.mean(axis=0).astype(np.float32, copy=False)
        if mono.size < n_fft:
            mono = np.pad(mono, (0, n_fft - mono.size))
        grouped_mono_waveforms.setdefault(int(mono.size), []).append((("target", key), mono))
    for slot_key, rendered_audio in rendered_audio_by_slot.items():
        mono = rendered_audio.mean(axis=0).astype(np.float32, copy=False)
        if mono.size < n_fft:
            mono = np.pad(mono, (0, n_fft - mono.size))
        grouped_mono_waveforms.setdefault(int(mono.size), []).append((("rendered", slot_key), mono))

    batched_features_by_tag: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    with torch.inference_mode():
        for group_items in grouped_mono_waveforms.values():
            batch_array = np.stack([item[1] for item in group_items], axis=0)
            batch_tensor = torch.from_numpy(batch_array).to(device=signal_device, dtype=torch.float32)
            spectrum = torch.stft(
                batch_tensor,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=hann_window_tensor,
                center=True,
                return_complex=True,
            )
            # power: [B, freq_bins, num_frames] -> mel_power: [B, num_mels, num_frames]
            power = spectrum.real.square() + spectrum.imag.square()
            mel_power = torch.einsum("mf,bft->bmt", mel_filterbank_tensor, power)
            log_mel_batch = (10.0 * torch.log10(mel_power.clamp_min(1e-10))).cpu().numpy().astype(np.float32)
            frame_db_batch = (10.0 * torch.log10(power.mean(dim=1).clamp_min(1e-10))).cpu().numpy().astype(np.float32)
            for item_index, (tag, _mono) in enumerate(group_items):
                batched_features_by_tag[tag] = (log_mel_batch[item_index], frame_db_batch[item_index])

    target_features_by_key: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, snippet in snippets_by_key.items():
        target_log_mel, target_frame_db = batched_features_by_tag[("target", key)]
        target_centroid = note_embeddings[snippet.note_indices].mean(axis=0).astype(np.float64)
        norm = float(np.linalg.norm(target_centroid))
        if norm > 1e-8:
            target_centroid /= norm
        target_features_by_key[key] = (target_log_mel, target_frame_db, target_centroid)

    # 5. クラスタごとに全候補 GM プログラムの総合距離を集計し、最良プログラムを確定する。
    updated_class_ids = np.zeros(notes.note_count, dtype=np.int64)
    selected_program_numbers = np.zeros(notes.note_count, dtype=np.int64)
    updated_clusters: list[dict[str, Any]] = []

    for cluster_entry in clusters:
        cluster_id = int(cluster_entry["cluster_id"])
        member_indices = np.flatnonzero(cluster_ids == cluster_id)
        candidates = candidates_by_cluster.get(cluster_id, [])
        snippets = snippets_by_cluster.get(cluster_id, [])

        # 候補が 0〜1 個、または評価スニペットが作れなかった場合は既存クラス代表へフォールバック。
        if not candidates or len(candidates) == 1 or not snippets:
            if candidates:
                best_program = candidates[0].program_number
                best_class_id = candidates[0].class_id
            else:
                best_class_id = int(cluster_entry["candidates"][0]["class_id"])
                fallback_programs = get_program_numbers_for_class_id(best_class_id)
                best_program = int(fallback_programs[0]) if fallback_programs else 0
            updated_class_ids[member_indices] = best_class_id
            selected_program_numbers[member_indices] = best_program
            updated_entry = dict(cluster_entry)
            updated_entry["selected_program"] = int(best_program)
            updated_entry["selected_program_name"] = get_gm_program_name(best_program)
            updated_entry["selected_class_id"] = int(best_class_id)
            updated_entry["selected_class_name"] = INSTRUMENT_CLASSES[best_class_id]
            updated_entry["render_compare_snippets"] = []
            updated_entry["render_compare_candidates"] = []
            updated_clusters.append(updated_entry)
            continue

        # クラス内での GM プログラム出現頻度（song_group_count）を正規化し、
        # 特殊奏法パッチ (Tremolo Strings 等) への過大な偏りを抑える弱い事前項を作る。
        class_frequency_totals: dict[int, float] = {}
        for candidate in candidates:
            frequency = float(max(1, get_gm_program_frequency(candidate.program_number)))
            class_frequency_totals[candidate.class_id] = (
                class_frequency_totals.get(candidate.class_id, 0.0) + frequency
            )

        candidate_reports: list[dict[str, Any]] = []
        for candidate in candidates:
            weighted_embedding_sum = 0.0
            weighted_spectral_sum = 0.0
            weighted_envelope_sum = 0.0
            weight_total = 0.0
            rms_dbfs_values: list[float] = []

            for snippet in snippets:
                slot_key = (cluster_id, snippet.snippet_index, candidate.program_number)
                rendered_audio = rendered_audio_by_slot[slot_key]
                rms_value = float(np.sqrt(np.mean(np.square(rendered_audio), dtype=np.float64) + 1e-12))
                rms_dbfs = 20.0 * math.log10(max(rms_value, 1e-8))
                rms_dbfs_values.append(rms_dbfs)

                target_log_mel, target_frame_db, target_centroid = target_features_by_key[
                    (cluster_id, snippet.snippet_index)
                ]
                rendered_log_mel, rendered_frame_db = batched_features_by_tag[("rendered", slot_key)]
                spectral_distance, envelope_distance = _compute_spectral_and_envelope_distances(
                    target_log_mel,
                    target_frame_db,
                    rendered_log_mel,
                    rendered_frame_db,
                    snippet,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                )

                if slot_key in rendered_embeddings:
                    cosine_similarity = float(np.clip(np.dot(target_centroid, rendered_embeddings[slot_key]), -1.0, 1.0))
                    embedding_distance = 1.0 - cosine_similarity
                else:
                    embedding_distance = 0.0

                snippet_weight = max(1.0, float(snippet.pitches.size)) * max(0.1, float(snippet.purity))
                weighted_embedding_sum += embedding_distance * snippet_weight
                weighted_spectral_sum += spectral_distance * snippet_weight
                weighted_envelope_sum += envelope_distance * snippet_weight
                weight_total += snippet_weight

            mean_embedding_distance = weighted_embedding_sum / max(weight_total, 1e-8)
            mean_spectral_distance = weighted_spectral_sum / max(weight_total, 1e-8)
            mean_envelope_distance = weighted_envelope_sum / max(weight_total, 1e-8)
            mean_rms_dbfs = float(np.mean(rms_dbfs_values))

            within_class_probability = float(max(1, get_gm_program_frequency(candidate.program_number))) / max(
                class_frequency_totals[candidate.class_id],
                1.0,
            )
            prior_penalty = -float(render_compare_config.classifier_prior_weight) * math.log(
                max(candidate.prior_probability, 1e-4)
            ) - float(render_compare_config.program_prior_weight) * math.log(max(within_class_probability, 1e-3))
            acoustic_distance = (
                float(render_compare_config.embedding_weight) * mean_embedding_distance
                + float(render_compare_config.spectral_weight) * mean_spectral_distance
                + float(render_compare_config.envelope_weight) * mean_envelope_distance
            )

            candidate_reports.append(
                {
                    "program_number": int(candidate.program_number),
                    "program_name": get_gm_program_name(candidate.program_number),
                    "class_id": int(candidate.class_id),
                    "class_name": INSTRUMENT_CLASSES[candidate.class_id],
                    "prior_probability": float(candidate.prior_probability),
                    "within_class_probability": float(within_class_probability),
                    "total_distance": float(acoustic_distance + prior_penalty),
                    "acoustic_distance": float(acoustic_distance),
                    "embedding_distance": float(mean_embedding_distance),
                    "spectral_distance": float(mean_spectral_distance),
                    "envelope_distance": float(mean_envelope_distance),
                    "prior_penalty": float(prior_penalty),
                    "rendered_rms_dbfs": float(mean_rms_dbfs),
                    "audible": True,
                }
            )

        # 絶対無音ゲート + 相対無音ゲート（音域外で極端に音が小さくなったパッチを除外）。
        rms_all = np.asarray([entry["rendered_rms_dbfs"] for entry in candidate_reports], dtype=np.float64)
        reference_rms_dbfs = float(np.percentile(rms_all, 75))
        relative_rms_floor = reference_rms_dbfs - float(render_compare_config.max_relative_rms_drop_db)
        effective_rms_floor = max(float(render_compare_config.min_rendered_rms_dbfs), relative_rms_floor)
        for entry in candidate_reports:
            is_audible = float(entry["rendered_rms_dbfs"]) >= effective_rms_floor
            entry["audible"] = bool(is_audible)
            if not is_audible:
                entry["total_distance"] = float(entry["total_distance"]) + 100.0

        candidate_reports.sort(key=lambda entry: (entry["total_distance"], entry["program_number"]))
        winner = candidate_reports[0]
        best_program = int(winner["program_number"])
        best_class_id = int(get_instrument_class_id(best_program, is_drum=False))

        updated_class_ids[member_indices] = best_class_id
        selected_program_numbers[member_indices] = best_program

        updated_entry = dict(cluster_entry)
        updated_entry["selected_program"] = best_program
        updated_entry["selected_program_name"] = str(winner["program_name"])
        updated_entry["selected_class_id"] = best_class_id
        updated_entry["selected_class_name"] = INSTRUMENT_CLASSES[best_class_id]
        updated_entry["render_compare_snippets"] = [
            {
                "snippet_index": snippet.snippet_index,
                "start_seconds": snippet.start_seconds,
                "duration_seconds": snippet.duration_seconds,
                "note_count": int(snippet.pitches.size),
                "purity": snippet.purity,
            }
            for snippet in snippets
        ]
        updated_entry["render_compare_candidates"] = candidate_reports[: int(render_compare_config.top_report_programs)]
        updated_clusters.append(updated_entry)

    return updated_class_ids, selected_program_numbers, updated_clusters
