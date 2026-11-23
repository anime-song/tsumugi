"""SheetSage2-Refined メロディと tsumugi ボーカル／ハモリ採譜の音声参照融合。

SheetSage2-Refined (0.5.2+) のデコーダーなしメロディ推定 (HeadMelody) は単旋律の
ピッチ・フレーズ精度が非常に高い一方、原曲ミックスを直接聴くため以下の 2 点が課題になる。

1. イントロ・間奏・ギターソロ等の「歌声以外の楽器メロディ」も採譜される。
2. 単旋律モデルのため、ボーカルハモリ (vocal_harmony) は採譜されない。

一方、tsumugi (`vocal_harmony_v1_5`) は分離済み `vocals` ステムから主旋律とハモリを
両方採譜できるが、主旋律トラックへのハモリ混入・ハモリトラックへの主旋律混入が起き得る。

本モジュールでは以下の 2 段階で両者の長所だけを統合する。

1. **分離ボーカルステム音声による歌唱ゲート**:
   - `vocals` ステム全体が無音ならインスト曲と判定して SheetSage2 ノートをすべて棄却する。
   - 各 SheetSage2 ノート区間について、(a) ボーカル帯域 (150〜4500 Hz) エネルギー、
     (b) 原曲ミックスに対するボーカル比 (vocal-to-mix dB)、(c) 該当ピッチの倍音エネルギー、
     (d) SheetSage2 自身の `melody_vocal` 確率を統合評価し、実際に歌われている主旋律だけを通す。
2. **確定主旋律アンカーによる主旋律／ハモリ再仕分け**:
   - ゲート通過後の SheetSage2 主旋律を確定アンカーとし、tsumugi が出力した全ノート
     (`melody` + `vocal_harmony`) を元ラベルに縛られず再分類する。
   - アンカーと同時間帯・同音高のノートは、元ラベルが `vocal_harmony` でも主旋律の重複として破棄。
   - アンカーと同時間帯・別音高 (2〜19 半音差) のノートは、元ラベルが `melody` でもハモリへ収容。
   - アンカーが休符の区間では、コーラス／追っかけ (`vocal_harmony`) を保持し、
     明確に独立した強歌唱フレーズ（アンカー端の微小ズレを除く）のみ主旋律へ補完する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import torch

from ..soundfont import resolve_signal_processing_device
from ..velocity.render_compare_velocity import _build_harmonic_pitch_filterbank, _load_stereo_audio
from .sheetsage import SheetSageMelodyNote, SheetSageMelodyPrediction


@dataclass(frozen=True)
class VocalMelodyFusionConfig:
    """SheetSage2 メロディと tsumugi ボーカル／ハモリ融合の閾値設定。"""

    sample_rate: int = 16_000
    n_fft: int = 1024
    # 8 ms フレーム分解能により、スウィング／3 連符のオンセット比較を高精度に行う。
    hop_length: int = 128
    vocal_band_min_hz: float = 150.0
    vocal_band_max_hz: float = 4500.0

    # インスト曲（歌唱なし）を誤ってボーカル化しないための曲全体ガード。
    instrumental_song_p95_dbfs: float = -42.0
    instrumental_song_min_active_seconds: float = 0.45

    # ノート別ボーカルゲート（楽器メロディ除去 + 強歌唱救済）。
    minimum_note_duration_seconds: float = 0.04
    hard_silence_dbfs: float = -50.0
    max_drop_below_vocal_reference_db: float = 24.0
    minimum_vocal_to_mix_db: float = -20.0
    sheetsage_sung_probability_threshold: float = 0.38
    minimum_pitch_support_above_floor_db: float = 3.5

    # SheetSage2 が track=1 (played) と誤判定した強い歌声（エフェクト・ファルセット等）の救済条件。
    played_rescue_max_drop_db: float = 12.0
    played_rescue_min_vocal_to_mix_db: float = -10.0

    # 主旋律アンカー vs tsumugi ノート再仕分けのルール。
    lead_match_overlap_ratio: float = 0.25
    harmony_minimum_semitone_gap: int = 2
    harmony_maximum_semitone_gap: int = 19
    harmony_minimum_duration_seconds: float = 0.07
    suppress_melody_labeled_octave_ghosts: bool = True
    # melody ラベルからハモリへ昇格する際、主旋律ピッチ比で極端に弱い音程ズレ誤検出を弾く閾値 (dB)。
    harmony_promotion_max_drop_below_lead_pitch_db: float = 14.0

    # SheetSage2 が休符の区間に残った独立歌唱フレーズを補完する条件（境界の微小ズレによる二重発音を防ぐ）。
    fill_uncovered_lead_gap_margin_seconds: float = 0.15
    fill_uncovered_lead_min_duration_seconds: float = 0.08
    fill_uncovered_lead_max_drop_db: float = 18.0
    fill_uncovered_lead_min_vocal_to_mix_db: float = -14.0

    # SheetSage2 の 16 分量子化バイアスを補正する音声参照スウィング／ルバート復元。
    # ストレート音符 (< 25ms) は SheetSage2 の安定したリズムを保ちつつ、
    # スウィング・3 連符・タメ (25〜135ms 差) で実ボーカル音声が明確に tsumugi 側で立ち上がる場合のみ復元する。
    refine_lead_timing_for_swing: bool = True
    swing_refine_min_diff_seconds: float = 0.025
    swing_refine_max_diff_seconds: float = 0.135
    swing_refine_min_onset_gain_db: float = 3.0
    swing_refine_min_onset_ratio: float = 1.30

    # SheetSage2 主旋律に対するハモリのリズム揃え（モデル間遅延補正＋同時音節アタック／リリース吸着）。
    align_harmony_to_lead: bool = True
    max_global_timing_bias_seconds: float = 0.08
    harmony_internal_split_merge_gap_seconds: float = 0.055
    harmony_onset_snap_window_seconds: float = 0.075
    harmony_offset_snap_window_seconds: float = 0.085

    # 出力 MIDI トラック規約（tsumugi の EXTRA_INSTRUMENT_CLASS_PROGRAMS と同一）。
    melody_program: int = 65
    harmony_program: int = 65
    default_velocity: int = 100


@dataclass(frozen=True)
class _SourceVocalNote:
    start_seconds: float
    end_seconds: float
    pitch: int
    velocity: int
    is_harmony_label: bool


def _is_harmony_instrument_name(name: str) -> bool:
    normalized = name.strip().lower()
    return "harmony" in normalized or "choir" in normalized or "backing" in normalized or "chorus" in normalized


def _extract_tsumugi_vocal_notes(midi_path: Path | None) -> list[_SourceVocalNote]:
    """tsumugi の vocals ステム MIDI から全ノートを元ラベル付きで取り出す。"""
    if midi_path is None or not Path(midi_path).is_file():
        return []
    midi_object = pretty_midi.PrettyMIDI(str(midi_path))
    extracted: list[_SourceVocalNote] = []
    for instrument in midi_object.instruments:
        if instrument.is_drum:
            continue
        is_harmony = _is_harmony_instrument_name(instrument.name)
        for note in instrument.notes:
            start_seconds = float(note.start)
            end_seconds = float(note.end)
            if end_seconds <= start_seconds:
                continue
            extracted.append(
                _SourceVocalNote(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    pitch=int(note.pitch),
                    velocity=max(1, min(127, int(note.velocity))),
                    is_harmony_label=is_harmony,
                )
            )
    extracted.sort(key=lambda item: (item.start_seconds, item.pitch, item.end_seconds))
    return extracted


def _compute_vocal_and_mix_acoustic_maps(
    vocal_mono: np.ndarray,
    mix_mono: np.ndarray,
    *,
    config: VocalMelodyFusionConfig,
    pitch_min: int = 21,
    pitch_max: int = 108,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """ボーカルステムと原曲ミックスの帯域パワー列 [T] とピッチ倍音マップ [88, T] を GPU 優先で算出する。"""
    sample_rate = int(config.sample_rate)
    n_fft = int(config.n_fft)
    hop_length = int(config.hop_length)
    frames_per_second = float(sample_rate) / float(hop_length)

    target_length = max(n_fft, int(vocal_mono.size), int(mix_mono.size))
    vocal_padded = np.zeros(target_length, dtype=np.float32)
    mix_padded = np.zeros(target_length, dtype=np.float32)
    vocal_padded[: vocal_mono.size] = vocal_mono.astype(np.float32, copy=False)
    mix_padded[: mix_mono.size] = mix_mono.astype(np.float32, copy=False)

    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    band_mask = (frequencies >= float(config.vocal_band_min_hz)) & (frequencies <= float(config.vocal_band_max_hz))
    band_weights = np.zeros_like(frequencies, dtype=np.float32)
    # 片側スペクトルのパーセバル正規化 (DC / Nyquist 以外は 2 倍) で時間領域パワーと整合させる。
    band_weights[band_mask] = 2.0
    if band_mask[0]:
        band_weights[0] = 1.0
    if n_fft % 2 == 0 and band_mask[-1]:
        band_weights[-1] = 1.0

    pitch_filterbank, _ = _build_harmonic_pitch_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
    )

    signal_device = resolve_signal_processing_device()
    with torch.inference_mode():
        window = torch.hann_window(n_fft, dtype=torch.float32, device=signal_device)
        window_energy = max(float(window.square().sum().item()), 1e-8)
        batch_wave = torch.from_numpy(np.stack((vocal_padded, mix_padded), axis=0)).to(signal_device)
        spectrum = torch.stft(
            batch_wave,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        # power: [2, freq_bins, num_frames]
        power = (spectrum.real.square() + spectrum.imag.square()) / window_energy
        band_weights_tensor = torch.from_numpy(band_weights).to(signal_device)
        vocal_band_power = ((band_weights_tensor @ power[0]) / float(n_fft)).cpu().numpy()
        mix_band_power = ((band_weights_tensor @ power[1]) / float(n_fft)).cpu().numpy()

        pitch_fb_tensor = torch.from_numpy(np.ascontiguousarray(pitch_filterbank, dtype=np.float32)).to(signal_device)
        vocal_pitch_power = (pitch_fb_tensor @ power[0]).cpu().numpy()

    return (
        vocal_band_power.astype(np.float64, copy=False),
        mix_band_power.astype(np.float64, copy=False),
        vocal_pitch_power.astype(np.float64, copy=False),
        frames_per_second,
    )


def _segment_db_stats(
    start_seconds: float,
    end_seconds: float,
    pitch: int,
    *,
    vocal_band_db: np.ndarray,
    mix_band_db: np.ndarray,
    vocal_pitch_db: np.ndarray,
    frames_per_second: float,
    pitch_min: int = 21,
    pitch_max: int = 108,
) -> tuple[float, float, float]:
    """ノート区間における (ボーカル帯域 dB, vocal-to-mix 比 dB, ピッチ倍音支持 dB) を返す。"""
    num_frames = int(vocal_band_db.size)
    start_frame = max(0, min(num_frames - 1, int(math.floor(start_seconds * frames_per_second))))
    end_frame = min(num_frames, max(start_frame + 1, int(math.ceil(end_seconds * frames_per_second))))

    vocal_segment = vocal_band_db[start_frame:end_frame]
    mix_segment = mix_band_db[start_frame:end_frame]
    # 75 パーセンタイルでノート端のブレス・無音の影響を抑え、発音本体の音量を見る。
    vocal_level_db = float(np.percentile(vocal_segment, 75))
    mix_level_db = float(np.percentile(mix_segment, 75))
    vocal_to_mix_db = vocal_level_db - mix_level_db

    clamped_pitch = max(pitch_min, min(pitch_max, int(pitch)))
    pitch_row = clamped_pitch - pitch_min
    pitch_segment = vocal_pitch_db[pitch_row, start_frame:end_frame]
    # ビブラートや微小なピッチ揺れを考慮し、該当ピッチおよび ±1 半音の最大倍音エネルギーを見る。
    low_row = max(0, pitch_row - 1)
    high_row = min(vocal_pitch_db.shape[0], pitch_row + 2)
    pitch_peak_db = float(np.percentile(vocal_pitch_db[low_row:high_row, start_frame:end_frame].max(axis=0), 75))
    frame_pitch_floor_db = float(np.median(vocal_pitch_db[:, start_frame:end_frame]))
    pitch_support_db = max(float(np.percentile(pitch_segment, 75)), pitch_peak_db) - frame_pitch_floor_db
    return vocal_level_db, vocal_to_mix_db, pitch_support_db


def _time_overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _measure_vocal_onset_jump_db(
    time_seconds: float,
    pitch: int,
    *,
    vocal_band_db: np.ndarray,
    vocal_pitch_db: np.ndarray,
    frames_per_second: float,
    pitch_min: int = 21,
    pitch_max: int = 108,
) -> float:
    """指定時刻における該当ピッチおよびボーカル帯域の立ち上がり跳ね上がり量 (dB) を測る。

    直前区間 (約 40ms) と直後区間 (約 40ms) の音量差を ±16ms 窓の最大値で評価するため、
    本当のオンセット位置では大きな正の dB になり、発音途中の定常部や発音前の無音部では 0 dB 付近になる。
    """
    num_frames = int(vocal_band_db.size)
    if num_frames < 8:
        return 0.0
    center_frame = max(0, min(num_frames - 1, int(round(time_seconds * frames_per_second))))
    window_frames = max(2, int(round(0.040 * frames_per_second)))
    search_radius = max(1, int(round(0.016 * frames_per_second)))

    clamped_pitch = max(pitch_min, min(pitch_max, int(pitch)))
    pitch_row = clamped_pitch - pitch_min
    low_row = max(0, pitch_row - 1)
    high_row = min(vocal_pitch_db.shape[0], pitch_row + 2)
    pitch_track = vocal_pitch_db[low_row:high_row].max(axis=0)

    best_jump_db = 0.0
    for frame_index in range(max(1, center_frame - search_radius), min(num_frames - 1, center_frame + search_radius + 1)):
        pre_start = max(0, frame_index - window_frames)
        post_end = min(num_frames, frame_index + window_frames)
        if frame_index <= pre_start or post_end <= frame_index:
            continue
        pitch_jump = max(
            0.0,
            float(pitch_track[frame_index:post_end].mean()) - float(pitch_track[pre_start:frame_index].mean()),
        )
        band_jump = max(
            0.0,
            float(vocal_band_db[frame_index:post_end].mean()) - float(vocal_band_db[pre_start:frame_index].mean()),
        )
        combined_jump = 0.65 * pitch_jump + 0.35 * band_jump
        if combined_jump > best_jump_db:
            best_jump_db = combined_jump
    return best_jump_db


def _deduplicate_monophonic_notes(notes: list[pretty_midi.Note]) -> list[pretty_midi.Note]:
    """主旋律トラックの重複・逆転を整えてクリーンな単旋律列にする。"""
    ordered = sorted(notes, key=lambda item: (item.start, item.end, item.pitch))
    cleaned: list[pretty_midi.Note] = []
    for note in ordered:
        if note.end - note.start < 0.02:
            continue
        if cleaned and note.start < cleaned[-1].end:
            previous = cleaned[-1]
            if note.start - previous.start < 0.03 and note.pitch == previous.pitch:
                previous.end = max(previous.end, note.end)
                continue
            previous.end = max(previous.start + 0.02, note.start)
        cleaned.append(note)
    return cleaned


def _deduplicate_same_pitch_harmony_notes(notes: list[pretty_midi.Note]) -> list[pretty_midi.Note]:
    """ハモリトラック内の同ピッチ重複ノートを整理する（連続同音高の別音節は潰さず、真の重複のみ統合）。"""
    by_pitch: dict[int, list[pretty_midi.Note]] = {}
    for note in notes:
        by_pitch.setdefault(int(note.pitch), []).append(note)

    merged_all: list[pretty_midi.Note] = []
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda item: (item.start, item.end))
        merged_pitch: list[pretty_midi.Note] = []
        for note in pitch_notes:
            if note.end - note.start < 0.02:
                continue
            if not merged_pitch:
                merged_pitch.append(note)
                continue
            previous = merged_pitch[-1]
            overlap = _time_overlap_seconds(previous.start, previous.end, note.start, note.end)
            min_duration = max(min(previous.end - previous.start, note.end - note.start), 1e-6)
            if overlap >= 0.35 * min_duration or abs(note.start - previous.start) < 0.03:
                # 同一ノートの重複検出（melody/harmony 両トラックから同じ音が来た場合など）は 1 本にまとめる。
                previous.end = max(previous.end, note.end)
                previous.velocity = max(previous.velocity, note.velocity)
            else:
                # レガートで連続する同音高の別音節は結合せず、わずかな重なりだけ前ノートの終端を詰めて分離する。
                if note.start < previous.end:
                    previous.end = max(previous.start + 0.02, note.start)
                merged_pitch.append(note)
        merged_all.extend(merged_pitch)
    merged_all.sort(key=lambda item: (item.start, item.pitch, item.end))
    return merged_all


def _align_harmony_notes_to_lead_rhythm(
    harmony_notes: list[pretty_midi.Note],
    lead_anchor_notes: list[SheetSageMelodyNote],
    *,
    start_bias_seconds: float,
    end_bias_seconds: float,
    config: VocalMelodyFusionConfig,
) -> tuple[list[pretty_midi.Note], int, int]:
    """tsumugi 由来ハモリのリズムを SheetSage2 主旋律アンカーへ揃える。

    1. 一致主旋律ペアから求めたモデル間グローバル遅延 (start_bias / end_bias) を全体適用する。
    2. 単一の主旋律ノート内で細切れになった同音高ハモリ (隙間 <= 55ms) を結合する。
    3. 主旋律の開始 (±75ms)・終了 (±85ms) に近い同時音節ハモリを主旋律へ厳密吸着する。
       一方、対位法や追っかけ（主旋律開始から 75ms 超離れた音）や複数音にまたがるロングハモリの中間部は
       壊さずに保持する。
    """
    if not harmony_notes or not lead_anchor_notes:
        return harmony_notes, 0, 0

    # 1. モデル間の系統的時間オフセット補正。
    shifted_notes: list[pretty_midi.Note] = []
    for note in harmony_notes:
        shifted_start = max(0.0, float(note.start) + start_bias_seconds)
        shifted_end = max(shifted_start + 0.03, float(note.end) + end_bias_seconds)
        shifted_notes.append(
            pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=shifted_start,
                end=shifted_end,
            )
        )

    # 2. 単一の主旋律ノート発音中に細切れになった同音高ハモリを結合する。
    merge_gap = float(config.harmony_internal_split_merge_gap_seconds)
    by_pitch: dict[int, list[pretty_midi.Note]] = {}
    for note in shifted_notes:
        by_pitch.setdefault(int(note.pitch), []).append(note)

    consolidated_notes: list[pretty_midi.Note] = []
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda item: (item.start, item.end))
        merged_pitch: list[pretty_midi.Note] = []
        for note in pitch_notes:
            if not merged_pitch:
                merged_pitch.append(note)
                continue
            previous = merged_pitch[-1]
            gap_seconds = note.start - previous.end
            if 0.0 <= gap_seconds <= merge_gap:
                inside_single_lead = any(
                    lead.start_seconds - 0.05 <= previous.start and note.end <= lead.end_seconds + 0.05
                    for lead in lead_anchor_notes
                )
                if inside_single_lead:
                    previous.end = max(previous.end, note.end)
                    previous.velocity = max(previous.velocity, note.velocity)
                    continue
            merged_pitch.append(note)
        consolidated_notes.extend(merged_pitch)

    # 3. 同時音節の開始・終了を主旋律アンカーへ厳密吸着する。
    lead_starts = np.asarray([float(lead.start_seconds) for lead in lead_anchor_notes], dtype=np.float64)
    lead_ends = np.asarray([float(lead.end_seconds) for lead in lead_anchor_notes], dtype=np.float64)
    onset_window = float(config.harmony_onset_snap_window_seconds)
    offset_window = float(config.harmony_offset_snap_window_seconds)

    snapped_onset_count = 0
    snapped_offset_count = 0
    aligned_notes: list[pretty_midi.Note] = []

    for note in consolidated_notes:
        new_start = float(note.start)
        new_end = float(note.end)

        nearest_start_index = int(np.argmin(np.abs(lead_starts - new_start)))
        nearest_start = float(lead_starts[nearest_start_index])
        if abs(new_start - nearest_start) <= onset_window and (new_end - nearest_start) >= 0.04:
            new_start = nearest_start
            snapped_onset_count += 1

        nearest_end_index = int(np.argmin(np.abs(lead_ends - new_end)))
        nearest_end = float(lead_ends[nearest_end_index])
        if abs(new_end - nearest_end) <= offset_window and (nearest_end - new_start) >= 0.04:
            new_end = nearest_end
            snapped_offset_count += 1

        aligned_notes.append(
            pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=new_start,
                end=max(new_start + 0.03, new_end),
            )
        )

    return aligned_notes, snapped_onset_count, snapped_offset_count


def fuse_sheetsage_melody_with_tsumugi_vocals(
    sheetsage_prediction: SheetSageMelodyPrediction,
    *,
    vocal_audio_path: Path | str | None = None,
    mix_audio_path: Path | str | None = None,
    tsumugi_vocal_midi_path: Path | str | None = None,
    output_midi_path: Path | str | None = None,
    preloaded_vocal_mono: np.ndarray | None = None,
    preloaded_mix_mono: np.ndarray | None = None,
    config: VocalMelodyFusionConfig | None = None,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any]]:
    """SheetSage2 メロディを分離ボーカル音声でゲートし、tsumugi の主旋律／ハモリと融合する。"""
    effective_config = config or VocalMelodyFusionConfig()
    sample_rate = int(effective_config.sample_rate)

    if preloaded_vocal_mono is not None:
        vocal_mono = np.asarray(preloaded_vocal_mono, dtype=np.float32)
    elif vocal_audio_path is not None:
        vocal_mono = _load_stereo_audio(Path(vocal_audio_path), target_sample_rate=sample_rate).mean(axis=0)
    else:
        raise ValueError("Either preloaded_vocal_mono or vocal_audio_path must be provided")

    if preloaded_mix_mono is not None:
        mix_mono = np.asarray(preloaded_mix_mono, dtype=np.float32)
    elif mix_audio_path is not None:
        mix_mono = _load_stereo_audio(Path(mix_audio_path), target_sample_rate=sample_rate).mean(axis=0)
    else:
        mix_mono = vocal_mono

    tsumugi_notes = _extract_tsumugi_vocal_notes(
        Path(tsumugi_vocal_midi_path) if tsumugi_vocal_midi_path is not None else None
    )

    vocal_band_power, mix_band_power, vocal_pitch_power, frames_per_second = _compute_vocal_and_mix_acoustic_maps(
        vocal_mono,
        mix_mono,
        config=effective_config,
    )
    vocal_band_db = 10.0 * np.log10(np.maximum(vocal_band_power, 1e-12))
    mix_band_db = 10.0 * np.log10(np.maximum(mix_band_power, 1e-12))
    vocal_pitch_db = 10.0 * np.log10(np.maximum(vocal_pitch_power, 1e-12))

    # 1. インスト曲（歌唱なし）ガード:
    #    ボーカルステム全体の有音レベルと有音長を確認し、無音なら全ノートを棄却する。
    vocal_p95_db = float(np.percentile(vocal_band_db, 95))
    active_vocal_mask = (vocal_band_db >= float(effective_config.instrumental_song_p95_dbfs)) & (
        (vocal_band_db - mix_band_db) >= float(effective_config.minimum_vocal_to_mix_db)
    )
    active_vocal_seconds = float(active_vocal_mask.sum()) / frames_per_second
    is_instrumental_song = (
        vocal_p95_db < float(effective_config.instrumental_song_p95_dbfs)
        or active_vocal_seconds < float(effective_config.instrumental_song_min_active_seconds)
    )

    if is_instrumental_song:
        empty_midi = pretty_midi.PrettyMIDI()
        if output_midi_path is not None:
            destination = Path(output_midi_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            empty_midi.write(str(destination))
        return empty_midi, {
            "is_instrumental_song": True,
            "vocal_p95_dbfs": round(vocal_p95_db, 2),
            "active_vocal_seconds": round(active_vocal_seconds, 2),
            "sheetsage_raw_notes": len(sheetsage_prediction.notes),
            "sheetsage_gated_lead_notes": 0,
            "sheetsage_rejected_instrumental_notes": len(sheetsage_prediction.notes),
            "sheetsage_rescued_played_notes": 0,
            "tsumugi_raw_notes": len(tsumugi_notes),
            "tsumugi_lead_duplicates_dropped": 0,
            "tsumugi_harmony_promoted_from_melody": 0,
            "tsumugi_harmony_kept": 0,
            "tsumugi_uncovered_lead_filled": 0,
            "final_melody_notes": 0,
            "final_harmony_notes": 0,
        }

    # 有音ボーカル区間の基準音量（上位 90% 付近）を求め、曲の録音レベル差を吸収する。
    vocal_reference_db = float(np.percentile(vocal_band_db[active_vocal_mask], 90))
    minimum_active_vocal_db = max(
        float(effective_config.hard_silence_dbfs),
        vocal_reference_db - float(effective_config.max_drop_below_vocal_reference_db),
    )

    # 2. SheetSage2 ノートをボーカルステム音声でゲートし、楽器メロディを除去する。
    gated_lead_notes: list[SheetSageMelodyNote] = []
    rejected_instrumental_count = 0
    rescued_played_count = 0

    for candidate in sheetsage_prediction.notes:
        duration_seconds = float(candidate.end_seconds) - float(candidate.start_seconds)
        if duration_seconds < float(effective_config.minimum_note_duration_seconds):
            rejected_instrumental_count += 1
            continue

        vocal_level_db, vocal_to_mix_db, pitch_support_db = _segment_db_stats(
            candidate.start_seconds,
            candidate.end_seconds,
            candidate.pitch,
            vocal_band_db=vocal_band_db,
            mix_band_db=mix_band_db,
            vocal_pitch_db=vocal_pitch_db,
            frames_per_second=frames_per_second,
        )

        # ハード無音／楽器支配区間は track=0 でも必ず棄却する。
        if vocal_level_db < minimum_active_vocal_db or vocal_to_mix_db < float(effective_config.minimum_vocal_to_mix_db):
            rejected_instrumental_count += 1
            continue

        has_tsumugi_pitch_confirmation = any(
            abs(tsumugi_note.pitch - candidate.pitch) <= 1
            and _time_overlap_seconds(
                candidate.start_seconds,
                candidate.end_seconds,
                tsumugi_note.start_seconds,
                tsumugi_note.end_seconds,
            )
            >= 0.20 * min(duration_seconds, tsumugi_note.end_seconds - tsumugi_note.start_seconds)
            for tsumugi_note in tsumugi_notes
        )

        is_sung_by_sheetsage = (
            candidate.vocal_probability >= float(effective_config.sheetsage_sung_probability_threshold)
            or candidate.track == 0
        )
        if is_sung_by_sheetsage:
            # ピッチ倍音支持が弱く tsumugi 側にも一致がない純ノイズ／ブレスは弾く。
            if (
                pitch_support_db < float(effective_config.minimum_pitch_support_above_floor_db)
                and not has_tsumugi_pitch_confirmation
                and vocal_to_mix_db < -12.0
            ):
                rejected_instrumental_count += 1
                continue
            gated_lead_notes.append(candidate)
        else:
            # SheetSage2 が track=1 (played) と判定した場合でも、ボーカルステムが強発音かつ
            # tsumugi 側も同音高を検出していれば、エフェクト強めの歌声等として救済する。
            if (
                vocal_level_db >= vocal_reference_db - float(effective_config.played_rescue_max_drop_db)
                and vocal_to_mix_db >= float(effective_config.played_rescue_min_vocal_to_mix_db)
                and has_tsumugi_pitch_confirmation
            ):
                gated_lead_notes.append(candidate)
                rescued_played_count += 1
            else:
                rejected_instrumental_count += 1

    # 3. 確定主旋律アンカー (gated_lead_notes) を基準に、tsumugi 側全ノートを再仕分けする。
    final_harmony_notes: list[pretty_midi.Note] = []

    lead_duplicates_dropped = 0
    harmony_promoted_from_melody = 0
    harmony_kept_count = 0
    uncovered_lead_filled = 0
    matched_lead_start_diffs: list[float] = []
    matched_lead_end_diffs: list[float] = []
    # スウィング／ルバート復元用に、各 gated_lead_notes インデックスへ一致した tsumugi ノートを記録する。
    matched_tsumugi_by_lead_index: dict[int, tuple[float, _SourceVocalNote]] = {}
    uncovered_fill_notes: list[pretty_midi.Note] = []

    for tsumugi_note in tsumugi_notes:
        note_duration = tsumugi_note.end_seconds - tsumugi_note.start_seconds
        if note_duration < float(effective_config.minimum_note_duration_seconds):
            continue

        # 最も時間重複が大きい SheetSage2 主旋律アンカーを探す。
        best_overlap_ratio = 0.0
        best_overlapping_lead_index: int | None = None
        best_overlapping_lead: SheetSageMelodyNote | None = None
        nearest_lead_gap_seconds = float("inf")

        for lead_index, lead_note in enumerate(gated_lead_notes):
            overlap = _time_overlap_seconds(
                tsumugi_note.start_seconds,
                tsumugi_note.end_seconds,
                lead_note.start_seconds,
                lead_note.end_seconds,
            )
            if overlap > 0.0:
                nearest_lead_gap_seconds = 0.0
                ratio = overlap / max(min(note_duration, lead_note.end_seconds - lead_note.start_seconds), 1e-6)
                if ratio > best_overlap_ratio:
                    best_overlap_ratio = ratio
                    best_overlapping_lead_index = lead_index
                    best_overlapping_lead = lead_note
            else:
                gap = (
                    lead_note.start_seconds - tsumugi_note.end_seconds
                    if tsumugi_note.end_seconds <= lead_note.start_seconds
                    else tsumugi_note.start_seconds - lead_note.end_seconds
                )
                if gap < nearest_lead_gap_seconds:
                    nearest_lead_gap_seconds = gap

        # ケース A: SheetSage2 主旋律アンカーと時間重複しているノート。
        if best_overlapping_lead is not None and best_overlap_ratio >= float(effective_config.lead_match_overlap_ratio):
            semitone_gap = abs(tsumugi_note.pitch - best_overlapping_lead.pitch)

            # 同音高（または melody ラベルの 1 半音揺れ）は主旋律の重複とみなし破棄する。
            # これにより tsumugi が主旋律を誤って vocal_harmony へ入れた漏れを確実に除去できる。
            if semitone_gap == 0 or (semitone_gap == 1 and not tsumugi_note.is_harmony_label):
                if semitone_gap == 0 and best_overlap_ratio >= 0.50:
                    matched_lead_start_diffs.append(
                        float(best_overlapping_lead.start_seconds) - float(tsumugi_note.start_seconds)
                    )
                    matched_lead_end_diffs.append(
                        float(best_overlapping_lead.end_seconds) - float(tsumugi_note.end_seconds)
                    )
                    if best_overlapping_lead_index is not None:
                        previous_match = matched_tsumugi_by_lead_index.get(best_overlapping_lead_index)
                        if previous_match is None or best_overlap_ratio > previous_match[0]:
                            matched_tsumugi_by_lead_index[best_overlapping_lead_index] = (
                                best_overlap_ratio,
                                tsumugi_note,
                            )
                lead_duplicates_dropped += 1
                continue

            # 主旋律と同時に鳴っている別音高ノートはハモリ候補として評価する。
            if note_duration < float(effective_config.harmony_minimum_duration_seconds):
                continue
            if semitone_gap > int(effective_config.harmony_maximum_semitone_gap):
                continue
            if semitone_gap < int(effective_config.harmony_minimum_semitone_gap):
                # 1 半音衝突（短 2 度）はモデル間の音程ズレの可能性が高いためハモリへ入れない。
                continue
            if (
                semitone_gap == 12
                and not tsumugi_note.is_harmony_label
                and effective_config.suppress_melody_labeled_octave_ghosts
            ):
                # 単一歌唱で melody 側に出たオクターブ倍音ゴーストを除外する。
                continue

            # ボーカルステムに実際に該当ピッチのエネルギーがあるか軽く検証して誤検出を防ぐ。
            vocal_level_db, vocal_to_mix_db, pitch_support_db = _segment_db_stats(
                tsumugi_note.start_seconds,
                tsumugi_note.end_seconds,
                tsumugi_note.pitch,
                vocal_band_db=vocal_band_db,
                mix_band_db=mix_band_db,
                vocal_pitch_db=vocal_pitch_db,
                frames_per_second=frames_per_second,
            )
            if vocal_level_db < minimum_active_vocal_db or vocal_to_mix_db < float(
                effective_config.minimum_vocal_to_mix_db
            ):
                continue

            if not tsumugi_note.is_harmony_label:
                # melody ラベルのノートをハモリへ昇格する場合は、単独歌唱でのピッチ誤推定を除外するため
                # 「主旋律ピッチの倍音エネルギーと比べても十分な存在感があること」を確認する。
                _, _, lead_pitch_support_db = _segment_db_stats(
                    tsumugi_note.start_seconds,
                    tsumugi_note.end_seconds,
                    best_overlapping_lead.pitch,
                    vocal_band_db=vocal_band_db,
                    mix_band_db=mix_band_db,
                    vocal_pitch_db=vocal_pitch_db,
                    frames_per_second=frames_per_second,
                )
                if (
                    pitch_support_db < float(effective_config.minimum_pitch_support_above_floor_db)
                    or pitch_support_db
                    < lead_pitch_support_db - float(effective_config.harmony_promotion_max_drop_below_lead_pitch_db)
                ):
                    continue

            final_harmony_notes.append(
                pretty_midi.Note(
                    velocity=tsumugi_note.velocity,
                    pitch=tsumugi_note.pitch,
                    start=tsumugi_note.start_seconds,
                    end=tsumugi_note.end_seconds,
                )
            )
            if not tsumugi_note.is_harmony_label:
                harmony_promoted_from_melody += 1
            else:
                harmony_kept_count += 1
            continue

        # ケース B: SheetSage2 主旋律が休符の区間。
        vocal_level_db, vocal_to_mix_db, pitch_support_db = _segment_db_stats(
            tsumugi_note.start_seconds,
            tsumugi_note.end_seconds,
            tsumugi_note.pitch,
            vocal_band_db=vocal_band_db,
            mix_band_db=mix_band_db,
            vocal_pitch_db=vocal_pitch_db,
            frames_per_second=frames_per_second,
        )
        if tsumugi_note.is_harmony_label:
            # 追っかけ・コーラス応答・ハモリの余韻など。無音ノイズだけ弾いてハモリとして保持する。
            if (
                note_duration >= float(effective_config.harmony_minimum_duration_seconds)
                and vocal_level_db >= minimum_active_vocal_db
                and vocal_to_mix_db >= float(effective_config.minimum_vocal_to_mix_db)
            ):
                final_harmony_notes.append(
                    pretty_midi.Note(
                        velocity=tsumugi_note.velocity,
                        pitch=tsumugi_note.pitch,
                        start=tsumugi_note.start_seconds,
                        end=tsumugi_note.end_seconds,
                    )
                )
                harmony_kept_count += 1
        else:
            # SheetSage2 ノート端の微小ズレ（二重発音の原因）を除外し、
            # 明確に独立した強歌唱フレーズだけを主旋律の取りこぼし補完として採用する。
            if (
                nearest_lead_gap_seconds >= float(effective_config.fill_uncovered_lead_gap_margin_seconds)
                and note_duration >= float(effective_config.fill_uncovered_lead_min_duration_seconds)
                and vocal_level_db >= vocal_reference_db - float(effective_config.fill_uncovered_lead_max_drop_db)
                and vocal_to_mix_db >= float(effective_config.fill_uncovered_lead_min_vocal_to_mix_db)
                and pitch_support_db >= float(effective_config.minimum_pitch_support_above_floor_db)
            ):
                uncovered_fill_notes.append(
                    pretty_midi.Note(
                        velocity=tsumugi_note.velocity,
                        pitch=tsumugi_note.pitch,
                        start=tsumugi_note.start_seconds,
                        end=tsumugi_note.end_seconds,
                    )
                )
                uncovered_lead_filled += 1

    # 4. 一致主旋律ペアからモデル間の系統的時間オフセットを推定する。
    max_bias = float(effective_config.max_global_timing_bias_seconds)
    global_start_bias_seconds = (
        float(np.clip(np.median(matched_lead_start_diffs), -max_bias, max_bias))
        if len(matched_lead_start_diffs) >= 5
        else 0.0
    )
    global_end_bias_seconds = (
        float(np.clip(np.median(matched_lead_end_diffs), -max_bias, max_bias))
        if len(matched_lead_end_diffs) >= 5
        else 0.0
    )

    # 4.5 音声オンセット検証によるスウィング／3 連符／ルバート復元:
    #     SheetSage2 の 16 分量子化バイアスによりオンセットが実音声から 25〜135ms ズレたノートについて、
    #     分離ボーカル音声が明確に tsumugi 側で立ち上がっている場合のみ生オンセットを採用する。
    swing_refined_lead_count = 0
    refined_lead_anchors: list[SheetSageMelodyNote] = []
    for lead_index, lead_note in enumerate(gated_lead_notes):
        new_start = float(lead_note.start_seconds)
        new_end = float(lead_note.end_seconds)
        if effective_config.refine_lead_timing_for_swing and lead_index in matched_tsumugi_by_lead_index:
            _, matched_tsumugi = matched_tsumugi_by_lead_index[lead_index]
            candidate_start = max(0.0, float(matched_tsumugi.start_seconds) + global_start_bias_seconds)
            timing_diff = abs(candidate_start - new_start)
            if (
                float(effective_config.swing_refine_min_diff_seconds)
                <= timing_diff
                <= float(effective_config.swing_refine_max_diff_seconds)
                and (new_end - candidate_start) >= float(effective_config.minimum_note_duration_seconds)
            ):
                sheetsage_onset_jump = _measure_vocal_onset_jump_db(
                    new_start,
                    lead_note.pitch,
                    vocal_band_db=vocal_band_db,
                    vocal_pitch_db=vocal_pitch_db,
                    frames_per_second=frames_per_second,
                )
                tsumugi_onset_jump = _measure_vocal_onset_jump_db(
                    candidate_start,
                    lead_note.pitch,
                    vocal_band_db=vocal_band_db,
                    vocal_pitch_db=vocal_pitch_db,
                    frames_per_second=frames_per_second,
                )
                if (
                    tsumugi_onset_jump >= sheetsage_onset_jump + float(effective_config.swing_refine_min_onset_gain_db)
                    and tsumugi_onset_jump
                    >= max(sheetsage_onset_jump, 1.0) * float(effective_config.swing_refine_min_onset_ratio)
                ):
                    # 前の音とレガートで繋がっていた場合は前音の終端も追従させ、隙間や重複を残さない。
                    if refined_lead_anchors and abs(refined_lead_anchors[-1].end_seconds - new_start) <= 0.02:
                        previous_anchor = refined_lead_anchors[-1]
                        if candidate_start > previous_anchor.start_seconds + 0.03:
                            refined_lead_anchors[-1] = SheetSageMelodyNote(
                                start_seconds=previous_anchor.start_seconds,
                                end_seconds=candidate_start,
                                pitch=previous_anchor.pitch,
                                track=previous_anchor.track,
                                vocal_probability=previous_anchor.vocal_probability,
                                steps=previous_anchor.steps,
                                position_in_beat=previous_anchor.position_in_beat,
                            )
                    new_start = candidate_start
                    swing_refined_lead_count += 1

        refined_lead_anchors.append(
            SheetSageMelodyNote(
                start_seconds=new_start,
                end_seconds=max(new_start + 0.03, new_end),
                pitch=lead_note.pitch,
                track=lead_note.track,
                vocal_probability=lead_note.vocal_probability,
                steps=lead_note.steps,
                position_in_beat=lead_note.position_in_beat,
            )
        )

    final_melody_notes: list[pretty_midi.Note] = [
        pretty_midi.Note(
            velocity=int(effective_config.default_velocity),
            pitch=int(lead_note.pitch),
            start=float(lead_note.start_seconds),
            end=float(lead_note.end_seconds),
        )
        for lead_note in refined_lead_anchors
    ]

    for fill_note in uncovered_fill_notes:
        shifted_start = max(0.0, float(fill_note.start) + global_start_bias_seconds)
        shifted_end = max(shifted_start + 0.03, float(fill_note.end) + global_end_bias_seconds)
        final_melody_notes.append(
            pretty_midi.Note(
                velocity=fill_note.velocity,
                pitch=fill_note.pitch,
                start=shifted_start,
                end=shifted_end,
            )
        )

    harmony_snapped_onsets = 0
    harmony_snapped_offsets = 0
    if effective_config.align_harmony_to_lead and final_harmony_notes and refined_lead_anchors:
        final_harmony_notes, harmony_snapped_onsets, harmony_snapped_offsets = _align_harmony_notes_to_lead_rhythm(
            final_harmony_notes,
            refined_lead_anchors,
            start_bias_seconds=global_start_bias_seconds,
            end_bias_seconds=global_end_bias_seconds,
            config=effective_config,
        )

    cleaned_melody_notes = _deduplicate_monophonic_notes(final_melody_notes)
    cleaned_harmony_notes = _deduplicate_same_pitch_harmony_notes(final_harmony_notes)

    fused_midi = pretty_midi.PrettyMIDI()
    if cleaned_melody_notes:
        melody_instrument = pretty_midi.Instrument(
            program=int(effective_config.melody_program),
            is_drum=False,
            name="melody",
        )
        melody_instrument.notes = cleaned_melody_notes
        fused_midi.instruments.append(melody_instrument)
    if cleaned_harmony_notes:
        harmony_instrument = pretty_midi.Instrument(
            program=int(effective_config.harmony_program),
            is_drum=False,
            name="vocal_harmony",
        )
        harmony_instrument.notes = cleaned_harmony_notes
        fused_midi.instruments.append(harmony_instrument)

    if output_midi_path is not None:
        destination = Path(output_midi_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fused_midi.write(str(destination))

    report = {
        "is_instrumental_song": False,
        "vocal_p95_dbfs": round(vocal_p95_db, 2),
        "vocal_reference_dbfs": round(vocal_reference_db, 2),
        "active_vocal_seconds": round(active_vocal_seconds, 2),
        "sheetsage_raw_notes": len(sheetsage_prediction.notes),
        "sheetsage_gated_lead_notes": len(gated_lead_notes),
        "sheetsage_rejected_instrumental_notes": rejected_instrumental_count,
        "sheetsage_rescued_played_notes": rescued_played_count,
        "tsumugi_raw_notes": len(tsumugi_notes),
        "tsumugi_lead_duplicates_dropped": lead_duplicates_dropped,
        "tsumugi_harmony_promoted_from_melody": harmony_promoted_from_melody,
        "tsumugi_harmony_kept": harmony_kept_count,
        "tsumugi_uncovered_lead_filled": uncovered_lead_filled,
        "lead_swing_onsets_refined": swing_refined_lead_count,
        "harmony_global_start_bias_ms": round(global_start_bias_seconds * 1000.0, 2),
        "harmony_snapped_onsets": harmony_snapped_onsets,
        "harmony_snapped_offsets": harmony_snapped_offsets,
        "final_melody_notes": len(cleaned_melody_notes),
        "final_harmony_notes": len(cleaned_harmony_notes),
    }
    return fused_midi, report
