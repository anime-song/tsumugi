"""元ステム音声と FluidSynth Render & Compare による過剰分割ノート結合（Note De-fragmentation）。

オーケストラ楽曲のストリングス・管楽器・コーラスや、ギター・ピアノのロングトーンでは、
ビブラート・弓の返し・残響・分離ノイズによって 1 本の持続音が複数の短いノートに
過剰分割されることがある。

本モジュールでは以下の 3 段階で「本物の同音連打を守りながら過剰分割だけを結合」する。
1. 候補境界の抽出（楽器種別ごとのギャップ上限＋対象ピッチ倍音包絡線での事前ガード）:
   ドラム以外の各トラックで同一ピッチの連続ノートを走査し、明らかに強い再アタックや
   深い振幅の谷がある本物の弾き直しを除外したうえで結合候補境界を列挙する。
2. 2 スロット・パック MIDI × FluidSynth 1 回起動レンダリング:
   各ステムについて「分割のまま (split)」と「全候補結合 (merged)」の 2 スロットだけを
   1 本の直列パック MIDI に詰め、FluidSynth を 1 回だけ起動してレンダリングする。
3. 境界単位の Render & Compare 判定（対象ピッチ倍音 L1 包絡線＋再アタックフラックス比較）:
   各候補境界の前後 ±120 ms において、対象ピッチの 1〜5 次倍音帯域の対数エネルギー包絡線と
   再アタックフラックスを元ステム音声と比較し、merged の方が明確に近い境界だけを結合する。
   コサイン類似度ではなく中心化 dB-L1 距離を使うため、フラットな持続音でも分母ゼロによる
   不安定化が起きない。
"""

from __future__ import annotations

import copy
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

from ..velocity.expression_pan import is_sustained_gm_program
from ..velocity.render_compare_velocity import _merge_optimized_stem_midis
from .render_compare_quantize import (
    RenderCompareQuantizeConfig,
    _load_mono_audio,
    _render_packed_midi_with_fluidsynth,
    _write_packed_variants_midi,
)

_PackedMidiRenderer = Callable[[Path, Path, int], None]


@dataclass(frozen=True)
class RenderCompareDefragmentConfig:
    """Render & Compare 型 過剰分割ノート結合の設定。"""

    soundfont_path: Path
    fluidsynth_executable: Path | None = None
    sample_rate: int = 16_000
    slot_gap_seconds: float = 0.6
    render_gain: float = 0.5
    disable_reverb: bool = True
    disable_chorus: bool = True
    # 既定ではストリングス・管楽器・オルガン等の持続系 GM 楽器のみを対象とする。
    # 歪みギター・アコギ・ピアノ等の刻み連打（コンプや弦共鳴で包絡線が平坦になりやすい）を誤結合しないため。
    sustained_only: bool = True
    # 楽器種別ごとの結合候補ギャップ上限（秒）。
    sustained_max_gap_seconds: float = 0.130
    plucked_max_gap_seconds: float = 0.075
    # 結合後の最大ノート長（秒）。延々と繋がり続ける暴走を防ぐ。
    max_merged_duration_seconds: float = 10.0
    # 事前音響ガード：元音声の対象ピッチ倍音帯域でこれ以上の明確な再アタック／谷がある境界は除外する。
    sustained_max_reattack_rise_db: float = 9.0
    sustained_max_valley_dip_db: float = 6.5
    sustained_max_net_jump_db: float = 7.5
    plucked_max_reattack_rise_db: float = 5.5
    plucked_max_valley_dip_db: float = 5.0
    plucked_max_net_jump_db: float = 4.0
    # 境界前後で Render & Compare 距離を評価する半窓長（秒）。
    boundary_half_window_seconds: float = 0.120
    # merged 距離が split 距離よりこの dB 以上改善した境界を採用する。
    sustained_min_improvement_db: float = 0.15
    plucked_min_improvement_db: float = 0.35
    # 持続系楽器で 5ms〜35ms の密着ブツ切りかつ元音声の対象ピッチ倍音が完全に滑らかな箇所は、
    # スローアタック SF2 の持続音量差や LFO で Render & Compare が負値になっても確実に結合する。
    smooth_sustain_max_gap_seconds: float = 0.035
    smooth_sustain_max_reattack_rise_db: float = 3.5
    smooth_sustain_max_valley_dip_db: float = 2.5
    smooth_sustain_max_net_jump_db: float = 2.5
    smooth_sustain_max_rc_regression_db: float = 0.25
    # 極短ゴミ断片（リリース残響の誤検出等）が隣接ノートへ密着している場合の閾値。
    tiny_fragment_duration_seconds: float = 0.045
    tiny_fragment_max_gap_seconds: float = 0.025
    flux_distance_weight: float = 1.6
    max_melodic_instruments: int = 15

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.sustained_max_gap_seconds < 0.0 or self.plucked_max_gap_seconds < 0.0:
            raise ValueError("max_gap_seconds must be nonnegative")
        if self.boundary_half_window_seconds <= 0.0:
            raise ValueError("boundary_half_window_seconds must be positive")


@dataclass(frozen=True)
class _BoundaryCandidate:
    """同一トラック・同一ピッチの連続ノート間に検出された結合候補境界。"""

    stem_name: str
    instrument_index: int
    pitch: int
    left_index_in_pitch: int
    split_time_seconds: float
    gap_seconds: float
    left_duration_seconds: float
    right_duration_seconds: float
    is_sustained: bool
    is_overlapping_or_contained: bool
    reattack_rise_db: float
    valley_dip_db: float
    net_jump_db: float


def _build_pitch_harmonic_power_filterbank(
    *,
    sample_rate: int,
    n_fft: int = 1024,
    pitch_min: int = 21,
    pitch_max: int = 108,
    max_harmonics: int = 5,
) -> np.ndarray:
    """MIDI ピッチ [pitch_min..pitch_max] ごとの 1〜max_harmonics 次倍音フィルタバンク [P, F] を生成する。"""
    num_bins = n_fft // 2 + 1
    frequencies = np.linspace(0.0, float(sample_rate) * 0.5, num_bins, dtype=np.float64)
    nyquist = float(sample_rate) * 0.48
    num_pitches = pitch_max - pitch_min + 1
    filterbank = np.zeros((num_pitches, num_bins), dtype=np.float32)

    for pitch in range(pitch_min, pitch_max + 1):
        fundamental = 440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0))
        weights = np.zeros(num_bins, dtype=np.float64)
        for harmonic in range(1, max_harmonics + 1):
            center_hz = fundamental * float(harmonic)
            if center_hz >= nyquist:
                break
            # ±0.45 半音幅のガウス窓で基本波・倍音周辺のエネルギーだけを取り出す（隣接半音への漏れを抑制）。
            bandwidth_hz = max(14.0, center_hz * (2.0 ** (0.45 / 12.0) - 1.0))
            harmonic_weight = 1.0 / float(harmonic)
            weights += harmonic_weight * np.exp(-0.5 * np.square((frequencies - center_hz) / bandwidth_hz))
        weight_sum = float(weights.sum())
        if weight_sum > 0.0:
            weights /= weight_sum
        filterbank[pitch - pitch_min] = weights.astype(np.float32)
    return filterbank


def _compute_pitch_envelope_db(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    pitch_filterbank: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 128,
) -> tuple[np.ndarray, float]:
    """モノラル波形 [T] から各 MIDI ピッチの倍音帯域対数パワー包絡線 [88, F] (dB) を計算する。

    16 kHz / hop=128 のとき 1 フレーム = 8.0 ms。
    """
    if waveform.size < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.size))
    window = torch.hann_window(n_fft, dtype=torch.float32)
    window_energy = float(window.square().sum().item())
    spectrum = torch.stft(
        torch.from_numpy(waveform.astype(np.float32, copy=False)),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power = (spectrum.real.square() + spectrum.imag.square()).numpy() / max(window_energy, 1e-8)
    pitch_power = pitch_filterbank @ power
    pitch_db = (10.0 * np.log10(np.maximum(pitch_power, 1e-10))).astype(np.float32, copy=False)
    frames_per_second = float(sample_rate) / float(hop_length)
    return pitch_db, frames_per_second


def _measure_boundary_acoustic_cue(
    pitch_envelope_db: np.ndarray,
    *,
    frames_per_second: float,
    left_start_seconds: float,
    left_end_seconds: float,
    right_start_seconds: float,
    right_end_seconds: float,
) -> tuple[float, float, float]:
    """境界時刻における対象ピッチの (再アタック上昇 dB, 谷ディップ dB, 前後純上昇 dB) を計測する。"""
    num_frames = int(pitch_envelope_db.size)
    if num_frames < 6:
        return 0.0, 0.0, 0.0

    def _slice(start_seconds: float, end_seconds: float) -> np.ndarray:
        left_frame = max(0, min(num_frames - 1, int(math.floor(start_seconds * frames_per_second))))
        right_frame = min(num_frames, max(left_frame + 1, int(math.ceil(end_seconds * frames_per_second))))
        return pitch_envelope_db[left_frame:right_frame]

    before_start = max(left_start_seconds + 0.015, right_start_seconds - 0.130)
    before_end = max(before_start + 0.016, right_start_seconds - 0.025)
    valley_start = max(left_start_seconds, min(left_end_seconds, right_start_seconds) - 0.045)
    valley_end = min(right_end_seconds, right_start_seconds + 0.015)
    after_start = right_start_seconds + 0.008
    after_end = min(right_end_seconds, right_start_seconds + 0.095)
    if after_end <= after_start + 0.012:
        after_end = right_start_seconds + 0.040

    before_level = float(np.median(_slice(before_start, before_end)))
    valley_level = float(np.min(_slice(valley_start, valley_end)))
    after_level = float(np.percentile(_slice(after_start, after_end), 85))

    valley_dip_db = max(0.0, before_level - valley_level)
    reattack_rise_db = max(0.0, after_level - valley_level)
    net_jump_db = after_level - before_level
    return reattack_rise_db, valley_dip_db, net_jump_db


def _compute_boundary_envelope_distance_db(
    target_pitch_db: np.ndarray,
    rendered_pitch_db: np.ndarray,
    *,
    split_time_seconds: float,
    frames_per_second: float,
    half_window_seconds: float,
    flux_distance_weight: float,
) -> float:
    """境界前後 ±half_window_seconds における対象ピッチ包絡線の中心化 dB-L1 ＋再アタックフラックス距離を返す。"""
    num_frames = min(int(target_pitch_db.size), int(rendered_pitch_db.size))
    start_frame = max(0, int(math.floor((split_time_seconds - half_window_seconds) * frames_per_second)))
    end_frame = min(num_frames, int(math.ceil((split_time_seconds + half_window_seconds) * frames_per_second)))
    if end_frame - start_frame < 4:
        return 0.0

    target_slice = target_pitch_db[start_frame:end_frame].astype(np.float64)
    rendered_slice = rendered_pitch_db[start_frame:end_frame].astype(np.float64)

    # 音量差を消すため局所窓の平均を引いて、包絡線の「凹み・再上昇の形」だけを比較する。
    target_centered = target_slice - float(target_slice.mean())
    rendered_centered = rendered_slice - float(rendered_slice.mean())
    shape_l1_db = float(np.mean(np.abs(target_centered - rendered_centered)))

    # 境界中央付近の再アタック立ち上がり（正の一次差分）の一致度を重視して加算する。
    target_flux = np.maximum(0.0, np.diff(target_slice))
    rendered_flux = np.maximum(0.0, np.diff(rendered_slice))
    flux_l1_db = float(np.mean(np.abs(target_flux - rendered_flux)))

    return shape_l1_db + float(flux_distance_weight) * flux_l1_db


def _apply_boundary_merges_to_midi(
    source_midi: pretty_midi.PrettyMIDI,
    accepted_boundaries_by_track_pitch: Mapping[tuple[int, int], set[int]],
    *,
    max_merged_duration_seconds: float,
) -> tuple[pretty_midi.PrettyMIDI, int]:
    """指定された (instrument_index, pitch) -> 結合する境界インデックス集合 に従ってノートを結合する。

    戻り値は (結合済み PrettyMIDI, 実際に結合削減されたノート数)。
    """
    merged_midi = copy.deepcopy(source_midi)
    merged_count = 0

    for instrument_index, instrument in enumerate(merged_midi.instruments):
        if instrument.is_drum or not instrument.notes:
            continue
        notes_by_pitch: dict[int, list[pretty_midi.Note]] = {}
        for note in instrument.notes:
            notes_by_pitch.setdefault(int(note.pitch), []).append(note)

        rebuilt_notes: list[pretty_midi.Note] = []
        for pitch, pitch_notes in notes_by_pitch.items():
            pitch_notes.sort(key=lambda item: (item.start, item.end))
            accepted_indices = accepted_boundaries_by_track_pitch.get((instrument_index, pitch), set())
            if not accepted_indices:
                rebuilt_notes.extend(pitch_notes)
                continue

            current_note = copy.deepcopy(pitch_notes[0])
            for left_index in range(len(pitch_notes) - 1):
                next_note = pitch_notes[left_index + 1]
                candidate_end = max(float(current_note.end), float(next_note.end))
                within_max_duration = (candidate_end - float(current_note.start)) <= float(max_merged_duration_seconds)
                if left_index in accepted_indices and within_max_duration:
                    # 過剰分割の後ろ断片は弱く出がちなので、先頭主音の Velocity を基準に保ちつつ終端を延ばす。
                    current_note.end = candidate_end
                    current_note.velocity = max(int(current_note.velocity), int(round(0.85 * float(next_note.velocity))))
                    merged_count += 1
                else:
                    rebuilt_notes.append(current_note)
                    current_note = copy.deepcopy(next_note)
            rebuilt_notes.append(current_note)

        rebuilt_notes.sort(key=lambda note: (note.start, note.pitch, note.end))
        instrument.notes = rebuilt_notes

    return merged_midi, merged_count


def defragment_stems_by_render_compare(
    stem_midis: Mapping[str, Path | str],
    stem_audios: Mapping[str, Path | str],
    *,
    config: RenderCompareDefragmentConfig,
    output_midi_path: Path | str | None = None,
    output_stem_dir: Path | str | None = None,
    packed_midi_renderer: _PackedMidiRenderer | None = None,
) -> dict[str, Any]:
    """全ステム MIDI の過剰分割ノートを Render & Compare（2 スロット 1 回起動）で検証・結合する。"""
    sample_rate = int(config.sample_rate)
    resolved_midis = {str(name).lower(): Path(path) for name, path in stem_midis.items() if Path(path).is_file()}
    resolved_audios = {str(name).lower(): Path(path) for name, path in stem_audios.items() if Path(path).is_file()}
    active_stems = sorted(set(resolved_midis) & set(resolved_audios))
    if not active_stems:
        raise ValueError("No matching stem MIDI and audio pairs were found.")

    # 1. 元ステム音声・MIDI を読み込み、重複トラックを統合する。
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

        mono_audio = _load_mono_audio(resolved_audios[stem_name], target_sample_rate=sample_rate)
        original_midis[stem_name] = midi_object
        target_waveforms[stem_name] = mono_audio
        stem_durations[stem_name] = max(0.25, float(mono_audio.size) / float(sample_rate))

    stems_with_notes = sorted(original_midis)
    if not stems_with_notes:
        raise ValueError("No notes found in matching stem MIDI files.")

    # 2. 各ステムの対象ピッチ倍音包絡線を計算し、事前音響ガード付きで結合候補境界を抽出する。
    pitch_min = 21
    pitch_max = 108
    pitch_filterbank = _build_pitch_harmonic_power_filterbank(
        sample_rate=sample_rate,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
    )
    target_pitch_db_by_stem: dict[str, tuple[np.ndarray, float]] = {}
    candidates_by_stem: dict[str, list[_BoundaryCandidate]] = {}
    proposed_merged_midis: dict[str, pretty_midi.PrettyMIDI] = {}

    for stem_name in stems_with_notes:
        target_pitch_db, frames_per_second = _compute_pitch_envelope_db(
            target_waveforms[stem_name],
            sample_rate=sample_rate,
            pitch_filterbank=pitch_filterbank,
        )
        target_pitch_db_by_stem[stem_name] = (target_pitch_db, frames_per_second)

        stem_candidates: list[_BoundaryCandidate] = []
        proposed_indices_by_track_pitch: dict[tuple[int, int], set[int]] = {}
        is_vocal_stem = "vocal" in stem_name

        for instrument_index, instrument in enumerate(original_midis[stem_name].instruments):
            if instrument.is_drum or not instrument.notes:
                continue
            is_sustained = is_sustained_gm_program(int(instrument.program), is_drum=False) and not is_vocal_stem
            if config.sustained_only and not is_sustained:
                continue
            max_gap = float(
                config.sustained_max_gap_seconds if is_sustained else config.plucked_max_gap_seconds
            )
            max_reattack = float(
                config.sustained_max_reattack_rise_db if is_sustained else config.plucked_max_reattack_rise_db
            )
            max_valley = float(
                config.sustained_max_valley_dip_db if is_sustained else config.plucked_max_valley_dip_db
            )
            max_net_jump = float(
                config.sustained_max_net_jump_db if is_sustained else config.plucked_max_net_jump_db
            )

            notes_by_pitch: dict[int, list[pretty_midi.Note]] = {}
            for note in instrument.notes:
                notes_by_pitch.setdefault(int(note.pitch), []).append(note)

            for pitch, pitch_notes in notes_by_pitch.items():
                if len(pitch_notes) < 2:
                    continue
                pitch_notes.sort(key=lambda item: (item.start, item.end))
                clamped_pitch = max(pitch_min, min(pitch_max, int(pitch)))
                pitch_curve = target_pitch_db[clamped_pitch - pitch_min]

                for left_index in range(len(pitch_notes) - 1):
                    left_note = pitch_notes[left_index]
                    right_note = pitch_notes[left_index + 1]
                    gap_seconds = float(right_note.start) - float(left_note.end)
                    if gap_seconds > max_gap:
                        continue
                    reattack_rise_db, valley_dip_db, net_jump_db = _measure_boundary_acoustic_cue(
                        pitch_curve,
                        frames_per_second=frames_per_second,
                        left_start_seconds=float(left_note.start),
                        left_end_seconds=float(left_note.end),
                        right_start_seconds=float(right_note.start),
                        right_end_seconds=float(right_note.end),
                    )
                    # 明確な再アタックや深い谷ディップがある本物の同音連打は候補から除外する。
                    if is_sustained:
                        is_clear_rearticulation = (
                            (reattack_rise_db >= max_reattack and valley_dip_db >= max_valley)
                            or net_jump_db >= max_net_jump
                        )
                    else:
                        is_clear_rearticulation = (
                            reattack_rise_db >= max_reattack
                            or valley_dip_db >= max_valley
                            or net_jump_db >= max_net_jump
                        )
                    left_duration = float(left_note.end) - float(left_note.start)
                    right_duration = float(right_note.end) - float(right_note.start)
                    is_overlapping_or_contained = (
                        gap_seconds <= -0.015
                        or float(right_note.end) <= float(left_note.end) + 0.005
                    )
                    # 重複／包含ノートは明確な再アタックが無い限り必ず統合候補にする。
                    if is_clear_rearticulation and not (is_overlapping_or_contained and net_jump_db < max_net_jump):
                        continue

                    stem_candidates.append(
                        _BoundaryCandidate(
                            stem_name=stem_name,
                            instrument_index=instrument_index,
                            pitch=pitch,
                            left_index_in_pitch=left_index,
                            split_time_seconds=float(right_note.start),
                            gap_seconds=gap_seconds,
                            left_duration_seconds=left_duration,
                            right_duration_seconds=right_duration,
                            is_sustained=is_sustained,
                            is_overlapping_or_contained=is_overlapping_or_contained,
                            reattack_rise_db=reattack_rise_db,
                            valley_dip_db=valley_dip_db,
                            net_jump_db=net_jump_db,
                        )
                    )
                    proposed_indices_by_track_pitch.setdefault((instrument_index, pitch), set()).add(left_index)

        candidates_by_stem[stem_name] = stem_candidates
        if stem_candidates:
            proposed_midi, _ = _apply_boundary_merges_to_midi(
                original_midis[stem_name],
                proposed_indices_by_track_pitch,
                max_merged_duration_seconds=float(config.max_merged_duration_seconds),
            )
            proposed_merged_midis[stem_name] = proposed_midi
        else:
            proposed_merged_midis[stem_name] = copy.deepcopy(original_midis[stem_name])

    # 3. 候補が 1 つでもあるステムだけ 2 スロット (split vs merged) を 1 パック MIDI でレンダリングする。
    stems_needing_render = [stem_name for stem_name in stems_with_notes if candidates_by_stem[stem_name]]
    final_midis: dict[str, pretty_midi.PrettyMIDI] = {
        stem_name: copy.deepcopy(original_midis[stem_name]) for stem_name in stems_with_notes
    }
    stem_reports: dict[str, dict[str, Any]] = {}
    track_reports: list[dict[str, Any]] = []

    rendered_mono = np.zeros(0, dtype=np.float32)
    slots_by_key: dict[str, Any] = {}
    if stems_needing_render:
        slot_specs = []
        for stem_name in stems_needing_render:
            duration_seconds = stem_durations[stem_name]
            slot_specs.append((f"{stem_name}::split", stem_name, "split", original_midis[stem_name].instruments, duration_seconds))
            slot_specs.append((f"{stem_name}::merged", stem_name, "merged", proposed_merged_midis[stem_name].instruments, duration_seconds))

        render_config = RenderCompareQuantizeConfig(
            soundfont_path=config.soundfont_path,
            fluidsynth_executable=config.fluidsynth_executable,
            sample_rate=sample_rate,
            slot_gap_seconds=float(config.slot_gap_seconds),
            render_gain=float(config.render_gain),
            disable_reverb=bool(config.disable_reverb),
            disable_chorus=bool(config.disable_chorus),
        )
        with tempfile.TemporaryDirectory(prefix="tsumugi_rc_defragment_") as temp_directory:
            temp_root = Path(temp_directory)
            packed_midi_path = temp_root / "defragment_packed.mid"
            packed_wav_path = temp_root / "defragment_packed.wav"
            slots = _write_packed_variants_midi(
                slot_specs,
                output_midi_path=packed_midi_path,
                slot_gap_seconds=float(config.slot_gap_seconds),
            )
            if packed_midi_renderer is not None:
                packed_midi_renderer(packed_midi_path, packed_wav_path, sample_rate)
            else:
                _render_packed_midi_with_fluidsynth(
                    packed_midi_path,
                    packed_wav_path,
                    sample_rate,
                    config=render_config,
                )
            rendered_audio, rendered_rate = sf.read(str(packed_wav_path), dtype="float32", always_2d=True)
            if int(rendered_rate) != sample_rate:
                raise ValueError(f"Rendered sample rate mismatch: expected {sample_rate}, got {rendered_rate}")
            rendered_mono = rendered_audio.mean(axis=1).astype(np.float32, copy=False)
            slots_by_key = {slot.slot_key: slot for slot in slots}

    # 4. 境界ごとに split vs merged の対象ピッチ局所包絡線距離を比較し、改善する境界だけを確定結合する。
    for stem_name in stems_with_notes:
        stem_candidates = candidates_by_stem[stem_name]
        before_note_count = sum(len(instrument.notes) for instrument in original_midis[stem_name].instruments)
        if not stem_candidates:
            stem_reports[stem_name] = {
                "before_note_count": int(before_note_count),
                "after_note_count": int(before_note_count),
                "candidate_boundaries": 0,
                "merged_boundaries": 0,
                "mean_improvement_db": 0.0,
            }
            continue

        target_pitch_db, frames_per_second = target_pitch_db_by_stem[stem_name]
        expected_samples = int(target_waveforms[stem_name].size)

        def _extract_slot_pitch_db(slot_key: str) -> np.ndarray:
            slot = slots_by_key[slot_key]
            start_sample = max(0, int(round(slot.slot_start_seconds * sample_rate)))
            segment = rendered_mono[start_sample : min(rendered_mono.size, start_sample + expected_samples)]
            if segment.size < expected_samples:
                segment = np.pad(segment, (0, expected_samples - segment.size))
            pitch_db, _ = _compute_pitch_envelope_db(
                segment,
                sample_rate=sample_rate,
                pitch_filterbank=pitch_filterbank,
            )
            return pitch_db

        split_pitch_db = _extract_slot_pitch_db(f"{stem_name}::split")
        merged_pitch_db = _extract_slot_pitch_db(f"{stem_name}::merged")

        accepted_by_track_pitch: dict[tuple[int, int], set[int]] = {}
        accepted_improvements: list[float] = []
        accepted_per_instrument: dict[int, int] = {}
        candidate_per_instrument: dict[int, int] = {}

        for candidate in stem_candidates:
            candidate_per_instrument[candidate.instrument_index] = (
                candidate_per_instrument.get(candidate.instrument_index, 0) + 1
            )
            clamped_pitch = max(pitch_min, min(pitch_max, int(candidate.pitch)))
            pitch_row = clamped_pitch - pitch_min
            split_distance = _compute_boundary_envelope_distance_db(
                target_pitch_db[pitch_row],
                split_pitch_db[pitch_row],
                split_time_seconds=candidate.split_time_seconds,
                frames_per_second=frames_per_second,
                half_window_seconds=float(config.boundary_half_window_seconds),
                flux_distance_weight=float(config.flux_distance_weight),
            )
            merged_distance = _compute_boundary_envelope_distance_db(
                target_pitch_db[pitch_row],
                merged_pitch_db[pitch_row],
                split_time_seconds=candidate.split_time_seconds,
                frames_per_second=frames_per_second,
                half_window_seconds=float(config.boundary_half_window_seconds),
                flux_distance_weight=float(config.flux_distance_weight),
            )
            improvement_db = split_distance - merged_distance
            min_required = float(
                config.sustained_min_improvement_db
                if candidate.is_sustained
                else config.plucked_min_improvement_db
            )
            # 1. Render & Compare 単独で明確に元音声へ近づいた境界。
            accept_by_render_compare = improvement_db >= min_required
            # 2. 持続系楽器で密着ブツ切り (<=35ms) かつ元音声の対象ピッチ倍音が完全に滑らか
            #    （谷 <2.5 dB かつ再アタック <3.5 dB かつ純上昇 <2.5 dB）な箇所は、
            #    SF2 のスローアタック音量差・LFO の影響を受けずに確実に結合する。
            is_smooth_sustained_seam = (
                candidate.is_sustained
                and candidate.gap_seconds <= float(config.smooth_sustain_max_gap_seconds)
                and candidate.reattack_rise_db < float(config.smooth_sustain_max_reattack_rise_db)
                and candidate.valley_dip_db < float(config.smooth_sustain_max_valley_dip_db)
                and candidate.net_jump_db < float(config.smooth_sustain_max_net_jump_db)
            )
            # 3. 同一トラック・同一ピッチで重複／包含されたノート、または持続音の端に付いた極短ゴミ断片。
            is_duplicate_or_tiny_fragment = candidate.is_overlapping_or_contained or (
                candidate.is_sustained
                and candidate.gap_seconds <= float(config.tiny_fragment_max_gap_seconds)
                and min(candidate.left_duration_seconds, candidate.right_duration_seconds)
                <= float(config.tiny_fragment_duration_seconds)
                and candidate.reattack_rise_db < float(config.sustained_max_reattack_rise_db) * 0.6
                and improvement_db >= -float(config.smooth_sustain_max_rc_regression_db)
            )
            if accept_by_render_compare or is_smooth_sustained_seam or is_duplicate_or_tiny_fragment:
                accepted_by_track_pitch.setdefault((candidate.instrument_index, candidate.pitch), set()).add(
                    candidate.left_index_in_pitch
                )
                accepted_improvements.append(improvement_db)
                accepted_per_instrument[candidate.instrument_index] = (
                    accepted_per_instrument.get(candidate.instrument_index, 0) + 1
                )

        final_stem_midi, merged_count = _apply_boundary_merges_to_midi(
            original_midis[stem_name],
            accepted_by_track_pitch,
            max_merged_duration_seconds=float(config.max_merged_duration_seconds),
        )
        final_midis[stem_name] = final_stem_midi
        after_note_count = sum(len(instrument.notes) for instrument in final_stem_midi.instruments)
        stem_reports[stem_name] = {
            "before_note_count": int(before_note_count),
            "after_note_count": int(after_note_count),
            "candidate_boundaries": int(len(stem_candidates)),
            "merged_boundaries": int(merged_count),
            "mean_improvement_db": float(np.mean(accepted_improvements)) if accepted_improvements else 0.0,
        }
        for instrument_index, (before_instrument, after_instrument) in enumerate(
            zip(original_midis[stem_name].instruments, final_stem_midi.instruments)
        ):
            if candidate_per_instrument.get(instrument_index, 0) == 0:
                continue
            track_reports.append(
                {
                    "stem_name": stem_name,
                    "instrument_name": str(before_instrument.name),
                    "program_number": int(before_instrument.program),
                    "before_note_count": int(len(before_instrument.notes)),
                    "after_note_count": int(len(after_instrument.notes)),
                    "candidate_boundaries": int(candidate_per_instrument.get(instrument_index, 0)),
                    "merged_boundaries": int(len(before_instrument.notes) - len(after_instrument.notes)),
                }
            )

    # 5. 結合済みステム MIDI・マージ済み MIDI を書き出す。
    written_stem_paths: dict[str, str] = {}
    if output_stem_dir is not None:
        stem_output_directory = Path(output_stem_dir)
        stem_output_directory.mkdir(parents=True, exist_ok=True)
        for stem_name, midi_object in final_midis.items():
            destination = stem_output_directory / f"{resolved_midis[stem_name].stem}_defragmented.mid"
            midi_object.write(str(destination))
            written_stem_paths[stem_name] = str(destination.resolve())

    merged_output_resolved: str | None = None
    if output_midi_path is not None:
        merged_midi = _merge_optimized_stem_midis(
            final_midis,
            max_melodic_instruments=int(config.max_melodic_instruments),
        )
        destination_path = Path(output_midi_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        merged_midi.write(str(destination_path))
        merged_output_resolved = str(destination_path.resolve())

    return {
        "stems": stems_with_notes,
        "total_before_notes": int(sum(item["before_note_count"] for item in stem_reports.values())),
        "total_after_notes": int(sum(item["after_note_count"] for item in stem_reports.values())),
        "total_candidate_boundaries": int(sum(item["candidate_boundaries"] for item in stem_reports.values())),
        "total_merged_boundaries": int(sum(item["merged_boundaries"] for item in stem_reports.values())),
        "stems_report": stem_reports,
        "tracks_report": track_reports,
        "output_midi_path": merged_output_resolved,
        "output_stem_paths": written_stem_paths,
    }
