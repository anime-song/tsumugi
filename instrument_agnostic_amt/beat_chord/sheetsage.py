"""SheetSage2-Refined のビートトラッカーをビート・ダウンビート・拍子の推定元として使う。

tsumugi の beat/chord モデルは MIDI ロールしか見ないため、ビートと拍子は元音源を
直接聴く SheetSage2-Refined (https://github.com/anime-song/SheetSage2-Refined) の方が
精度が高い。一方コードとキーは tsumugi の方が良いので、ここでは拍の格子だけを取り出し、
tempo map の書き出し (``tempo_map_export.export_tempo_mapped_midi``) が受け取る形に直す。

使うのは 0.2.0 で入ったデコーダーなしのビートトラッカー (``sheetsage2_refined.beats.HeadBeats``)。
凍結した SheetSage2 のエンコーダーに載せた bar head のフレーム確率から、曲全体を 1 本の
HMM (bar_tracker) で復号する。デコーダー版より速く、変拍子の小節も多く当たる。
デコーダーを通さないので、セクション (intro / verse / chorus ...) は出ない。

ビートトラッカーの出力 (``track_probabilities`` の戻り値) の読み方:

* ``beats`` / ``downbeats`` は秒。ダウンビートはどれかの拍と同じ時刻。
* ``bars`` はダウンビートごとの小節 ``{"start": 秒, "beats": 拍の数, "unit": 分母, "meter": "7/8"}``。
  ``unit`` は bar head の予測から、書かれるテンポが小節間で跳ばないように選ばれている
  (4/4 の後で拍が倍速の 7 拍の小節は 7/4 ではなく 7/8)。
* 最初のダウンビートより前の拍 (弱起) は ``bars`` に入らない。最後の小節は曲末までの拍を数える。

SheetSage2-Refined のコードは MIT、重みは CC BY-NC 4.0 (非商用) である。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .tempo_map_export import MeterSegmentSpec


@dataclass(frozen=True)
class SheetSageBeatGrid:
    beat_times: tuple[float, ...]
    downbeat_times: tuple[float, ...]
    # 1 小節 1 区間 (bar_count=1)。tempo map の拍子とテンポはここから作る。
    meter_segments: tuple[MeterSegmentSpec, ...]


@dataclass(frozen=True)
class SheetSageMelodyNote:
    """SheetSage2-Refined (HeadMelody) が出力する 1 音符と歌唱確率。"""

    start_seconds: float
    end_seconds: float
    pitch: int
    track: int  # 0: sung (歌唱), 1: played (楽器メロディ)
    vocal_probability: float
    steps: int | None = None
    position_in_beat: int | None = None


@dataclass(frozen=True)
class SheetSageMelodyPrediction:
    """SheetSage2-Refined (HeadMelody) による曲全体のメロディ推定結果。"""

    duration_seconds: float
    notes: tuple[SheetSageMelodyNote, ...]
    vocal_probabilities: np.ndarray  # [num_frames] float32 (25 Hz)
    frame_rate: float = 25.0


def load_sheetsage_model(device: torch.device | str) -> Any:
    """SheetSage2-Refined のビートトラッカーを ``device`` に読み込む。

    初回だけ SheetSage2 の重み (約 2.7GB、エンコーダー部分だけを使う) と
    8MB の bar head が Hugging Face からダウンロードされる。
    """
    try:
        from sheetsage2_refined.beats import HeadBeats
    except ImportError as error:
        raise ImportError(
            "SheetSage2-Refined (0.5.2 or later) is not installed. Install it with "
            "`uv sync --extra sheetsage` or "
            "`pip install git+https://github.com/anime-song/SheetSage2-Refined`."
        ) from error
    return HeadBeats.load(device=str(device))


def load_sheetsage_melody_model(
    device: torch.device | str,
    *,
    shared_encoder: Any | None = None,
) -> Any:
    """SheetSage2-Refined 0.5.2+ のデコーダーなしメロディトラッカー (HeadMelody) を読み込む。

    ``shared_encoder`` に読み込み済み ``HeadBeats.encoder`` を渡すと、約 2.7 GB の
    MERTEncoder 重みを二重ロードせず 8 MB の melody head のみを追加ロードして共有する。
    """
    try:
        from sheetsage2_refined.heads import BeatInformed, load_head
        from sheetsage2_refined.melody import HeadMelody, beat_head_files
        from sheetsage2_refined.pipeline import DEFAULT_MODEL, model_folder
    except ImportError as error:
        raise ImportError(
            "SheetSage2-Refined 0.5.2+ (with sheetsage2_refined.melody.HeadMelody) is not installed. "
            "Install or upgrade it with `uv sync --extra sheetsage`."
        ) from error

    target_device = torch.device(device)
    if shared_encoder is None:
        return HeadMelody.load(device=str(target_device))

    def _melody_head_only_files(config: Mapping[str, Any]) -> list[str]:
        melody_section = config.get("melody_tracker") or {}
        if not melody_section.get("head"):
            raise ValueError("This SheetSage2-Refined model has no melody_tracker.head")
        return [str(melody_section["head"]), *beat_head_files(dict(config))]

    folder = model_folder(DEFAULT_MODEL, None, _melody_head_only_files)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    melody_section = config.get("melody_tracker") or {}
    melody_head_path = folder / str(melody_section["head"])
    beat_head_candidates = beat_head_files(config)
    beat_head_path = folder / str(beat_head_candidates[0]) if beat_head_candidates else None
    decoding_options = dict(melody_section.get("decoding") or {})

    melody_head = load_head(Path(melody_head_path), target_device)
    is_line_head = melody_head.line is not None
    if melody_head.extra is not None:
        if beat_head_path is None:
            raise ValueError("Beat-informed melody head requires melody_tracker.beat_head")
        melody_head = BeatInformed(load_head(Path(beat_head_path), target_device), melody_head).eval()

    return HeadMelody(shared_encoder, melody_head, target_device, is_line_head, **decoding_options)


def _resample_waveform_for_encoder(
    waveform: np.ndarray | torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
) -> tuple[torch.Tensor, float]:
    """モノラル波形をエンコーダ規定レート (24 kHz) の float32 テンソルへ揃える。"""
    wave = torch.as_tensor(np.asarray(waveform, dtype=np.float32))
    if wave.ndim > 1:
        wave = wave.mean(dim=0)
    if int(sample_rate) != int(target_sample_rate):
        import torchaudio

        wave = torchaudio.functional.resample(wave, int(sample_rate), int(target_sample_rate))
    duration_seconds = float(len(wave)) / float(target_sample_rate)
    return wave, duration_seconds


def _build_melody_prediction_from_probabilities(
    melody_model: Any,
    melody_probabilities: Mapping[str, np.ndarray],
    duration_seconds: float,
) -> tuple[SheetSageMelodyPrediction, dict[str, Any]]:
    """HeadMelody のフレーム確率からノート列とノート別歌唱確率を構築する。"""
    try:
        from sheetsage2_refined.heads import FRAME_RATE

        frame_rate = float(FRAME_RATE)
    except ImportError:
        frame_rate = 25.0

    raw_notes = melody_model.notes(dict(melody_probabilities))
    vocal_probabilities = np.asarray(
        melody_probabilities.get("melody_vocal", np.ones(1, dtype=np.float32)),
        dtype=np.float32,
    )
    frame_count = int(vocal_probabilities.size)

    structured_notes: list[SheetSageMelodyNote] = []
    for raw_note in raw_notes:
        start_seconds = float(raw_note["start"])
        end_seconds = max(start_seconds + 0.01, float(raw_note["end"]))
        start_frame = max(0, min(frame_count - 1, int(round(start_seconds * frame_rate))))
        end_frame = min(frame_count, max(start_frame + 1, int(round(end_seconds * frame_rate))))
        note_vocal_probability = (
            float(vocal_probabilities[start_frame:end_frame].mean())
            if frame_count > 0
            else (1.0 if int(raw_note.get("track", 0)) == 0 else 0.0)
        )
        structured_notes.append(
            SheetSageMelodyNote(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                pitch=int(raw_note["pitch"]),
                track=int(raw_note.get("track", 0)),
                vocal_probability=note_vocal_probability,
                steps=int(raw_note["steps"]) if "steps" in raw_note else None,
                position_in_beat=int(raw_note["pos"]) if "pos" in raw_note else None,
            )
        )

    prediction = SheetSageMelodyPrediction(
        duration_seconds=duration_seconds,
        notes=tuple(structured_notes),
        vocal_probabilities=vocal_probabilities,
        frame_rate=frame_rate,
    )
    raw_result = {"duration": duration_seconds, "notes": raw_notes}
    return prediction, raw_result


@torch.no_grad()
def predict_sheetsage_beats_and_melody(
    beat_model: Any | None,
    melody_model: Any | None,
    waveform: np.ndarray | torch.Tensor,
    sample_rate: int,
) -> tuple[
    SheetSageBeatGrid | None,
    dict[str, Any] | None,
    SheetSageMelodyPrediction | None,
    dict[str, Any] | None,
]:
    """共有 MERTEncoder を 1 回だけ走査し、ビート格子とメロディを同時に推定する。

    `beat_model` と `melody_model` の片方だけ渡した場合もそのまま動作する。
    両方渡した場合はエンコーダの窓特徴 ``states`` を各ヘッドへ流用するため、
    追加 VRAM なし・ほぼ追加計算時間なしでビートとメロディを同時取得できる。
    """
    if beat_model is None and melody_model is None:
        return None, None, None, None

    primary_model = beat_model if beat_model is not None else melody_model
    encoder = primary_model.encoder
    model_rate = int(encoder.config.sample_rate)
    device = torch.device(primary_model.device)
    wave, duration_seconds = _resample_waveform_for_encoder(waveform, sample_rate, model_rate)

    # 両モデルが同一エンコーダを共有する場合は encode_windows を 1 パスで済ませる。
    if beat_model is not None and melody_model is not None and beat_model.encoder is melody_model.encoder:
        from sheetsage2_refined.beats import encode_windows
        from sheetsage2_refined.heads import CHUNK, LINE, OFFSETS

        beat_window_outputs: list[dict[str, torch.Tensor]] = []
        melody_window_outputs: list[dict[str, torch.Tensor]] = []
        for states, keep in encode_windows(encoder, wave, device):
            beat_window_outputs.append({key: value[:, :keep] for key, value in beat_model.head(states, CHUNK).items()})
            melody_window_outputs.append(
                {key: value[:, :keep] for key, value in melody_model.head(states, CHUNK).items()}
            )

        beat_logits = {
            key: torch.cat([window_out[key] for window_out in beat_window_outputs], dim=1)[0].float().cpu()
            for key in beat_window_outputs[0]
        }
        beat_probs: dict[str, np.ndarray] = {
            key: torch.sigmoid(beat_logits[key]).numpy() for key in ("beat", "downbeat")
        }
        beat_probs |= {
            key: torch.softmax(beat_logits[key], dim=-1).numpy()
            for key in ("bar_beats", "beats_left", "bar_unit")
            if key in beat_logits
        }
        beat_probs |= {key: beat_logits[key].numpy() for key in OFFSETS if key in beat_logits}
        raw_beats = {"duration": duration_seconds} | beat_model.track_probabilities(beat_probs)
        beat_grid = sheetsage_beats_to_beat_grid(raw_beats, duration_seconds)

        melody_logits = {
            key: torch.cat([window_out[key] for window_out in melody_window_outputs], dim=1)[0].float().cpu()
            for key in melody_window_outputs[0]
        }
        melody_probs: dict[str, np.ndarray] = {
            key: torch.sigmoid(melody_logits[key]).numpy()
            for key in ("melody", "melody_on", "melody_vocal")
            if key in melody_logits
        }
        melody_probs |= {key: torch.softmax(melody_logits[key], dim=-1).numpy() for key in LINE if key in melody_logits}
        melody_probs["melody_offset"] = melody_logits["melody_offset"].numpy()
        melody_prediction, raw_melody = _build_melody_prediction_from_probabilities(
            melody_model,
            melody_probs,
            duration_seconds,
        )
        return beat_grid, raw_beats, melody_prediction, raw_melody

    beat_grid: SheetSageBeatGrid | None = None
    raw_beats: dict[str, Any] | None = None
    if beat_model is not None:
        raw_beats = {"duration": duration_seconds} | beat_model.track_probabilities(beat_model.probabilities(wave))
        beat_grid = sheetsage_beats_to_beat_grid(raw_beats, duration_seconds)

    melody_prediction: SheetSageMelodyPrediction | None = None
    raw_melody: dict[str, Any] | None = None
    if melody_model is not None:
        melody_probs = melody_model.probabilities(wave)
        melody_prediction, raw_melody = _build_melody_prediction_from_probabilities(
            melody_model,
            melody_probs,
            duration_seconds,
        )
    return beat_grid, raw_beats, melody_prediction, raw_melody


def predict_sheetsage_beat_grid(
    model: Any,
    waveform: np.ndarray | torch.Tensor,
    sample_rate: int,
) -> tuple[SheetSageBeatGrid, dict[str, Any]]:
    """モノラル波形から拍の格子を推定する。

    戻り値は (拍の格子, ビートトラッカーの出力)。出力は
    ``python -m sheetsage2_refined.beats`` が書く JSON と同じ形 (``duration`` 付き) で、
    呼び出し側で保存できるように返す。
    """
    # エンコーダーは 24 kHz モノラルの float32 波形を受け取る。
    model_rate = int(model.encoder.config.sample_rate)
    wave, duration = _resample_waveform_for_encoder(waveform, sample_rate, model_rate)
    # HeadBeats.track はファイルを読むので、読み込み済みの波形からは
    # probabilities (エンコーダー + bar head) と track_probabilities (HMM) を順に呼ぶ。
    result = {"duration": duration} | model.track_probabilities(model.probabilities(wave))
    return sheetsage_beats_to_beat_grid(result, duration), result


def sheetsage_beats_to_beat_grid(
    result: Mapping[str, Any],
    duration: float,
) -> SheetSageBeatGrid:
    """ビートトラッカーの出力を拍の格子に直す。"""
    beats = [float(time) for time in result["beats"]]
    downbeats = [float(time) for time in result["downbeats"]]
    bars = list(result["bars"])

    # 1. 小節ごとの (開始, 終了, 分子, 分母)。分子はトラッカーが数えた拍の数、
    #    分母は ``unit`` をそのまま使う。終わりは次の小節の頭。
    meter_bars: list[tuple[float, float, int, int]] = [
        (float(bar["start"]), float(next_bar["start"]), int(bar["beats"]), int(bar["unit"]))
        for bar, next_bar in zip(bars[:-1], bars[1:])
    ]

    # 2. 最後の小節は終わりが分からないので、最後の拍 + 1 拍分 (直前の拍間隔と同じ長さ)
    #    で閉じる。曲が小節の途中で終わっていると拍の数が足りず、短い拍子の小節に
    #    見えてしまうので、それより前のどこかの小節と同じ拍の数 (曲中で使われている拍子)
    #    のときだけ残す。小節が 1 つしかなければそのまま残す。
    if bars:
        last_bar = bars[-1]
        last_beats = [time for time in beats if time >= float(last_bar["start"]) - 1e-6]
        is_complete = len(bars) == 1 or int(last_bar["beats"]) in {
            int(bar["beats"]) for bar in bars[:-1]
        }
        if len(last_beats) >= 2 and is_complete:
            end_seconds = last_beats[-1] + (last_beats[-1] - last_beats[-2])
            if end_seconds <= duration + 1e-6:
                meter_bars.append(
                    (float(last_bar["start"]), end_seconds, int(last_bar["beats"]), int(last_bar["unit"]))
                )

    # 3. 最初のダウンビートより前の拍 (弱起) は、小節の後ろ側だけがある 1 小節にする。
    #    分子は拍の数、分母は最初の小節と同じ (同じ拍の長さが同じ音価で書かれる)。
    #    4/4 の 3 拍目から始まる曲なら 2/4。
    if bars:
        pickup_count = sum(1 for time in beats if time < downbeats[0] - 1e-6)
        if pickup_count > 0:
            meter_bars.insert(0, (beats[0], downbeats[0], pickup_count, int(bars[0]["unit"])))

    return SheetSageBeatGrid(
        beat_times=tuple(beats),
        downbeat_times=tuple(downbeats),
        meter_segments=tuple(
            MeterSegmentSpec(
                start_seconds=start,
                end_seconds=end,
                numerator=numerator,
                denominator=denominator,
            )
            for start, end, numerator, denominator in meter_bars
        ),
    )
