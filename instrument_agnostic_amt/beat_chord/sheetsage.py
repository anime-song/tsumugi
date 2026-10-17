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

from dataclasses import dataclass
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


def load_sheetsage_model(device: torch.device | str) -> Any:
    """SheetSage2-Refined のビートトラッカーを ``device`` に読み込む。

    初回だけ SheetSage2 の重み (約 2.7GB、エンコーダー部分だけを使う) と
    8MB の bar head が Hugging Face からダウンロードされる。
    """
    try:
        from sheetsage2_refined.beats import HeadBeats
    except ImportError as error:
        raise ImportError(
            "SheetSage2-Refined (0.2.0 or later) is not installed. Install it with "
            "`uv sync --extra sheetsage` or "
            "`pip install git+https://github.com/anime-song/SheetSage2-Refined`."
        ) from error
    return HeadBeats.load(device=str(device))


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
    wave = torch.as_tensor(np.asarray(waveform, dtype=np.float32))
    if int(sample_rate) != model_rate:
        import torchaudio

        wave = torchaudio.functional.resample(wave, int(sample_rate), model_rate)
    duration = len(wave) / model_rate
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
