"""SheetSage2-Refined をビート・ダウンビート・拍子・セクションの推定元として使う。

tsumugi の beat/chord モデルは MIDI ロールしか見ないため、ビートと拍子は元音源を
直接聴く SheetSage2-Refined (https://github.com/anime-song/SheetSage2-Refined) の方が
精度が高い。一方コードとキーは tsumugi の方が良いので、ここでは SheetSage2-Refined の
イベント列から拍の格子とセクションだけを取り出し、tempo map の書き出し
(``tempo_map_export.export_tempo_mapped_midi``) が受け取る形に直す。

SheetSage2-Refined のイベント (``values``) の読み方:

* ``rhythm`` を持つイベントが 1 拍。``eighth_position`` は小節頭からの位置を
  8 分音符単位で表し、0 がダウンビート (4/4 なら 0, 2, 4, 6)。
* ``rhythm.meter`` (分子, 分母) は拍子が変わる小節頭と窓の先頭にしか現れないので、
  直前の値を持ち越す。
* ``structure`` (intro / verse / chorus ...) はセクション頭に現れる。同じラベルが
  続くのはフレーズの区切り (例: 8 小節ごとの verse) で、そのまま残す。

SheetSage2-Refined のコードは MIT、重みは CC BY-NC 4.0 (非商用) である。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .tempo_map_export import MeterSegmentSpec


@dataclass(frozen=True)
class SheetSageBeatGrid:
    beat_times: tuple[float, ...]
    downbeat_times: tuple[float, ...]
    # 1 小節 1 区間 (bar_count=1)。tempo map の拍子とテンポはここから作る。
    meter_segments: tuple[MeterSegmentSpec, ...]
    # {"start": 秒, "end": 秒, "label": "chorus"} の列。
    section_segments: tuple[dict[str, Any], ...]


def load_sheetsage_model(device: torch.device | str) -> Any:
    """SheetSage2-Refined を ``device`` に読み込む。重み (約 2.9GB) は初回だけダウンロードされる。"""
    try:
        from sheetsage2_refined import SheetSage2Refined
    except ImportError as error:
        raise ImportError(
            "SheetSage2-Refined is not installed. Install it with "
            "`uv sync --extra sheetsage` or "
            "`pip install git+https://github.com/anime-song/SheetSage2-Refined`."
        ) from error
    return SheetSage2Refined.load(device=str(device))


def predict_sheetsage_beat_grid(
    model: Any,
    waveform: np.ndarray | torch.Tensor,
    sample_rate: int,
) -> tuple[SheetSageBeatGrid, list[dict[str, Any]], float]:
    """モノラル波形から拍の格子を推定する。

    戻り値は (拍の格子, SheetSage2-Refined の生イベント列, 曲の長さ秒)。
    生イベント列は ``python -m sheetsage2_refined.visualize`` で確認できるよう、
    呼び出し側で JSON に保存できるように返す。
    """
    # SheetSage2-Refined は 24 kHz モノラルの float32 波形を受け取る。
    model_rate = int(model.model.sample_rate)
    wave = torch.as_tensor(np.asarray(waveform, dtype=np.float32))
    if int(sample_rate) != model_rate:
        import torchaudio

        wave = torchaudio.functional.resample(wave, int(sample_rate), model_rate)
    duration = len(wave) / model_rate
    # プロンプトはモデルの既定 (メロディ・コード・キーも含む全部) のまま使う。
    # 拍子と構造だけに絞ると約 2 倍速くなるが、ビートが数 % 変わる。
    # 公表されている精度は既定プロンプトで測ったものなので、それに合わせる。
    events = model.transcribe([wave])[0]
    return sheetsage_events_to_beat_grid(events, duration), events, duration


def sheetsage_events_to_beat_grid(
    events: Sequence[dict[str, Any]],
    duration: float,
) -> SheetSageBeatGrid:
    """SheetSage2-Refined のイベント列を拍の格子とセクションに直す。"""

    # 1. 拍を時刻順に拾う。拍子トークンは持ち越して各拍に付ける。
    #    窓のつなぎ目で同じ時刻の拍が重複しても 1 拍として扱う。
    ordered_events = sorted(events, key=lambda item: float(item["time"]))
    beats: list[tuple[float, int, tuple[int, int]]] = []
    meter = (4, 4)
    for event in ordered_events:
        rhythm = event["values"].get("rhythm")
        if rhythm is None:
            continue
        if "meter" in rhythm:
            meter = (int(rhythm["meter"][0]), int(rhythm["meter"][1]))
        time = float(event["time"])
        if beats and time <= beats[-1][0] + 1e-6:
            continue
        beats.append((time, int(rhythm.get("eighth_position", -1)), meter))

    # 2. 拍の長さ (8 分音符いくつ分か) の代表値。1 拍しかない小節の長さを決めるのに使う。
    #    4/4 なら 2、8 分の 6 拍子を 8 分音符で刻んでいれば 1。
    steps = Counter(
        right[1] - left[1]
        for left, right in zip(beats[:-1], beats[1:])
        if right[1] > left[1] >= 0
    )
    typical_step = steps.most_common(1)[0][0] if steps else 2

    # 3. 小節に分ける。ダウンビートから次のダウンビートまでを 1 小節とする。
    #    曲頭のダウンビートより前の拍 (弱起) も、小節の後ろ側だけがある 1 小節として扱う。
    #    最後のダウンビート以降は小節の終わりが分からないので、最後の拍 + 1 拍分
    #    (直前の拍間隔と同じ長さ) で閉じる。
    #    length_eighths は位置から求めた小節の長さ (8 分音符単位、
    #    「最後の拍の位置 + その拍の長さ」)。
    bars: list[dict[str, Any]] = []
    downbeat_indices = [index for index, beat in enumerate(beats) if beat[1] == 0]
    if downbeat_indices and downbeat_indices[0] > 0 and beats[0][1] > 0:
        first_downbeat = beats[downbeat_indices[0]]
        token = first_downbeat[2]
        length_eighths = token[0] * 8 // token[1] - beats[0][1]
        if length_eighths > 0:
            bars.append(
                {
                    "start": beats[0][0],
                    "end": first_downbeat[0],
                    "beat_count": downbeat_indices[0],
                    "length_eighths": length_eighths,
                    "token": token,
                    "is_last": False,
                }
            )
    for bar_index, start_index in enumerate(downbeat_indices):
        is_last = bar_index + 1 == len(downbeat_indices)
        end_index = len(beats) if is_last else downbeat_indices[bar_index + 1]
        bar_beats = beats[start_index:end_index]
        positions = [beat[1] for beat in bar_beats]
        if len(positions) >= 2 and all(
            right > left for left, right in zip(positions[:-1], positions[1:])
        ):
            length_eighths = positions[-1] + (positions[-1] - positions[-2])
        else:
            length_eighths = len(positions) * typical_step
        if is_last:
            if len(bar_beats) < 2:
                continue
            end_seconds = bar_beats[-1][0] + (bar_beats[-1][0] - bar_beats[-2][0])
            if end_seconds > duration + 1e-6:
                continue
        else:
            end_seconds = beats[end_index][0]
        bars.append(
            {
                "start": bar_beats[0][0],
                "end": end_seconds,
                "beat_count": len(bar_beats),
                "length_eighths": length_eighths,
                "token": bar_beats[0][2],
                "is_last": is_last,
            }
        )

    # 4. 拍子トークンを優先する。小節の拍の数がトークンの分子と一致するか、
    #    位置から求めた長さがトークンどおり (6/8 を付点 4 分で 2 拍に刻む場合など) なら
    #    トークンを使う。トークンどおりの小節では 4 分音符の長さ (秒) も控えておき、
    #    トークンと合わない小節の分母を決める基準にする。
    for bar in bars:
        numerator, denominator = bar["token"]
        fits_token = (
            bar["beat_count"] == numerator
            or bar["length_eighths"] * denominator == numerator * 8
        )
        bar["meter"] = bar["token"] if fits_token else None
        bar["quarter_seconds"] = (
            (bar["end"] - bar["start"]) * denominator / (4 * numerator)
            if fits_token
            else None
        )

    # 5. トークンと合わない小節 (4/4 のトークンのまま実際は 7 拍だった小節、
    #    曲頭の 2 拍だけの小節など) は、分子を拍の数とし、分母は前後のつながりで決める。
    #    拍の数だけでは 7/8 と 7/4 を区別できないので、前後でいちばん近いトークンどおりの
    #    小節の 4 分音符の長さを基準に、この小節の 1 拍がその何分音符に当たるかを
    #    2・4・8・16 から選ぶ。前後の 4 分音符が 0.5 秒で 1 拍が 0.25 秒なら 8 分音符
    #    (7/8)、0.5 秒なら 4 分音符 (7/4)。分母を誤るとその小節だけ MIDI のテンポが
    #    倍や半分に跳ぶので、テンポが前後とつながる分母を選ぶことにもなる。
    #    トークンどおりの小節が曲中に 1 つもなければ、位置から求めた長さを
    #    トークンの分母で表し直す。最後の小節は終わりを推測で閉じているので、
    #    トークンと合わなければ残さない。
    for bar_index, bar in enumerate(bars):
        if bar["meter"] is not None:
            continue
        if bar["is_last"]:
            continue
        previous_quarter = next(
            (
                other["quarter_seconds"]
                for other in reversed(bars[:bar_index])
                if other["quarter_seconds"] is not None
            ),
            None,
        )
        next_quarter = next(
            (
                other["quarter_seconds"]
                for other in bars[bar_index + 1 :]
                if other["quarter_seconds"] is not None
            ),
            None,
        )
        neighbor_quarters = [
            value for value in (previous_quarter, next_quarter) if value is not None
        ]
        if neighbor_quarters:
            # 前後の 4 分音符の長さは幾何平均でまとめる (テンポの比で比べるため)。
            reference_quarter = float(np.exp(np.mean(np.log(neighbor_quarters))))
            beat_seconds = (bar["end"] - bar["start"]) / bar["beat_count"]
            denominator = min(
                (2, 4, 8, 16),
                key=lambda value: abs(
                    np.log(beat_seconds * value / 4 / reference_quarter)
                ),
            )
            bar["meter"] = (bar["beat_count"], denominator)
            continue
        token_denominator = bar["token"][1]
        if (bar["length_eighths"] * token_denominator) % 8 == 0:
            bar["meter"] = (
                bar["length_eighths"] * token_denominator // 8,
                token_denominator,
            )
        else:
            bar["meter"] = (bar["length_eighths"], 8)

    meter_segments = [
        MeterSegmentSpec(
            start_seconds=bar["start"],
            end_seconds=bar["end"],
            numerator=bar["meter"][0],
            denominator=bar["meter"][1],
        )
        for bar in bars
        if bar["meter"] is not None
    ]

    # 6. セクション。各 structure イベントを区間の頭とし、次の頭 (最後は曲末) までとする。
    section_starts = [
        (float(event["time"]), str(event["values"]["structure"]))
        for event in ordered_events
        if "structure" in event["values"]
    ]
    section_segments = tuple(
        {
            "start": start,
            "end": section_starts[index + 1][0] if index + 1 < len(section_starts) else float(duration),
            "label": label,
        }
        for index, (start, label) in enumerate(section_starts)
    )

    return SheetSageBeatGrid(
        beat_times=tuple(beat[0] for beat in beats),
        downbeat_times=tuple(beats[index][0] for index in downbeat_indices),
        meter_segments=tuple(meter_segments),
        section_segments=section_segments,
    )
