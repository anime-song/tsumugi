from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from instrument_agnostic_amt.beat_chord.cli import infer as beat_chord_infer
from instrument_agnostic_amt.beat_chord.sheetsage import (
    SheetSageBeatGrid,
    sheetsage_events_to_beat_grid,
)
from instrument_agnostic_amt.beat_chord.tempo_map_export import MeterSegmentSpec


def _beat(
    time: float,
    position: int,
    meter: tuple[int, int] | None = None,
    structure: str | None = None,
) -> dict[str, object]:
    """SheetSage2-Refined の 1 拍分のイベント。meter は JSON 経由と同じくリストで持つ。"""
    rhythm: dict[str, object] = {"eighth_position": position}
    if meter is not None:
        rhythm["meter"] = list(meter)
    values: dict[str, object] = {"timestamp": time, "rhythm": rhythm}
    if structure is not None:
        values["structure"] = structure
    return {"time": time, "values": values}


def _bars(grid: SheetSageBeatGrid) -> list[tuple[float, float, int, int]]:
    return [
        (segment.start_seconds, segment.end_seconds, segment.numerator, segment.denominator)
        for segment in grid.meter_segments
    ]


def test_short_first_bar_and_later_meter_change_become_their_own_meters() -> None:
    # 曲頭の 2 拍だけの小節 (実曲で出たもの)、4/4 の小節、3/4 への変更。
    events = [
        _beat(0.0, 0, meter=(4, 4), structure="intro"),
        _beat(0.5, 2),
        _beat(1.0, 0),
        _beat(1.5, 2),
        _beat(2.0, 4),
        _beat(2.5, 6),
        _beat(3.0, 0, meter=(3, 4), structure="verse"),
        _beat(3.5, 2),
        _beat(4.0, 4),
        _beat(4.5, 0),
        _beat(5.0, 2),
        _beat(5.5, 4),
    ]

    grid = sheetsage_events_to_beat_grid(events, duration=8.0)

    assert grid.beat_times == tuple(0.5 * index for index in range(12))
    assert grid.downbeat_times == (0.0, 1.0, 3.0, 4.5)
    # 最後の小節は拍子どおり 3 拍揃っているので、最後の拍 + 1 拍分で閉じる。
    assert _bars(grid) == [
        (0.0, 1.0, 2, 4),
        (1.0, 3.0, 4, 4),
        (3.0, 4.5, 3, 4),
        (4.5, 6.0, 3, 4),
    ]


def test_bar_shorter_than_its_meter_token_is_reexpressed_in_its_beat_unit() -> None:
    events = [
        _beat(0.0, 0, meter=(4, 4)),
        _beat(0.5, 2),
        _beat(1.0, 4),
        _beat(1.5, 0),
        _beat(2.0, 2),
    ]

    grid = sheetsage_events_to_beat_grid(events, duration=4.0)

    # 4/4 のトークンのまま 3 拍で次のダウンビートが来た小節は 3/4。
    # 最後の 2 拍は拍子どおりに揃っていないので小節にしない。
    assert _bars(grid) == [(0.0, 1.5, 3, 4)]


def test_compound_meter_keeps_its_token_even_with_dotted_quarter_beats() -> None:
    # 6/8 を付点 4 分で刻むと 1 小節 2 拍 (位置 0, 3)。長さは 8 分音符 6 つでトークンと合う。
    events = [
        _beat(0.0, 0, meter=(6, 8)),
        _beat(0.6, 3),
        _beat(1.2, 0),
        _beat(1.8, 3),
        _beat(2.4, 0),
    ]

    grid = sheetsage_events_to_beat_grid(events, duration=3.0)

    assert _bars(grid)[:2] == [(0.0, 1.2, 6, 8), (1.2, 2.4, 6, 8)]


def test_meter_token_wins_when_the_beat_count_matches_its_numerator() -> None:
    # 6/8 を 8 分音符で 6 拍刻む小節は、拍の数が分子と一致するのでトークンどおり。
    # 同じ長さの 3/4 とは拍の数とトークンで区別される。
    events = [_beat(0.2 * index, index % 6, meter=(6, 8) if index == 0 else None) for index in range(12)]
    events.append(_beat(2.4, 0, meter=(3, 4)))
    events += [_beat(2.4 + 0.4 * index, 2 * index) for index in range(1, 3)]

    grid = sheetsage_events_to_beat_grid(events, duration=4.0)

    assert [bar[2:] for bar in _bars(grid)] == [(6, 8), (6, 8), (3, 4)]


@pytest.mark.parametrize(
    ("beat_seconds", "expected_meter"),
    [(0.25, (7, 8)), (0.5, (7, 4))],
)
def test_bar_that_breaks_its_meter_token_takes_its_unit_from_neighboring_bars(
    beat_seconds: float,
    expected_meter: tuple[int, int],
) -> None:
    # トークンは 4/4 のまま、実際は 7 拍の小節。位置もトークンにつられて
    # 4 分音符刻み (0, 2, ..., 12) で付いているので、位置からは常に 7/4 に見える。
    # 前後の 4/4 の 4 分音符 (0.5 秒) と比べた 1 拍の長さで分母を決める。
    events = [_beat(0.5 * index, 2 * index, meter=(4, 4) if index == 0 else None) for index in range(4)]
    odd_start = 2.0
    events += [_beat(odd_start + beat_seconds * index, 2 * index) for index in range(7)]
    after_start = odd_start + beat_seconds * 7
    events += [_beat(after_start + 0.5 * index, 2 * index) for index in range(4)]
    events.append(_beat(after_start + 2.0, 0))

    grid = sheetsage_events_to_beat_grid(events, duration=after_start + 4.0)

    assert [bar[2:] for bar in _bars(grid)] == [(4, 4), expected_meter, (4, 4)]


def test_pickup_beats_before_the_first_downbeat_form_a_partial_bar() -> None:
    events = [
        _beat(0.0, 4, meter=(4, 4)),
        _beat(0.5, 6),
        _beat(1.0, 0),
        _beat(1.5, 2),
        _beat(2.0, 4),
        _beat(2.5, 6),
        _beat(3.0, 0),
    ]

    grid = sheetsage_events_to_beat_grid(events, duration=3.2)

    # 弱起 (4/4 の 3 拍目から) は 2/4 の小節。ダウンビート 1 拍だけの最後の小節は
    # 長さが分からないので残さない。
    assert _bars(grid) == [(0.0, 1.0, 2, 4), (1.0, 3.0, 4, 4)]


def test_sections_keep_repeated_labels_as_phrase_boundaries() -> None:
    events = [
        _beat(0.0, 0, meter=(4, 4), structure="intro"),
        _beat(2.0, 0, structure="verse"),
        _beat(4.0, 0, structure="verse"),
        {"time": 6.0, "values": {"timestamp": 6.0, "structure": "chorus"}},
    ]

    grid = sheetsage_events_to_beat_grid(events, duration=10.0)

    assert grid.section_segments == (
        {"start": 0.0, "end": 2.0, "label": "intro"},
        {"start": 2.0, "end": 4.0, "label": "verse"},
        {"start": 4.0, "end": 6.0, "label": "verse"},
        {"start": 6.0, "end": 10.0, "label": "chorus"},
    )


def test_predict_beat_chord_uses_the_given_beat_grid_and_skips_beat_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_midi = tmp_path / "input.mid"
    input_midi.write_bytes(b"midi")
    beat_grid = SheetSageBeatGrid(
        beat_times=(0.0, 0.5, 1.0, 1.5),
        downbeat_times=(0.0,),
        meter_segments=(MeterSegmentSpec(0.0, 2.0, 4, 4),),
        section_segments=({"start": 0.0, "end": 2.0, "label": "intro"},),
    )
    captured: dict[str, object] = {}
    chord_segments = [{"start": 0.0, "end": 2.0, "chord": "C:maj"}]
    key_segments = [{"start": 0.0, "end": 2.0, "key": "C"}]

    def fake_run(**kwargs: object) -> SimpleNamespace:
        captured["config"] = kwargs["config"]
        return SimpleNamespace(
            beat_times=[],
            meter_segments=[],
            chord_segments=chord_segments,
            key_segments=key_segments,
            duration_seconds=2.0,
            hop_length=1,
            sample_rate=1,
        )

    def fake_export(**kwargs: object) -> None:
        captured["export"] = kwargs

    monkeypatch.setattr(beat_chord_infer, "run_beat_chord_inference", fake_run)
    monkeypatch.setattr(beat_chord_infer, "export_tempo_mapped_midi", fake_export)

    beat_chord_infer.predict_beat_chord_for_midi(
        input_midi,
        device="cpu",
        grid_jit=False,
        preloaded_model=torch.nn.Linear(1, 1),
        preloaded_model_config=object(),
        preloaded_metadata={"checkpoint_args": {}},
        audio_path=tmp_path / "song.wav",
        beat_grid=beat_grid,
    )

    config = captured["config"]
    assert config.beat_decode_mode == "none"
    assert config.audio_path is None
    export = captured["export"]
    assert export["beat_times"] == list(beat_grid.beat_times)
    assert export["meter_segments"] == list(beat_grid.meter_segments)
    assert export["section_segments"] == list(beat_grid.section_segments)
    assert export["chord_segments"] is chord_segments
    assert export["key_segments"] is key_segments
