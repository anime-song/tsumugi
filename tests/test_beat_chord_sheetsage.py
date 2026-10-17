from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from instrument_agnostic_amt.beat_chord.cli import infer as beat_chord_infer
from instrument_agnostic_amt.beat_chord.sheetsage import (
    SheetSageBeatGrid,
    predict_sheetsage_beat_grid,
    sheetsage_beats_to_beat_grid,
)
from instrument_agnostic_amt.beat_chord.tempo_map_export import MeterSegmentSpec


def _result(
    beats: list[float],
    bars: list[tuple[float, int, int]],
) -> dict[str, object]:
    """ビートトラッカーの出力。bars は (開始, 拍の数, 分母)。"""
    return {
        "beats": beats,
        "downbeats": [start for start, _count, _unit in bars],
        "bars": [
            {"start": start, "beats": count, "unit": unit, "meter": f"{count}/{unit}"}
            for start, count, unit in bars
        ],
        "meters": sorted({count for _start, count, _unit in bars}),
    }


def _bars(grid: SheetSageBeatGrid) -> list[tuple[float, float, int, int]]:
    return [
        (segment.start_seconds, segment.end_seconds, segment.numerator, segment.denominator)
        for segment in grid.meter_segments
    ]


def test_bars_take_the_trackers_beat_count_and_unit_with_a_pickup_bar() -> None:
    # 弱起 2 拍、4/4 が 2 小節、拍が倍速になった 7/8、4/4 に戻る。
    beats = [0.0, 0.5]
    beats += [1.0 + 0.5 * index for index in range(8)]
    beats += [5.0 + 0.25 * index for index in range(7)]
    beats += [6.75 + 0.5 * index for index in range(4)]
    result = _result(beats, [(1.0, 4, 4), (3.0, 4, 4), (5.0, 7, 8), (6.75, 4, 4)])

    grid = sheetsage_beats_to_beat_grid(result, duration=10.0)

    assert grid.beat_times == tuple(beats)
    assert grid.downbeat_times == (1.0, 3.0, 5.0, 6.75)
    # 弱起は最初の小節と同じ分母で 2/4。最後の小節は曲中の 4/4 と同じ 4 拍あるので、
    # 直前の 7/8 と拍の数が違っても、最後の拍 + 1 拍分で閉じて残す。
    assert _bars(grid) == [
        (0.0, 1.0, 2, 4),
        (1.0, 3.0, 4, 4),
        (3.0, 5.0, 4, 4),
        (5.0, 6.75, 7, 8),
        (6.75, 8.75, 4, 4),
    ]


def test_last_bar_cut_short_by_the_end_of_the_song_is_dropped() -> None:
    # 曲が最後の小節の 2 拍目で終わっている。2/4 の小節として残すと拍子が嘘になる。
    beats = [0.5 * index for index in range(10)]
    result = _result(beats, [(0.0, 4, 4), (2.0, 4, 4), (4.0, 2, 4)])

    grid = sheetsage_beats_to_beat_grid(result, duration=5.0)

    assert _bars(grid) == [(0.0, 2.0, 4, 4), (2.0, 4.0, 4, 4)]


def test_last_bar_running_past_the_end_of_the_song_is_dropped() -> None:
    beats = [0.5 * index for index in range(8)]
    result = _result(beats, [(0.0, 4, 4), (2.0, 4, 4)])

    grid = sheetsage_beats_to_beat_grid(result, duration=3.8)

    assert _bars(grid) == [(0.0, 2.0, 4, 4)]


def test_song_without_downbeats_has_beats_but_no_bars() -> None:
    result = _result([0.0, 0.5, 1.0], [])

    grid = sheetsage_beats_to_beat_grid(result, duration=2.0)

    assert grid.beat_times == (0.0, 0.5, 1.0)
    assert grid.meter_segments == ()


def test_predict_resamples_to_the_encoder_rate_and_runs_the_tracker_on_the_waveform() -> None:
    calls: dict[str, object] = {}

    class FakeTracker:
        encoder = SimpleNamespace(config=SimpleNamespace(sample_rate=100))

        def probabilities(self, wave: torch.Tensor) -> dict[str, object]:
            calls["wave_length"] = len(wave)
            return {"beat": "probabilities"}

        def track_probabilities(self, probs: dict[str, object]) -> dict[str, object]:
            calls["probs"] = probs
            return _result([0.0, 0.5, 1.0, 1.5], [(0.0, 4, 4)])

    grid, result = predict_sheetsage_beat_grid(FakeTracker(), np.zeros(400), sample_rate=200)

    # 2 秒の波形を 100 Hz に落としてからエンコーダーに渡す。
    assert calls["wave_length"] == 200
    assert calls["probs"] == {"beat": "probabilities"}
    assert result["duration"] == 2.0
    assert result["bars"][0]["meter"] == "4/4"
    assert _bars(grid) == [(0.0, 2.0, 4, 4)]


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
    # ビートトラッカーはセクションを出さないので、セクションのトラックは作らない。
    assert "section_segments" not in export
    assert export["chord_segments"] is chord_segments
    assert export["key_segments"] is key_segments
