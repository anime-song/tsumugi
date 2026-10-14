"""The streaming emitters must agree with the offline ``finalize()`` output."""

from __future__ import annotations

from instrument_agnostic_amt.amt.inference.types import PredictedNote
from instrument_agnostic_amt.amt.inference.windowed import WindowNoteStitcher
from instrument_agnostic_amt.streaming.emitters import make_emitter

HOP = 512
TOTAL_FRAMES = 40_000
MERGE_GAP = 512
MERGE_ONSET = 1_102


def _note(
    start: int,
    end: int,
    *,
    pitch: int = 41,
    has_onset: bool,
    has_offset: bool,
    slot_index: int = 0,
    candidates: tuple[int, ...] = (21, 11, 23),
) -> PredictedNote:
    return PredictedNote(
        instrument_id=0,
        pitch=pitch,
        start_sample=start,
        end_sample=end,
        velocity=100,
        slot_index=slot_index,
        has_onset=has_onset,
        has_offset=has_offset,
        instrument_candidates=candidates,
    )


def _stitcher() -> WindowNoteStitcher:
    return WindowNoteStitcher(
        hop_length=HOP,
        total_audio_frames=TOTAL_FRAMES,
        velocity=100,
        merge_gap_samples=MERGE_GAP,
        merge_onset_samples=MERGE_ONSET,
    )


def _emitter(stitcher: WindowNoteStitcher, version: str):
    return make_emitter(
        version,
        stitcher,
        stem="piano",
        sample_rate=22_050,
        merge_gap_samples=MERGE_GAP,
        merge_onset_samples=MERGE_ONSET,
        instrument_label=lambda instrument_id: f"class-{instrument_id}",
    )


def _streamed(
    tracks: dict[int, list[PredictedNote]], version: str
) -> tuple[list[tuple[int, int, int]], list]:
    """Feed one window per track state, then finish; return the final note set."""
    stitcher = _stitcher()
    emitter = _emitter(stitcher, version)
    events = []
    windows = max((len(notes) for notes in tracks.values()), default=0)
    for index in range(windows):
        for track, notes in tracks.items():
            stitcher.notes_by_pair[track] = notes[: index + 1]
        events.extend(emitter.on_window(index * 4_000))
    events.extend(emitter.finish())
    return events, stitcher


def _offline(tracks: dict[int, list[PredictedNote]]) -> list[tuple[int, int, int]]:
    stitcher = _stitcher()
    for track, notes in tracks.items():
        stitcher.notes_by_pair[track] = [note for note in notes]
    return [
        (note.pitch, note.start_sample, note.end_sample)
        for note in stitcher.finalize()
    ]


def _final_events(events: list) -> list[tuple[int, int, int]]:
    return sorted(
        (event.pitch, round(event.start * 22_050), round(event.end * 22_050))
        for event in events
        if event.final
    )


def _assert_close(
    streamed: list[tuple[int, int, int]], offline: list[tuple[int, int, int]]
) -> None:
    """Seconds are rounded for the stream, so allow a sample of slack."""
    assert len(streamed) == len(offline)
    for (stream_pitch, stream_start, stream_end), (pitch, start, end) in zip(
        streamed, offline
    ):
        assert stream_pitch == pitch
        assert abs(stream_start - start) <= 2
        assert abs(stream_end - end) <= 2


def test_v1_stream_matches_finalize() -> None:
    tracks = {
        20: [
            _note(2_100, 4_000, has_onset=True, has_offset=False),
            _note(2_000, 6_000, has_onset=True, has_offset=False),
            _note(4_001, 8_000, has_onset=False, has_offset=False),
            _note(4_500, 9_000, has_onset=False, has_offset=False),
        ],
        31: [
            _note(12_000, 13_000, pitch=60, has_onset=True, has_offset=True),
            _note(20_000, 21_000, pitch=60, has_onset=True, has_offset=True),
        ],
    }
    events, _ = _streamed(tracks, "v1")
    _assert_close(_final_events(events), sorted(_offline(tracks)))


def test_v2_stream_matches_finalize() -> None:
    tracks = {
        20: [
            _note(1_000, 2_000, has_onset=True, has_offset=True),
            _note(5_000, 6_000, has_onset=True, has_offset=False),
            _note(5_100, 7_000, has_onset=False, has_offset=True),
        ],
        31: [
            _note(3_000, 4_000, pitch=55, has_onset=True, has_offset=True),
            _note(9_000, 9_500, pitch=55, has_onset=True, has_offset=True),
        ],
    }
    events, _ = _streamed(tracks, "v2")
    _assert_close(_final_events(events), sorted(_offline(tracks)))


def test_growing_note_keeps_one_id_and_reports_progress() -> None:
    """A note that is extended must keep its id and be re-emitted as it grows."""
    stitcher = _stitcher()
    emitter = _emitter(stitcher, "v2")
    stitcher.notes_by_pair[20] = [
        _note(1_000, 2_000, has_onset=True, has_offset=False)
    ]
    first = emitter.on_window(0)
    open_events = [event for event in first if not event.final]
    assert len(open_events) == 1
    note_id = open_events[0].id
    assert open_events[0].end == round(2_000 / 22_050, 4)

    # The next window extends the same note in place (what consume_window does).
    stitcher.notes_by_pair[20][-1] = _note(
        1_000, 3_400, has_onset=True, has_offset=False
    )
    grown = [event for event in emitter.on_window(4_000) if not event.final]
    assert len(grown) == 1
    assert grown[0].id == note_id
    assert grown[0].end == round(3_400 / 22_050, 4)

    # The boundary head then closes the note; it stays the tail, so nothing is
    # emitted yet.
    stitcher.notes_by_pair[20][-1] = _note(
        1_000, 3_400, has_onset=True, has_offset=True
    )
    assert [event for event in emitter.on_window(6_000) if event.final] == []

    # A following note pushes the closed one out of the tail: it is final now,
    # still under the same id.
    stitcher.notes_by_pair[20].append(
        _note(9_000, 10_000, has_onset=True, has_offset=True)
    )
    final = [event for event in emitter.on_window(12_000) if event.final]
    assert len(final) == 1
    assert final[0].id == note_id
    assert final[0].end == round(3_400 / 22_050, 4)


def test_finish_flushes_the_last_note() -> None:
    stitcher = _stitcher()
    emitter = _emitter(stitcher, "v1")
    stitcher.notes_by_pair[20] = [
        _note(1_000, 2_000, has_onset=True, has_offset=False)
    ]
    emitter.on_window(0)
    events = emitter.finish()
    final = [event for event in events if event.final]
    assert len(final) == 1
    assert final[0].end == round(2_000 / 22_050, 4)


def test_stream_uses_the_notes_own_instrument_candidate() -> None:
    stitcher = _stitcher()
    emitter = _emitter(stitcher, "v2")
    stitcher.notes_by_pair[20] = [
        _note(1_000, 2_000, has_onset=True, has_offset=True, candidates=(7, 3))
    ]
    events = emitter.on_window(0) + emitter.finish()
    assert [event.instrument for event in events if event.final] == ["class-7"]
