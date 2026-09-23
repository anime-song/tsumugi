"""Incremental note emission for the windowed AMT decoders.

``WindowNoteStitcher.finalize`` merges the per-track note list before the offset
filter runs, so a note is only final once no later window can still change it.
This module replays exactly that merge as the windows arrive:

* after every window the track is re-merged with the stitcher's own rule
  (``_merge_nearby_notes``) — the same call ``finalize`` makes;
* every merged note except the last is frozen and emitted as ``final=True``;
* the last one is re-emitted as ``final=False`` while it still has no offset, so
  the piano roll can show it growing;
* ``finish`` mirrors ``finalize``'s tail handling and flushes the rest.

Re-merging a track costs O(notes) per window; the alternative — tracking the
merge by index — cannot express the in-place tail updates the V2 decoder makes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..amt.inference.types import PredictedNote
from ..amt.inference.windowed import WindowNoteStitcher


@dataclass(frozen=True)
class NoteEvent:
    """One note as the frontend should draw it right now."""

    id: str
    instrument: str
    instrument_id: int
    pitch: int
    start: float
    end: float
    velocity: int
    final: bool
    stem: str


class NoteEmitter(Protocol):
    def on_window(self, window_start_frame: int) -> list[NoteEvent]: ...

    def finish(self) -> list[NoteEvent]: ...


@dataclass
class _TrackState:
    emitted: int = 0
    last_open: tuple[int, int] | None = None


class StreamingNoteEmitter:
    """Emits notes from a stitcher as soon as they can no longer change."""

    def __init__(
        self,
        stitcher: WindowNoteStitcher,
        *,
        stem: str,
        sample_rate: int,
        instrument_label: Callable[[int], str],
    ) -> None:
        self.stitcher = stitcher
        self.stem = str(stem)
        self.sample_rate = int(sample_rate)
        self.instrument_label = instrument_label
        self._state: dict[object, _TrackState] = {}

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _instrument_id_of(note: PredictedNote) -> int:
        # Prefer the note's own ranked class over the stem level label so the
        # roll can colour each instrument separately.
        if note.instrument_candidates:
            return int(note.instrument_candidates[0])
        return int(note.instrument_id)

    def _event(
        self, track: object, index: int, note: PredictedNote, *, final: bool
    ) -> NoteEvent:
        instrument_id = self._instrument_id_of(note)
        return NoteEvent(
            id=f"{self.stem}:{track}:{index}",
            instrument=self.instrument_label(instrument_id),
            instrument_id=int(instrument_id),
            pitch=int(note.pitch),
            start=round(int(note.start_sample) / self.sample_rate, 4),
            end=round(int(note.end_sample) / self.sample_rate, 4),
            velocity=int(note.velocity),
            final=bool(final),
            stem=self.stem,
        )

    def _merged(self, track: object) -> list[PredictedNote]:
        notes = self.stitcher.notes_by_pair.get(track) or []
        if not notes:
            return []
        return self.stitcher._merge_nearby_notes(list(notes))

    @staticmethod
    def _open_key(note: PredictedNote) -> tuple[int, int]:
        return (int(note.start_sample), int(note.end_sample))

    # ── public API ───────────────────────────────────────────────────────────
    def on_window(self, window_start_frame: int) -> list[NoteEvent]:
        events: list[NoteEvent] = []
        for track in list(self.stitcher.notes_by_pair):
            merged = self._merged(track)
            if not merged:
                continue
            state = self._state.setdefault(track, _TrackState())
            while state.emitted < len(merged) - 1:
                note = merged[state.emitted]
                index = state.emitted
                state.emitted += 1
                if note.has_offset:
                    events.append(self._event(track, index, note, final=True))
                    state.last_open = None
            tail = merged[-1]
            if tail.has_onset and not tail.has_offset:
                key = self._open_key(tail)
                if state.last_open != key:
                    state.last_open = key
                    events.append(
                        self._event(track, state.emitted, tail, final=False)
                    )
        return events

    def finish(self) -> list[NoteEvent]:
        events: list[NoteEvent] = []
        for track, notes in self.stitcher.notes_by_pair.items():
            if notes:
                # Mirror finalize(): the last segment counts as closed.
                notes[-1].has_offset = True
            merged = self._merged(track)
            state = self._state.setdefault(track, _TrackState())
            while state.emitted < len(merged):
                note = merged[state.emitted]
                index = state.emitted
                state.emitted += 1
                if note.has_offset:
                    events.append(self._event(track, index, note, final=True))
            state.last_open = None
        return events


def make_emitter(
    semi_crf_version: str,
    stitcher: WindowNoteStitcher,
    *,
    stem: str,
    sample_rate: int,
    instrument_label: Callable[[int], str],
    merge_gap_samples: int | None = None,
    merge_onset_samples: int | None = None,
) -> NoteEmitter:
    """Build an emitter for either decode version.

    ``finalize`` applies the same merge to V1 and V2 tracks, so there is a single
    implementation; the version only matters to the decoder itself.
    """
    del semi_crf_version, merge_gap_samples, merge_onset_samples
    return StreamingNoteEmitter(
        stitcher,
        stem=stem,
        sample_rate=sample_rate,
        instrument_label=instrument_label,
    )
