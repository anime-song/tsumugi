"""Streaming (incremental) note output for the windowed AMT decoders.

The decoder hooks (``on_window_consumed``) hand every finished window to a
``NoteEmitter``, which decides which notes can no longer change and yields them
one at a time. Wire it up like this::

    from instrument_agnostic_amt.streaming import make_emitter

    emitter = None

    def on_window(stitcher, window_start_frame):
        global emitter
        if emitter is None:
            emitter = make_emitter(
                config.semi_crf_version,
                stitcher,
                stem="piano",
                sample_rate=config.sample_rate,
                instrument_label=lambda instrument_id: INSTRUMENT_CLASSES[instrument_id],
            )
        for event in emitter.on_window(window_start_frame):
            ...  # push the event to a UI, append to a MIDI chunk, log it, ...

    run_inference(..., on_window_consumed=on_window)
    for event in emitter.finish():
        ...
"""

from .emitters import NoteEmitter, NoteEvent, StreamingNoteEmitter, make_emitter

__all__ = ["NoteEmitter", "NoteEvent", "StreamingNoteEmitter", "make_emitter"]
