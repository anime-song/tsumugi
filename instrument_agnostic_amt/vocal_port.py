# -*- coding: utf-8 -*-
"""Vocal port mode: port 0 keeps the transcribed instruments, port 1 carries the vocals.

The vocals stem is analyzed frame by frame (`vocal_pitch`): f0 plus one harmonic
level curve per frame. The first N harmonics (15 by default) are rendered as
additive synthesis, one MIDI channel per harmonic on port 1, all with a fixed
program (bank 0 / GM program 80, ocarina).

Mapping rules:
- RPN 0,0 sets pitch bend range to +-12 semitones at the start of every channel.
- Pitch bend is written per frame (62.5 fps), so glides and vibrato follow f0.
- Notes start/end only at gated segment boundaries; pitch inside a note is bend only.

Output is a type-1 MIDI file: all original tracks (port 0) plus one vocal track
with a `midi_port` meta event (FF 21) set to port 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mido

from instrument_agnostic_amt.vocal_pitch import analyze_vocals

PROGRAM_DEFAULT = 79       # GM 80 (ocarina)
BANK_MSB_DEFAULT = 0
N_COMPONENTS = 15          # port 1 has 16 channels; ch9 (drums) is skipped
SKIP_CHANNELS = {9}
BEND_RANGE_SEMI = 12.0     # pitch bend range set through RPN
# CC11 gain maps directly to the true harmonic amplitude; no floor, no lift.
# A floor is deliberately avoided: it would raise weak high harmonics far above
# their real level and make the top band louder than the fundamental.
CC_MASTER_GAIN_DB = 0.0    # global level trim; 0.0 = exact measured amplitudes
CC_MASTER_GAIN = 10.0 ** (CC_MASTER_GAIN_DB / 20.0)
ACCOMPANIMENT_CC7 = 70     # channel volume written to every accompaniment track
CC_STEP_MIN = 2            # smallest CC11 change worth writing, about 0.7 dB
CC7_VALUE = 127            # channel volume for the vocal channels
COMP_GATE_DB = 45.0        # a frame is silent when f0 is 45 dB below its own peak
MIN_RUN_FRAMES = 2         # shortest note, in frames

def _sec2tick_factory(mid: mido.MidiFile):
    """Seconds -> ticks using the project's own tempo converter.

    The instrument tracks are written with _TempoConverter, so reusing it keeps
    the vocal track aligned with them instead of drifting.
    """
    from instrument_agnostic_amt.beat_chord.tempo_map_export import _TempoConverter

    events = []
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                events.append((int(t), int(msg.tempo)))
    converter = _TempoConverter.from_events(
        ticks_per_beat=int(mid.ticks_per_beat), tempo_events=events
    )
    return converter.seconds_to_tick


def _build_channel_events(run: dict, ch: int, program: int, bank_msb: int,
                          volume_cc: int = 11) -> list[tuple]:
    """Events of one harmonic channel: [(abs_tick, priority, message), ...]."""
    ev: list[tuple] = []
    ev.append((0, -100, mido.MetaMessage("midi_port", port=1, time=0)))
    ev.append((0, -95, mido.Message("control_change", channel=ch, control=0, value=bank_msb, time=0)))
    ev.append((0, -90, mido.Message("program_change", channel=ch, program=program, time=0)))
    # Full channel volume compensates the attenuation of the CC11 curve.
    ev.append((0, -85, mido.Message("control_change", channel=ch, control=7, value=CC7_VALUE, time=0)))
    # RPN 0,0 -> bend range +-12 semitones
    for cc, val in ((101, 0), (100, 0), (6, int(BEND_RANGE_SEMI)), (38, 0)):
        ev.append((0, -80, mido.Message("control_change", channel=ch, control=cc, value=val, time=0)))
    for seg in run["segs"]:
        start_t, end_t, note, bends, expr = seg
        ev.append((start_t, 10, mido.Message("note_on", channel=ch, note=note, velocity=run["vel"], time=0)))
        for bt, bval in bends:
            ev.append((bt, 20, mido.Message("pitchwheel", channel=ch, pitch=bval, time=0)))
        for et, eval_ in expr:
            ev.append((et, 15, mido.Message("control_change", channel=ch, control=volume_cc, value=eval_, time=0)))
        ev.append((end_t, 5, mido.Message("note_off", channel=ch, note=note, velocity=0, time=0)))
    return ev


def _f0_level_db(track, i: int) -> float:
    """Absolute level of the fundamental (k=1) in frame i."""
    f = float(track.f0[i])
    rel = float(np.interp(np.log2(max(f, 1e-9)), np.log2(track.centers_h),
                          track.harm_band_db[i]))
    return float(track.harm_peak_db[i]) + rel


def _active_frames(track, gate_db: float) -> np.ndarray:
    """Per-frame mask where f0 is voiced and above the gate.

    This is the ONLY place the silence threshold is applied. Every harmonic
    component reuses the same mask, so all channels start and stop on the same
    frames instead of gating independently per frequency band.
    """
    nfr = track.n_frames
    active = np.zeros(nfr, dtype=bool)
    for i in range(nfr):
        if not track.voiced[i] or track.f0[i] <= 0:
            continue
        if _f0_level_db(track, i) >= gate_db:
            active[i] = True
    return active


def _component_segments(track, k: int, sec2tick, active: np.ndarray,
                        ref_db: float) -> list[dict]:
    """Note segments of harmonic k, split only where the f0 gate closes.

    Segment boundaries come from `active` (f0 only) and the amplitude feeds the
    per-frame CC11 curve. CC11 is referenced to the f0 peak (`ref_db`) shared by
    every harmonic, so a weaker harmonic stays quieter than the fundamental.

    Returns {'vel': int, 'segs': [(start_tick, end_tick, note, bends, expr)]}.
    """
    f0 = track.f0
    nfr = track.n_frames
    hop = track.hop_size / track.sample_rate
    f_max = 0.95 * (track.sample_rate / 2.0)

    frames = []
    for i in range(nfr):
        if not active[i]:
            frames.append(None)
            continue
        freq = k * f0[i]
        if freq > f_max:
            frames.append(None)
            continue
        rel_db = float(np.interp(np.log2(freq), np.log2(track.centers_h), track.harm_band_db[i]))
        abs_db = float(track.harm_peak_db[i]) + rel_db
        frames.append((freq, abs_db))

    segs = []
    i = 0
    while i < nfr:
        if frames[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < nfr and frames[j + 1] is not None:
            j += 1
        if j - i + 1 >= MIN_RUN_FRAMES:
            run_frames = frames[i : j + 1]
            med_freq = float(np.median([f for f, _ in run_frames]))
            note = int(round(69 + 12 * np.log2(med_freq / 440.0)))
            note = max(0, min(127, note))
            note_hz = 440.0 * 2 ** ((note - 69) / 12)

            bend_frames = []
            last_bend = None
            for idx in range(i, j + 1):
                dev_semi = 12 * np.log2(frames[idx][0] / note_hz)
                dev_semi = float(np.clip(dev_semi, -BEND_RANGE_SEMI, BEND_RANGE_SEMI))
                bval = int(round(dev_semi / BEND_RANGE_SEMI * 8191))
                bval = max(-8192, min(8191, bval))
                if last_bend is None or abs(bval - last_bend) >= 4:  # ~6 cents
                    bend_frames.append((idx, bval))
                    last_bend = bval

            segs.append({
                "start": sec2tick(i * hop),
                "end": sec2tick((j + 1) * hop),
                "note": note,
                "bends": [(sec2tick(idx * hop), bv) for idx, bv in bend_frames],
                "frames": [(idx, db) for idx, (_, db) in zip(range(i, j + 1), run_frames)],
            })
        i = j + 1

    if not segs:
        return {}

    # Velocity stays at 127; loudness is carried by the CC11 curve alone.
    out_segs = []
    last_val = None
    for sg in segs:
        expr_t = []
        for idx, db in sg["frames"]:
            # Linear amplitude of this harmonic relative to the shared f0 peak,
            # with a single global gain. No per-channel floor or normalisation,
            # so the measured harmonic balance is preserved.
            ratio = (10.0 ** ((db - ref_db) / 20.0)) * CC_MASTER_GAIN
            # GM2 curve: amplitude = (cc / 127) ** 2, so cc = 127 * sqrt(amp).
            # The sqrt is the physical inverse of the synth curve, not shaping.
            val = int(np.clip(round(127.0 * np.sqrt(ratio)), 0, 127))
            # The curve runs across segment boundaries, so a change below the
            # audible step is dropped instead of being rewritten every frame.
            if last_val is None or abs(val - last_val) >= CC_STEP_MIN:
                expr_t.append((sec2tick(idx * hop), val))
                last_val = val
        out_segs.append((sg["start"], sg["end"], sg["note"], sg["bends"], expr_t))
    return {"vel": 127, "segs": out_segs}


def _set_instrument_cc7(mid: mido.MidiFile, value: int) -> list[str]:
    """Set CC7 of every accompaniment track to `value`.

    This is a channel-level volume trim only: note velocities (which shape the
    timbre/brightness) are left untouched. Tracks without an existing CC7 get one
    inserted at tick 0. Every accompaniment track is treated alike.
    """
    changed: list[str] = []
    for tr in mid.tracks:
        name = (tr.name or "").strip().lower()
        ch = next((m.channel for m in tr if hasattr(m, "channel")), None)
        if ch is None:
            continue
        cc7 = [m for m in tr if m.type == "control_change" and m.control == 7]
        if cc7:
            for m in cc7:
                m.value = int(value)
        else:
            tr.insert(0, mido.Message("control_change", channel=ch, control=7,
                                      value=int(value), time=0))
        changed.append(name or f"ch{ch}")
    return changed


def add_vocal_port(
    midi_path: str | Path,
    vocals_wav: str | Path,
    out_path: str | Path | None = None,
    n_components: int = N_COMPONENTS,
    program: int = PROGRAM_DEFAULT,
    bank_msb: int = BANK_MSB_DEFAULT,
    volume_cc: int = 11,
    accompaniment_cc7: int | None = ACCOMPANIMENT_CC7,
    melody_midi: str | Path | None = None,
    guide_threshold_st: float = 2.0,
    neg_tolerance_st: float = 2.0,
) -> Path:
    """Append a port-1 vocal track to midi_path and write out_path (*_vocalport.mid).

    If ``melody_midi`` is given, its melody (positive) and vocal_harmony
    (negative) tracks disambiguate f0 on harmony passages. Omit it for blind f0.
    """
    midi_path = Path(midi_path)
    if out_path is None:
        out_path = midi_path.with_name(midi_path.stem + "_vocalport.mid")
    out_path = Path(out_path)

    print(f"[vocal-port] analyzing {vocals_wav} ...")
    track = analyze_vocals(vocals_wav, melody_midi=melody_midi,
                           guide_threshold_st=guide_threshold_st,
                           neg_tolerance_st=neg_tolerance_st)
    nfr = track.n_frames
    hop = track.hop_size / track.sample_rate
    print(f"[vocal-port] {nfr} frames ({nfr * hop:.1f}s)")

    mid = mido.MidiFile(str(midi_path))
    sec2tick = _sec2tick_factory(mid)

    # Channel-level volume trim on accompaniment only. Note velocities are not
    # touched, so timbre/brightness is preserved. Runs before the voice track is
    # appended; the port's own CC7 stays at CC7_VALUE.
    if accompaniment_cc7 is not None:
        changed = _set_instrument_cc7(mid, accompaniment_cc7)
        if changed:
            print(f"[vocal-port] set CC7={accompaniment_cc7} on {len(changed)} "
                  f"accompaniment tracks: {changed}")

    # f0 alone decides which frames are active; every harmonic shares the mask
    f0_levels = [_f0_level_db(track, i) for i in range(nfr)
                 if track.voiced[i] and track.f0[i] > 0]
    peak = float(np.max(f0_levels)) if f0_levels else 0.0
    gate_db = peak - COMP_GATE_DB
    active = _active_frames(track, gate_db)
    print(f"[vocal-port] f0 gate {gate_db:.1f} dB, {int(active.sum())}/{nfr} frames active")

    channels = [c for c in range(16) if c not in SKIP_CHANNELS][:n_components]
    voc_track = mido.MidiTrack()
    all_events: list[tuple] = []
    used = 0
    for ci, ch in enumerate(channels):
        k = ci + 1  # harmonic index
        run = _component_segments(track, k, sec2tick, active, peak)
        if not run:
            continue
        used += 1
        all_events.extend(_build_channel_events(run, ch, program, bank_msb, volume_cc))
        print(f"[vocal-port] ch{ch}: h{k} -> {len(run['segs'])} notes, vel={run['vel']}")
    if used == 0:
        raise RuntimeError("no voiced components found; vocal stem may be silent")

    # Sort by absolute tick, then encode as delta times
    all_events.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for abs_t, _, msg in all_events:
        msg.time = max(0, abs_t - last)
        last = abs_t
        voc_track.append(msg)
    voc_track.append(mido.MetaMessage("end_of_track", time=0))

    out = mido.MidiFile(type=1, ticks_per_beat=mid.ticks_per_beat)
    for tr in mid.tracks:
        out.tracks.append(tr)
    out.tracks.append(voc_track)
    out.save(str(out_path))
    print(f"[vocal-port] wrote {out_path} ({used} component channels)")
    return out_path
