# -*- coding: utf-8 -*-
"""Frame-wise f0 and harmonic amplitude estimation for the vocal port.

Each frame of the vocals stem goes through:

1. YIN on a 64 ms evidence window, which gives a periodicity curve.
2. A peak-based spectral estimate, which decides the octave. When a melody /
   harmony reference is supplied, strong peaks are scored against it so f0 is
   seeded on the melody pitch rather than a harmony note or wrong octave.
3. The two are multiplied and normalized to a candidate grid of 8 cents steps.
4. Viterbi tracking over the whole stem with a smoothness prior, so the chosen
   f0 follows the previous frame instead of jumping between candidates.
5. Parabolic interpolation on the harmonic peaks refines f0 to sub-cent
   accuracy, followed by a 5-frame median filter inside voiced runs.

Voicing is decided by the spectral flatness of the 200-4000 Hz band, with a
frame-level silence gate. For voiced frames the harmonic amplitudes are solved
by joint least squares over a window of four periods, then fitted onto a
24-band log-frequency grid to give one relative dB curve plus one absolute peak
level per frame.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 24000
WIN_SIZE = 768
HOP_SIZE = 384
F0_MIN = 60.0
F0_MAX = 600.0
N_CAND_STEP_CENTS = 8.0
SMOOTH_COST = 20.0
OCTAVE_COST = 0.6
N_HARM_BANDS = 24
HARM_BAND_LO = 70.0
HARM_FLOOR_DB = -72.0
SFM_BAND_LO = 200.0
SFM_BAND_HI = 4000.0
SFM_VOICED_MAX = 0.20
SILENCE_DBFS = -60.0
ACTIVE_RANGE_DB = 45.0
PROJ_PERIODS = 4.0
PROJ_MIN = 128
PROJ_MAX = 2048
MAX_HARMONICS = 200
PITCH_CURVE_DEG = 2
PITCH_CURVE_MAX_CENTS = 35.0
PITCH_CURVE_MIN_LEN = 4
PITCH_CURVE_BLEND = 2


class VocalTrack:
    """Per-frame vocal parameters: f0, voicing, harmonic level curve."""

    def __init__(self, voiced, f0, harm_peak_db, harm_band_db, centers_h,
                 hop_size=HOP_SIZE, sample_rate=SAMPLE_RATE):
        self.voiced = voiced
        self.f0 = f0
        self.harm_peak_db = harm_peak_db
        self.harm_band_db = harm_band_db
        self.centers_h = centers_h
        self.hop_size = hop_size
        self.sample_rate = sample_rate

    @property
    def n_frames(self) -> int:
        return int(self.voiced.size)

    @property
    def hop_seconds(self) -> float:
        return self.hop_size / float(self.sample_rate)


def _vorbis_window(n: int) -> np.ndarray:
    k = np.arange(n)
    return np.sin((np.pi / 2.0) * np.sin(np.pi * k / n) ** 2)


def _frame_count(n_samples: int, win: int, hop: int) -> int:
    if n_samples <= 0:
        return 0
    if n_samples <= win:
        return 1
    return int(np.ceil((n_samples - win) / hop)) + 1


def _frame(x: np.ndarray, i: int, win: int, hop: int) -> np.ndarray:
    seg = x[i * hop:i * hop + win]
    if seg.size == win:
        return seg
    out = np.zeros(win)
    out[:seg.size] = seg
    return out


def _window_segment(x: np.ndarray, center: int, length: int):
    """Fixed-length window around center, shifted inside the signal (no zero pad)."""
    n = x.size
    if length >= n:
        seg = np.zeros(length)
        seg[:n] = x
        return seg, 0
    start = int(np.clip(center - length // 2, 0, n - length))
    return x[start:start + length].copy(), start


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -np.inf
    return 20.0 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-20)


def _spectral_flatness(x: np.ndarray, fs: float) -> float:
    """Geometric over arithmetic mean of the vocal-band power spectrum."""
    if x.size < 32 or not np.any(x):
        return 1.0
    power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    sel = (freqs >= SFM_BAND_LO) & (freqs <= SFM_BAND_HI)
    if int(sel.sum()) < 8:
        return 1.0
    p = power[sel]
    p = np.maximum(p, float(p.max()) * 1e-9 + 1e-30)
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def _candidates(f_min: float, f_max: float, step_cents: float) -> np.ndarray:
    n = int(np.ceil(1200.0 * np.log2(f_max / f_min) / step_cents)) + 1
    return f_min * 2.0 ** (np.arange(n) * step_cents / 1200.0)


def _cumulative_mean_norm_diff(x: np.ndarray, tau_max: int) -> np.ndarray:
    """Normalized difference function, computed through FFT cross-correlation."""
    n = x.size
    span = n - tau_max
    head = x[:span]
    nfft = 1 << (n + span - 1).bit_length()
    cross = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(head, nfft)), nfft)
    cross = cross[:tau_max + 1]
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    tau = np.arange(tau_max + 1)
    diff = (cs[span] + (cs[span + tau] - cs[tau])) - 2.0 * cross
    diff[0] = 0.0
    cmnd = np.ones(tau_max + 1)
    running = np.cumsum(diff[1:])
    nz = running > 0
    cmnd[1:][nz] = diff[1:][nz] * np.arange(1, tau_max + 1)[nz] / running[nz]
    return cmnd


def _yin_evidence(x: np.ndarray, fs: float, candidates: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    tau_max = min(n - 2, int(np.ceil(fs / float(candidates.min()))) + 1)
    tau_min = max(2, int(np.floor(fs / float(candidates.max()))))
    if tau_max <= tau_min + 1 or n - tau_max < 8:
        return np.zeros(candidates.size)
    cmnd = _cumulative_mean_norm_diff(x, tau_max)
    lags = np.clip(fs / np.maximum(candidates, 1e-6), 1, tau_max)
    ev = 1.0 - np.interp(lags, np.arange(tau_max + 1), cmnd)
    return np.clip(ev, 0.0, 1.0)


def _dft_evidence(x: np.ndarray, fs: float, candidates: np.ndarray,
                  band_lo: float = 60.0, band_hi: float = 1000.0,
                  prom_db: float = 8.0, prom_look: float = 0.25,
                  sigma_oct: float = 0.12, octave_rel_db: float = 12.0,
                  strong_rel_db: float = 12.0,
                  guide_hz: float = 0.0, neg_guide_hz: float = 0.0,
                  attract_st: float = 2.0, repel_st: float = 2.0,
                  attract_db: float = 60.0, repel_db: float = 60.0):
    """Evidence around a strong spectral peak, with octave fallback.

    Blind mode (no guide/neg_guide): the lowest strong peak wins, then folds
    down in octaves while a strong sub-octave peak exists.

    Guided mode: every strong peak (plus sub-octave candidates) is scored by its
    prominence PLUS a Gaussian attract field around the melody note MINUS a
    Gaussian repel field around the harmony note. The peak nearest melody and
    away from harmony wins even when it is not the loudest.
    """
    if x.size < 32 or not np.any(x):
        return np.zeros(candidates.size), 0.0
    n_fft = 1 << int(np.ceil(np.log2(max(x.size * 8, 4096))))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size), n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    db = 20.0 * np.log10(spec + 1e-30)
    df = freqs[1] - freqs[0]

    def peak_prom(f0: float) -> float:
        if f0 < band_lo or f0 > band_hi:
            return -np.inf
        r = max(prom_look * f0, 2.0 * df)
        lo = int(np.searchsorted(freqs, f0 - r))
        hi = int(np.searchsorted(freqs, f0 + r))
        if hi - lo < 3:
            return -np.inf
        k = lo + int(np.argmax(spec[lo:hi]))
        if k <= 0 or k >= spec.size - 1:
            return -np.inf
        if not (spec[k] > spec[k - 1] and spec[k] >= spec[k + 1]):
            return -np.inf
        w = max(int(r / df), 2)
        bg = float(np.median(db[max(k - 4 * w, 0):min(k + 4 * w + 1, db.size)]))
        return float(db[k] - bg)

    idx = np.nonzero((freqs >= band_lo) & (freqs <= band_hi))[0]
    if idx.size == 0:
        return np.zeros(candidates.size), 0.0
    vmax = float(db[idx].max())

    # All strong peaks, in ascending frequency order (idx is sorted).
    raw: list[tuple[float, float]] = []  # (freq_hz, prominence_db)
    for k in idx:
        if k <= 0 or k >= spec.size - 1:
            continue
        if not (spec[k] > spec[k - 1] and spec[k] >= spec[k + 1]):
            continue
        pr = peak_prom(float(freqs[k]))
        if pr >= prom_db and db[k] >= vmax - strong_rel_db:
            raw.append((float(freqs[k]), pr))
    if not raw:
        return np.zeros(candidates.size), 0.0

    guided = guide_hz > 0 or neg_guide_hz > 0
    if not guided:
        # Blind: lowest strong peak, then fold down while a strong sub-octave exists.
        c = raw[0][0]
        while c / 2.0 >= band_lo:
            p_cur = peak_prom(c)
            p_half = peak_prom(c / 2.0)
            if p_half < prom_db or p_half < p_cur - octave_rel_db:
                break
            c /= 2.0
    else:
        # Guided: score every peak and its sub-octave candidates.
        pool: list[tuple[float, float]] = []  # (freq_hz, prominence_db)
        for peak, pr in raw:
            cur, cur_pr = peak, pr
            pool.append((cur, cur_pr))
            while cur / 2.0 >= band_lo:
                p_half = peak_prom(cur / 2.0)
                if p_half < prom_db or p_half < peak_prom(cur) - octave_rel_db:
                    break
                cur /= 2.0
                pool.append((cur, p_half))

        def score(freq: float, pr: float) -> float:
            s = pr  # loudness / prominence is the base evidence
            if guide_hz > 0:  # positive sample: pull peaks onto melody
                d_mel = 12.0 * np.log2(max(freq, 1e-9) / guide_hz)
                s += attract_db * np.exp(-0.5 * (d_mel / max(attract_st, 1e-3)) ** 2)
            if neg_guide_hz > 0:  # negative sample: push peaks off harmony
                d_harm = 12.0 * np.log2(max(freq, 1e-9) / neg_guide_hz)
                s -= repel_db * np.exp(-0.5 * (d_harm / max(repel_st, 1e-3)) ** 2)
            return s

        c = max(pool, key=lambda item: score(item[0], item[1]))[0]

    d = np.log2(np.maximum(candidates, 1e-9) / c)
    return np.exp(-0.5 * (d / max(sigma_oct, 1e-3)) ** 2), c


def _combine(ev_yin: np.ndarray, ev_dft: np.ndarray) -> np.ndarray:
    ev = ev_yin * ev_dft
    if float(np.max(ev)) <= 0.0:
        return np.clip(ev_yin, 0.0, 1.0)
    return ev


def _track_pitch(evidence: np.ndarray, candidates: np.ndarray, valid: np.ndarray,
                 voiced_prior: np.ndarray):
    """Viterbi over the candidate grid plus one unvoiced state."""
    nfr, nc = evidence.shape
    log_c = np.log2(candidates)
    n_states = nc + 1

    d = log_c[:, None] - log_c[None, :]
    trans = np.zeros((n_states, n_states))
    trans[:nc, :nc] = SMOOTH_COST * d ** 2
    trans[:nc, :nc] += OCTAVE_COST * np.exp(-((np.abs(d) - 1.0) ** 2) / 0.02)
    trans[:nc, nc] = 0.35
    trans[nc, :nc] = 0.35

    def emission(t: int) -> np.ndarray:
        e = np.clip(evidence[t], 1e-6, 1.0)
        out = np.empty(n_states)
        out[:nc] = -np.log(e)
        out[nc] = -np.log(max(1.0 - float(np.max(evidence[t])), 1e-3))
        pp = float(np.clip(voiced_prior[t], 1e-3, 1.0 - 1e-3))
        out[:nc] += -np.log(pp)
        out[nc] += -np.log(1.0 - pp)
        if not valid[t]:
            out[:nc] += 1e3
        return out

    back = np.zeros((nfr, n_states), dtype=np.int32)
    cost = emission(0)
    for t in range(1, nfr):
        tot = cost[:, None] + trans
        back[t] = np.argmin(tot, axis=0)
        cost = emission(t) + tot[back[t], np.arange(n_states)]

    states = np.zeros(nfr, dtype=np.int32)
    states[-1] = int(np.argmin(cost))
    for t in range(nfr - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    voiced = states < nc
    f0 = np.zeros(nfr)
    f0[voiced] = candidates[states[voiced]]
    return f0, voiced


def _refine_f0(x: np.ndarray, fs: float, f0_coarse: float, max_harm: int = 16,
               search_rel: float = 0.35, min_rel_db: float = -45.0,
               max_cents: float = 120.0):
    """Amplitude-weighted least squares over interpolated harmonic peaks."""
    if f0_coarse <= 0 or x.size < 32 or not np.any(x):
        return float(f0_coarse), 0
    n_fft = 1 << int(np.ceil(np.log2(max(x.size * 8, 2048))))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size), n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    df = freqs[1] - freqs[0]
    thresh = (float(spec.max()) + 1e-30) * 10.0 ** (min_rel_db / 20.0)

    num = den = 0.0
    used = 0
    for h in range(1, max_harm + 1):
        c = h * f0_coarse
        if c >= 0.95 * fs / 2.0:
            break
        r = max(search_rel * f0_coarse, 2.0 * df)
        lo = int(np.searchsorted(freqs, c - r))
        hi = int(np.searchsorted(freqs, c + r))
        if hi - lo < 3:
            continue
        k = lo + int(np.argmax(spec[lo:hi]))
        if k <= 0 or k >= spec.size - 1 or spec[k] < thresh:
            continue
        y0 = 20 * np.log10(spec[k - 1] + 1e-30)
        y1 = 20 * np.log10(spec[k] + 1e-30)
        y2 = 20 * np.log10(spec[k + 1] + 1e-30)
        denom = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
        shift = float(np.clip(shift, -0.5, 0.5))
        f_h = float(freqs[k] + shift * df)
        w = float(spec[k]) * h
        num += w * h * f_h
        den += w * h * h
        used += 1
    if den <= 0 or used < 2:
        return float(f0_coarse), used
    f0_ref = num / den
    if abs(1200.0 * np.log2(max(f0_ref, 1e-9) / f0_coarse)) > max_cents:
        return float(f0_coarse), used
    return float(f0_ref), used


def _median_filter(f0: np.ndarray, voiced: np.ndarray, width: int = 5) -> np.ndarray:
    """Median filter in log2 domain, kept inside each voiced run."""
    out = f0.copy()
    half = width // 2
    lf = np.log2(np.maximum(f0, 1e-9))
    idx = np.nonzero(voiced & (f0 > 0))[0]
    if idx.size == 0:
        return out
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [idx.size - 1]])
    for s, e in zip(starts, ends):
        run = idx[s:e + 1]
        vals = lf[run]
        if vals.size < 3:
            continue
        for k, i in enumerate(run):
            lo, hi = max(0, k - half), min(vals.size, k + half + 1)
            out[i] = 2.0 ** float(np.median(vals[lo:hi]))
    return out


def fit_pitch_curve(f0: np.ndarray, voiced: np.ndarray, hop_seconds: float,
                    deg: int = PITCH_CURVE_DEG,
                    max_cents: float = PITCH_CURVE_MAX_CENTS,
                    min_len: int = PITCH_CURVE_MIN_LEN) -> list[dict]:
    """Fit the log2(f0) track with piecewise polynomials.

    Segments are split recursively until the fit stays within `max_cents`, so the
    curve is smooth where the pitch is steady and still follows vibrato and
    glides. Returns [{'start', 'count', 'coef', 'span'}], with `coef` applied to
    u = (t - t0) / span.
    """
    segs: list[dict] = []
    idx = np.nonzero(voiced)[0]
    if idx.size == 0:
        return segs
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [idx.size - 1]])

    def add(a: int, b: int) -> None:
        n = b - a + 1
        t = (idx[a:b + 1] - idx[a]) * hop_seconds
        y = np.log2(np.maximum(f0[idx[a:b + 1]], 1e-6))
        d = min(deg, max(0, n - 1))
        span = float(max(t[-1] - t[0], 1e-9)) if n > 1 else 1e-9
        u = (t - t[0]) / span if n > 1 else np.zeros(1)
        coef = np.polyfit(u, y, d) if n > d else np.array([float(y.mean())])
        err = np.polyval(coef, u) - y
        if n <= min_len or float(np.max(np.abs(err))) * 1200.0 <= max_cents:
            segs.append({"start": int(idx[a]), "count": int(n),
                         "coef": coef, "span": span})
            return
        mid = (a + b) // 2
        add(a, mid)
        add(mid + 1, b)

    for s, e in zip(starts, ends):
        add(int(s), int(e))
    segs.sort(key=lambda s: s["start"])
    return segs


def eval_pitch_curve(segs: list[dict], n_frames: int, hop_seconds: float,
                     blend: int = PITCH_CURVE_BLEND) -> np.ndarray:
    """Rebuild per-frame f0 (Hz) from the fitted segments.

    Neighbouring segments are cross-faded over a couple of frames, otherwise the
    join shows up as a step in the pitch bend.
    """
    n_frames = int(n_frames)
    f0 = np.zeros(n_frames)
    val = np.zeros(n_frames)
    for s in segs:
        a, n = int(s["start"]), int(s["count"])
        if n <= 0:
            continue
        if n == 1:
            f0[a] = 2.0 ** float(np.polyval(s["coef"], 0.0))
            val[a] = 1.0
            continue
        t = np.arange(a, a + n) * hop_seconds
        u = (t - t[0]) / s["span"]
        f0[a:a + n] = 2.0 ** np.polyval(s["coef"], u)
        val[a:a + n] = 1.0

    for s in segs[1:]:
        a = int(s["start"])
        for j in range(max(0, a - blend), min(n_frames, a + blend)):
            w = (j - (a - blend)) / max(2 * blend, 1)
            prev = None
            for sp in segs:
                b = int(sp["start"])
                if b <= j < b + int(sp["count"]) and b < a:
                    if int(sp["count"]) > 1:
                        t = j * hop_seconds - b * hop_seconds
                        prev = 2.0 ** float(np.polyval(sp["coef"], t / sp["span"]))
                    else:
                        prev = 2.0 ** float(np.polyval(sp["coef"], 0.0))
            if prev is not None and val[j] > 0:
                f0[j] = (1.0 - w) * prev + w * f0[j]
    return f0


def _median_smooth(f0: np.ndarray, voiced: np.ndarray, width: int = 5) -> np.ndarray:
    """Replace frames that sit more than 4 semitones away from the local median."""
    out = f0.copy()
    half = width // 2
    n = f0.size
    for i in range(n):
        if not voiced[i]:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        sel = np.arange(lo, hi)[voiced[lo:hi]]
        if sel.size >= 3:
            med = float(np.median(f0[sel]))
            if abs(12.0 * np.log2(max(out[i], 1e-6) / max(med, 1e-6))) > 4.0:
                out[i] = med
    return out


def _project_harmonics_joint(x: np.ndarray, w: np.ndarray, fs: float,
                             freqs: np.ndarray, band: int = 4,
                             ridge: float = 1e-8):
    """Solve all harmonic cos/sin amplitudes together to limit leakage between them."""
    n = x.size
    nh = int(freqs.size)
    if nh == 0:
        return np.zeros(0)
    idx = np.arange(n)
    ang = 2.0 * np.pi * np.outer(freqs, idx) / fs
    cos_m = np.cos(ang)
    sin_m = np.sin(ang)
    xw = x * w
    wcos = w[None, :] * cos_m
    wsin = w[None, :] * sin_m

    w2cos = (w * w)[None, :] * cos_m
    w2sin = (w * w)[None, :] * sin_m
    g = np.zeros((2 * nh, 2 * nh))
    cc0 = np.sum(w2cos * cos_m, axis=1)
    ss0 = np.sum(w2sin * sin_m, axis=1)
    cs0 = np.sum(w2cos * sin_m, axis=1)
    g[0::2, 0::2] = np.diag(cc0)
    g[1::2, 1::2] = np.diag(ss0)
    g[0::2, 1::2] = np.diag(cs0)
    g[1::2, 0::2] = np.diag(cs0)
    for dd in range(1, min(band, nh - 1) + 1):
        cc = np.sum(w2cos[:nh - dd] * cos_m[dd:], axis=1)
        ss = np.sum(w2sin[:nh - dd] * sin_m[dd:], axis=1)
        cs = np.sum(w2cos[:nh - dd] * sin_m[dd:], axis=1)
        sc = np.sum(w2sin[:nh - dd] * cos_m[dd:], axis=1)
        for h in range(nh - dd):
            hp = h + dd
            g[2 * h, 2 * hp] = g[2 * hp, 2 * h] = cc[h]
            g[2 * h + 1, 2 * hp + 1] = g[2 * hp + 1, 2 * h + 1] = ss[h]
            g[2 * h, 2 * hp + 1] = g[2 * hp + 1, 2 * h] = sc[h]
            g[2 * h + 1, 2 * hp] = g[2 * hp, 2 * h + 1] = cs[h]

    y = np.empty(2 * nh)
    y[0::2] = wcos @ xw
    y[1::2] = wsin @ xw
    g[0::2, 0::2] += ridge * np.trace(g) / max(2 * nh, 1)
    try:
        theta = np.linalg.solve(g, y)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(g, y, rcond=None)[0]
    return np.hypot(theta[0::2], theta[1::2])


def _fit_envelope(freqs: np.ndarray, amps: np.ndarray, centers: np.ndarray,
                  floor_db: float = HARM_FLOOR_DB):
    """Fit harmonic amplitudes onto the log band grid; returns (band_db, peak_db)."""
    band_db = np.full(centers.size, floor_db)
    good = np.isfinite(amps) & (amps > 0) & np.isfinite(freqs) & (freqs > 0)
    if not np.any(good):
        return band_db, floor_db
    f_h = freqs[good]
    db_abs = 20.0 * np.log10(amps[good] + 1e-30)
    peak_db = float(db_abs.max())
    db_rel = db_abs - peak_db
    order = np.argsort(f_h)
    band_db = np.interp(np.log(centers), np.log(f_h[order]), db_rel[order],
                        left=float(db_rel[order][0]), right=floor_db)
    band_db = np.clip(band_db, floor_db, 0.0)
    return band_db, peak_db


def _eval_envelope(freqs: np.ndarray, band_db: np.ndarray,
                   centers: np.ndarray) -> np.ndarray:
    return np.interp(np.log(np.maximum(freqs, 1e-9)), np.log(centers), band_db,
                     left=float(band_db[0]), right=float(band_db[-1]))


def _proj_length(f0: float) -> int:
    length = int(round(PROJ_PERIODS * SAMPLE_RATE / max(f0, 1e-6)))
    length = int(np.clip(length, PROJ_MIN, PROJ_MAX))
    return max(4, (length // 4) * 4)


def analyze_array(x: np.ndarray, sample_rate: int = SAMPLE_RATE,
                  guide: np.ndarray | None = None,
                  neg_guide: np.ndarray | None = None,
                  guide_threshold_st: float = 2.0,
                  neg_tolerance_st: float = 2.0) -> VocalTrack:
    """Analyze one mono waveform and return per-frame vocal parameters."""
    x = np.asarray(x, dtype=np.float64)
    n_samples = x.size
    win, hop = WIN_SIZE, HOP_SIZE
    nfr = _frame_count(n_samples, win, hop)
    if nfr == 0:
        return VocalTrack(np.zeros(0, dtype=bool), np.zeros(0),
                          np.zeros(0), np.zeros((0, N_HARM_BANDS)),
                          np.geomspace(HARM_BAND_LO, 0.94 * sample_rate / 2.0,
                                       N_HARM_BANDS), hop, sample_rate)

    centers = np.geomspace(HARM_BAND_LO, 0.94 * sample_rate / 2.0, N_HARM_BANDS)
    rms_all = np.array([_rms_db(_frame(x, i, win, hop)) for i in range(nfr)])
    active_floor = max(SILENCE_DBFS,
                       float(np.percentile(rms_all, 95)) - ACTIVE_RANGE_DB)
    valid = rms_all >= active_floor

    cand = _candidates(F0_MIN, F0_MAX, N_CAND_STEP_CENTS)
    ev_win = max(1536, win)
    evidence = np.zeros((nfr, cand.size))
    voiced_prior = np.zeros(nfr)
    peak_c = np.zeros(nfr)
    has_ref = guide is not None or neg_guide is not None
    for i in range(nfr):
        if not valid[i]:
            continue
        seg, _ = _window_segment(x, i * hop + win // 2, ev_win)
        sfm = _spectral_flatness(seg, sample_rate)
        voiced_prior[i] = float(np.clip((SFM_VOICED_MAX - sfm) / 0.15, 0.0, 1.0))
        ev_y = _yin_evidence(seg, sample_rate, cand)
        if has_ref:
            g_hz = float(guide[i]) if guide is not None else 0.0
            ng_hz = float(neg_guide[i]) if neg_guide is not None else 0.0
            ev_d, c = _dft_evidence(seg, sample_rate, cand,
                                    guide_hz=g_hz, neg_guide_hz=ng_hz,
                                    attract_st=guide_threshold_st,
                                    repel_st=neg_tolerance_st)
        else:
            ev_d, c = _dft_evidence(seg, sample_rate, cand)
        peak_c[i] = c
        ev = _combine(ev_y, ev_d)
        m = float(np.max(ev)) if ev.size else 0.0
        evidence[i] = ev / m if m > 0 else ev

    f0_pred, voiced_pred = _track_pitch(evidence, cand, valid, voiced_prior)
    f0_refined = f0_pred.copy()
    for i in np.nonzero(voiced_pred)[0]:
        seg, _ = _window_segment(x, i * hop + win // 2, ev_win)
        coarse = peak_c[i] if peak_c[i] > 0 else f0_pred[i]
        r, used = _refine_f0(seg, sample_rate, coarse)
        if r > 0 and used >= 2:
            f0_refined[i] = r
    f0_refined = _median_filter(f0_refined, voiced_pred, 5)

    voiced = np.zeros(nfr, dtype=bool)
    f0_track = np.zeros(nfr)
    peak_db = np.full(nfr, -100.0)
    harm_band = np.full((nfr, centers.size), HARM_FLOOR_DB)

    for i in range(nfr):
        if not (voiced_pred[i] and valid[i]) or f0_refined[i] <= 0:
            continue
        seg, _ = _window_segment(x, i * hop + win // 2, ev_win)
        sfm = _spectral_flatness(seg, sample_rate)
        if sfm > SFM_VOICED_MAX:
            continue
        f0_i = float(f0_refined[i])
        hmax = min(MAX_HARMONICS, int(np.floor(0.95 * sample_rate / 2.0 / f0_i)))
        if hmax < 1:
            continue
        freqs = np.arange(1, hmax + 1) * f0_i
        proj_len = _proj_length(f0_i)
        proj_seg, _ = _window_segment(x, i * hop + win // 2, proj_len)
        amps = _project_harmonics_joint(proj_seg, _vorbis_window(proj_len),
                                        sample_rate, freqs)
        band_db, peak = _fit_envelope(freqs, amps, centers)
        # The band fit averages over peaks, so it reads slightly below the
        # projected amplitudes. Fold the energy ratio back into the peak level.
        a_fit = 10.0 ** ((_eval_envelope(freqs, band_db, centers) + peak) / 20.0)
        e_proj = float(np.sum(amps ** 2))
        e_fit = float(np.sum(a_fit ** 2))
        if e_fit > 0 and e_proj > 0:
            peak += float(10.0 * np.log10(e_proj / e_fit))
        voiced[i] = True
        f0_track[i] = f0_i
        harm_band[i] = band_db
        peak_db[i] = peak

    f0_track = _median_smooth(f0_track, voiced, 5)
    curve = eval_pitch_curve(fit_pitch_curve(f0_track, voiced, hop / sample_rate),
                             nfr, hop / sample_rate)
    f0_track = np.where(curve > 0, curve, f0_track)
    return VocalTrack(voiced, f0_track, peak_db, harm_band, centers, hop,
                      sample_rate)


def melody_guide_from_midi(midi_path, n_frames: int, hop_seconds: float,
                           track_name: str = "melody") -> np.ndarray:
    """Build a per-frame pitch guide (Hz, 0 = none) from a named MIDI track.

    Note pitches are converted to Hz and held for each note's duration, as a
    positive (melody) or negative (vocal_harmony) reference for peak selection.
    Frames without a covering note stay 0.
    """
    import mido

    guide = np.zeros(n_frames)
    mid = mido.MidiFile(str(midi_path))
    tempo = 500000  # default 120 BPM
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo = msg.tempo
            break
    tick2sec = tempo / 1e6 / mid.ticks_per_beat

    mel_tr = None
    for tr in mid.tracks:
        if (tr.name or "").strip().lower() == track_name.lower():
            mel_tr = tr
            break
    if mel_tr is None:
        return guide

    notes: list[list[float]] = []  # [pitch, start_sec, end_sec]
    t = 0
    for msg in mel_tr:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            notes.append([float(msg.note), t * tick2sec, -1.0])
        elif (msg.type == "note_off" or
              (msg.type == "note_on" and msg.velocity == 0)):
            for n in reversed(notes):
                if n[2] < 0:
                    n[2] = t * tick2sec
                    break
    for n in notes:
        if n[2] < 0:
            n[2] = n[1] + 0.2
    if not notes:
        return guide

    for i in range(n_frames):
        tc = (i + 0.5) * hop_seconds
        for pit, a, b in notes:
            if a <= tc < b:
                guide[i] = 440.0 * 2.0 ** ((pit - 69.0) / 12.0)
                break
    return guide


def analyze_vocals(wav_path, melody_midi=None,
                   guide_threshold_st: float = 2.0,
                   neg_tolerance_st: float = 2.0) -> VocalTrack:
    """Load a wav (any rate, any channel count) and analyze the mono mix.

    If ``melody_midi`` is given, its melody track is a positive reference and
    its vocal_harmony track a negative reference. They are applied at the
    spectral peak-selection stage: among the strong peaks, ones close to melody
    are preferred and ones close to harmony are penalised, so f0 is seeded on
    the right peak from the start. No post-hoc snap is applied to the tracked
    f0, so vibrato and fine pitch movement are preserved. Omit it (default) for
    fully blind f0.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    x = data.mean(axis=1).astype(np.float64)
    if sr != SAMPLE_RATE:
        from math import gcd
        g = gcd(int(sr), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, int(sr) // g)
    guide = None
    neg_guide = None
    if melody_midi is not None:
        win, hop = WIN_SIZE, HOP_SIZE
        nfr = _frame_count(x.size, win, hop)
        hop_s = hop / float(SAMPLE_RATE)
        guide = melody_guide_from_midi(melody_midi, nfr, hop_s,
                                       track_name="melody")
        neg_guide = melody_guide_from_midi(melody_midi, nfr, hop_s,
                                           track_name="vocal_harmony")
    return analyze_array(x, SAMPLE_RATE, guide=guide, neg_guide=neg_guide,
                         guide_threshold_st=guide_threshold_st,
                         neg_tolerance_st=neg_tolerance_st)
