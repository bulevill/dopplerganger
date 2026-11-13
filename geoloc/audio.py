"""Module M1: find aircraft Doppler signatures and silences in a recording.

Two detectors, chosen with detect(method=...):

"matched" (default; doppler-curve-detection-claude-code-brief.md, Phase 1). Search for the
whole Doppler curve at once instead of building it from per-frame peaks:
  1. detection spectrogram 30-1000 Hz, whitened locally (per-bin running median over
     blocks, per-frame median over +-15 Hz), robust z-score clipped to [0, 6];
  2. resampled to a log-frequency axis, where a straight constant-speed pass is one fixed
     shape per (speed, distance/speed) and f0 is only a vertical shift;
  3. a bank of such shapes (from physics.straight_pass_curve) is slid over the image
     (shift-and-add along each curve), each response standardised per frequency row, and
     the fundamental, 2nd and 3rd harmonic responses summed with weights 1, 0.7, 0.5;
  4. local maxima become candidates; each is traced in the fine linear spectrogram along
     its predicted curve, every harmonic separately, and fitted with the straight-pass
     model (seeded from the shape, refined twice);
  5. candidates describing the same aircraft are merged; each gets a confidence from its
     fit (rms within 1.2 % of frequency) and its contrast (mean level on the fitted curve
     minus parallel curves 4-7 % away; a free fit to noise inside the trace window passes
     the rms rule but not this). "high" and "medium" ones are reported as events, everything else (poor fit, or close/slow
     ground sources: d < 150 m or v < 30 m/s) goes to Perception.candidates, not deleted.
The design follows digitalis-p/doppler-locate (MIT): log axis, shape bank, per-band
normalisation, 1 %-of-frequency fit rule, ground-source split. The code is our own.

"ridge" (the earlier detector): spectrogram of the 30-500 Hz band -> whitening (so steady
hums and broadband noise do not count) -> per-frame tone peaks -> ridge tracking through
time -> fit of the straight-pass model to each ridge -> grouping of harmonics into events.

Two acceptance tiers (ridge detector). A "tight" fit matches the straight-pass curve within max_rms_hz.
Real tracks wobble (small turns, speed changes), so clean passes often miss that by
1-3 Hz; a "loose" fit (up to loose_rms_hz) is kept only when the ridge is long, has
tone well on both sides of closest approach and pins the distance down well. Measured
on 24 G3b recordings of real 2026-11-06 tracks (REFERENCE §6): true fits 43 -> 107,
false fits 36 -> 55 (a flat 2 Hz tolerance gave 110 true, 236 false).
"""
from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.ndimage import maximum_filter, maximum_filter1d, median_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks, stft

from .physics import straight_pass_curve

ANALYSIS_FS = 4000


@dataclass
class Spectrogram:
    t: np.ndarray       # frame centre times, s from recording start
    f: np.ndarray       # bin frequencies, Hz
    S: np.ndarray       # whitened level, dB above background, shape (f, t)
    loud: np.ndarray    # broadband level per frame, dB relative to the recording median

    def value(self, t, f):
        """Whitened level at arbitrary (t, f); nearest bin. NaN outside the spectrogram."""
        t_arr = np.asarray(t, dtype=float)
        f_arr = np.asarray(f, dtype=float)
        # Non-finite inputs (e.g. a modelled frequency from an
        # interpolated ratio that is NaN outside the emission-time
        # range, in search._score_event_matrix) must return NaN
        # DIRECTLY, before any index arithmetic.  The previous fix
        # replaced non-finite inputs with a value one step outside
        # the grid and relied on the index then being out of range;
        # that only works while the grid step and bounds stay as
        # assumed, and could silently return a real value if the
        # audio settings change.  Here the non-finite mask is part
        # of the validity check itself, so NaN/inf in -> NaN out
        # for any grid, with no rint/cast ever applied to a
        # non-finite value (np.rint(NaN).astype(int) warns
        # "invalid value encountered in cast").  (Found 2026-11-08:
        # a large G3b test printed audio.py:29 "invalid value in
        # cast" from this line.)
        finite = np.isfinite(t_arr) & np.isfinite(f_arr)
        # Safe stand-ins for the index arithmetic only: any value
        # would do, because the finite mask below forces the
        # output to NaN wherever an input was non-finite.
        t_safe = np.where(np.isfinite(t_arr), t_arr, 0.0)
        f_safe = np.where(np.isfinite(f_arr), f_arr, 0.0)
        ti = np.rint((t_safe - self.t[0]) / (self.t[1] - self.t[0])).astype(int)
        fi = np.rint((f_safe - self.f[0]) / (self.f[1] - self.f[0])).astype(int)
        ok = finite & (ti >= 0) & (ti < len(self.t)) & (fi >= 0) & (fi < len(self.f))
        out = np.full(np.shape(ti), np.nan)
        out[ok] = self.S[fi[ok], ti[ok]]
        return out


@dataclass
class Event:
    t_c: float          # arrival time of closest approach, s from recording start
    d: float            # slant distance at closest approach, m
    v: float            # ground speed, m/s
    f0: float           # lowest detected tone at closest approach, Hz
    sigma_t: float
    sigma_d: float
    sigma_v: float
    harmonics: list = field(default_factory=list)  # multiples of f0 that were seen
    span: tuple = (0.0, 0.0)                         # first and last time the tone was seen
    quality: str = "tight"      # ridge: "tight"/"loose"; matched: "high"/"medium" (events), "low"/"ground" (candidates)
    score: float = 0.0          # matched filter: standardised whole-curve response (0 for the ridge detector)
    contrast: float = 0.0       # matched filter: mean level on the fitted curve minus parallel off-curve level
    fit_rel_rms: float = 0.0    # matched filter: straight-pass fit rms / frequency


@dataclass
class Perception:
    duration: float
    events: list
    silences: list      # (start, end) s with no tone and no loud broadband sound
    spec: Spectrogram
    candidates: list = field(default_factory=list)  # matched detector: low-confidence and ground-source Events


def load(path, fs=ANALYSIS_FS):
    y, _ = librosa.load(path, sr=fs, mono=True)
    return y


def spectrogram(y, fs=ANALYSIS_FS, nfft=4096, hop=1000, fmin=30.0, fmax=500.0):
    f, t, Z = stft(y, fs, window="hann", nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
    band = (f >= fmin) & (f <= fmax)
    P = 10 * np.log10(np.abs(Z[band]) ** 2 + 1e-20)
    loud = np.median(P, axis=0)
    loud = loud - np.median(loud)
    S = P - np.median(P, axis=1, keepdims=True)   # remove steady hums per frequency
    S = S - np.median(S, axis=0, keepdims=True)    # remove broadband swells per frame
    return Spectrogram(t=t, f=f[band], S=S, loud=loud)


def _ridges(spec, threshold_db, max_gap, min_seconds):
    """Greedy ridge tracking of per-frame tone peaks."""
    dt = spec.t[1] - spec.t[0]
    df = spec.f[1] - spec.f[0]
    active, done = [], []
    for j in range(len(spec.t)):
        col = spec.S[:, j]
        idx, _ = find_peaks(col, height=threshold_db, distance=3)
        # Parabolic interpolation for sub-bin frequency.
        a, b, c = col[np.maximum(idx - 1, 0)], col[idx], col[np.minimum(idx + 1, len(col) - 1)]
        shift = 0.5 * (a - c) / np.where(a - 2 * b + c == 0, -1e-9, a - 2 * b + c)
        freqs = spec.f[idx] + np.clip(shift, -0.5, 0.5) * df
        levels = b
        taken = np.zeros(len(freqs), bool)
        for r in sorted(active, key=lambda r: -len(r["f"])):
            gap = j - r["j"][-1]
            slope = (r["f"][-1] - r["f"][-5]) / (r["t"][-1] - r["t"][-5]) if len(r["f"]) >= 5 else 0.0
            pred = r["f"][-1] + slope * gap * dt
            tol = 2.0 + abs(slope) * gap * dt
            cand = np.where(~taken & (np.abs(freqs - pred) <= tol))[0]
            if cand.size:
                k = cand[np.argmin(np.abs(freqs[cand] - pred))]
                taken[k] = True
                r["f"].append(freqs[k]); r["t"].append(spec.t[j]); r["j"].append(j); r["lvl"].append(levels[k])
        for k in np.where(~taken)[0]:
            active.append({"f": [freqs[k]], "t": [spec.t[j]], "j": [j], "lvl": [levels[k]]})
        still = []
        for r in active:
            (still if j - r["j"][-1] <= max_gap else done).append(r)
        active = still
    done += active
    return [{k: np.array(v) for k, v in r.items()} for r in done if r["t"][-1] - r["t"][0] >= min_seconds]


def _fit(r, c=343.0):
    """Fit the straight-pass model; returns (params, sigmas, rms) or None."""
    t, f = r["t"], r["f"]
    if len(t) < 9:  # the 9-point smoother below needs at least 9 samples (sparse ridges at large max_gap)
        return None
    fs = np.convolve(f, np.ones(9) / 9, mode="same")
    slope = np.gradient(fs, t)
    inner = slice(4, len(t) - 4) if len(t) > 12 else slice(None)
    i = np.argmin(slope[inner]) + (inner.start or 0)
    if slope[i] >= 0:
        return None
    f_hi, f_lo = np.percentile(f, 95), np.percentile(f, 5)
    v0 = np.clip(c * (f_hi - f_lo) / (f_hi + f_lo), 20, 140)
    f00 = fs[i]
    d0 = np.clip(f00 * v0**2 / (c * abs(slope[i])), 100, 12000)
    x0 = [t[i], d0, v0, f00]
    lo = [t[0] - 60, 30.0, 10.0, f.min() * 0.7]
    hi = [t[-1] + 60, 20000.0, 160.0, f.max() * 1.3]
    try:
        res = least_squares(lambda p: straight_pass_curve(t, *p, c=c) - f, x0, bounds=(lo, hi),
                            loss="soft_l1", f_scale=0.5)
    except ValueError:
        return None
    resid = res.fun
    rms = float(np.sqrt(np.mean(resid**2)))
    dof = max(len(t) - 4, 1)
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * (np.sum(resid**2) / dof)
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        return None
    return res.x, sig, rms


def detect(y, fs=ANALYSIS_FS, method="matched", **kw):
    """Find Doppler events and silences. method="matched" (default) or "ridge"; see module docstring."""
    if method == "ridge":
        return _detect_ridge(y, fs, **kw)
    if method == "matched":
        return _detect_matched(y, fs, **kw)
    raise ValueError(f"unknown method {method!r}")


def _detect_ridge(y, fs=ANALYSIS_FS, threshold_db=8.0, max_gap=8, min_seconds=20.0, max_rms_hz=0.5,
           min_side_s=5.0, silence_min_s=30.0, silence_margin_s=10.0, loud_db=6.0,
           min_v=12.0, min_d=100.0, max_rel_sigma_d=0.5,
           loose_rms_hz=4.0, loose_min_seconds=90.0, loose_min_side_s=30.0, loose_max_rel_sigma_d=0.1,
           loose_min_v=20.0):
    """loose_rms_hz=None turns the loose tier off (tight fits only)."""
    spec = spectrogram(y, fs)
    duration = len(y) / fs
    ridges = _ridges(spec, threshold_db, max_gap, min_seconds)

    fits = []
    for r in ridges:
        out = _fit(r)
        if out is None:
            continue
        (t_c, d, v, f0), sig, rms = out
        # Require the tone on both sides of closest approach, else the geometry is unconstrained.
        side = min(t_c - r["t"][0], r["t"][-1] - t_c)
        rel_sd = sig[1] / max(d, 1.0)
        # v at the fit's 10 m/s floor or a tiny d means the model did not really fit.
        tight = (rms <= max_rms_hz and side >= min_side_s and v >= min_v and d >= min_d
                 and rel_sd <= max_rel_sigma_d)
        loose = (loose_rms_hz is not None and rms <= loose_rms_hz and side >= loose_min_side_s
                 and r["t"][-1] - r["t"][0] >= loose_min_seconds and rel_sd <= loose_max_rel_sigma_d
                 and v >= loose_min_v and d >= min_d)
        if not (tight or loose):
            continue
        fits.append({"t_c": t_c, "d": d, "v": v, "f0": f0, "sig": sig, "n": len(r["t"]),
                     "span": (r["t"][0], r["t"][-1]), "lvl": float(np.median(r["lvl"])),
                     "tight": bool(tight)})

    # Harmonics of one aircraft share t_c, v and d: merge them into one event.
    events = []
    for g in sorted(fits, key=lambda x: -x["n"]):
        for e in events:
            if abs(g["t_c"] - e["t_c"]) < 3 and abs(g["v"] - e["v"]) < 0.15 * e["v"] and \
                    abs(g["d"] - e["d"]) < 0.25 * e["d"]:
                e["members"].append(g)
                break
        else:
            events.append({**g, "members": [g]})
    result = []
    for e in events:
        best = max(e["members"], key=lambda m: m["n"])
        f_low = min(m["f0"] for m in e["members"])
        harmonics = sorted({int(round(m["f0"] / f_low)) for m in e["members"]})
        span = (min(m["span"][0] for m in e["members"]), max(m["span"][1] for m in e["members"]))
        result.append(Event(t_c=best["t_c"], d=best["d"], v=best["v"], f0=f_low,
                            sigma_t=float(best["sig"][0]), sigma_d=float(best["sig"][1]),
                            sigma_v=float(best["sig"][2]), harmonics=harmonics, span=span,
                            quality="tight" if any(m["tight"] for m in e["members"]) else "loose"))
    result.sort(key=lambda e: e.t_c)

    # Silence: no tracked tone (fitted or not) and no loud broadband sound.
    busy = spec.loud > loud_db
    for r in ridges:
        busy |= (spec.t >= r["t"][0] - silence_margin_s) & (spec.t <= r["t"][-1] + silence_margin_s)
    silences, start = [], None
    for tj, b in zip(spec.t, busy):
        if not b and start is None:
            start = tj
        if b and start is not None:
            if tj - start >= silence_min_s:
                silences.append((float(start), float(tj)))
            start = None
    if start is not None and spec.t[-1] - start >= silence_min_s:
        silences.append((float(start), float(spec.t[-1])))
    return Perception(duration=duration, events=result, silences=silences, spec=spec)


# ----------------------------------------------------------------------------------------
# Matched-filter detector (method="matched")
# ----------------------------------------------------------------------------------------

MF_V_GRID = np.geomspace(15.0, 250.0, 15)      # m/s (review B10: 160 m/s cap missed real passes)
MF_TAU_GRID = np.geomspace(2.0, 240.0, 14)     # s, tau = d / v
MF_D_RANGE = (100.0, 15000.0)                  # m, shapes outside are not built
GROUND_D, GROUND_V = 150.0, 30.0               # closer or slower fits: road vehicles, people
# Reported sigmas = formal fit error combined in quadrature with a model-error term. The formal
# error ignores turning/wobbling tracks (a bias, not noise), sound-speed mismatch and wind; the
# relative terms below were fitted on matched G3b (real-track) detections (review A2/B8;
# scripts/m1_eval.py reports the coverage). 3 sigma then covers ~95 % in v but only ~82 % in d:
# about 1 in 6 G3b distances is a gross failure (2x+ too close, the near-CPA degeneracy), which no
# Gaussian term can cover without making d uninformative for the rest.
SIGMA_T_FLOOR = 0.3
SIGMA_D_MODEL, SIGMA_V_MODEL = 0.200, 0.080


@dataclass
class _Shape:
    v: float
    d: float
    k: np.ndarray       # time offsets of the sampled cells, in coarse frames
    dl: np.ndarray      # log-frequency offsets of the cells, in coarse log bins


def _detection_image(y, fs, fmin, fmax, block_s, local_hz, hop_s=0.25, nfft=4096):
    """Locally whitened linear spectrogram as robust z-scores clipped to [0, 6]."""
    hop = int(round(hop_s * fs))
    f, t, Z = stft(y, fs, window="hann", nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
    band = (f >= fmin * 0.97) & (f <= fmax * 1.03)
    f = f[band]
    P = (10 * np.log10(np.abs(Z[band]) ** 2 + 1e-20)).astype(np.float32)
    # Per-bin running median: block medians, blended linearly between block centres.
    w = max(8, int(round(block_s / hop_s)))
    if P.shape[1] <= w:
        P -= np.median(P, axis=1, keepdims=True)
    else:
        starts = np.arange(0, P.shape[1], w)
        cm = np.stack([np.median(P[:, a:a + w], axis=1) for a in starts], axis=1)
        centres = np.array([(a + min(a + w, P.shape[1])) / 2 for a in starts])
        x = np.arange(P.shape[1])
        i1 = np.clip(np.searchsorted(centres, x), 1, len(centres) - 1)
        i0 = i1 - 1
        wt = np.clip((x - centres[i0]) / (centres[i1] - centres[i0]), 0.0, 1.0)
        P -= cm[:, i0] * (1 - wt) + cm[:, i1] * wt
    # Per-frame local median over +-local_hz removes broadband events (cars, wind, voices).
    k = max(3, int(round(2 * local_hz / (f[1] - f[0]))) | 1)
    P -= median_filter(P, size=(k, 1), mode="nearest")
    mad = float(np.median(np.abs(P))) * 1.4826 + 1e-6
    return t, f, np.clip(P / mad, 0.0, 6.0)


def _to_log(Zs, f, lf, step):
    """(log bins, frames) image: max over linear bins inside a log bin, interpolation where a
    log bin is narrower than a linear one."""
    fl = np.exp(lf)
    j = np.clip(np.searchsorted(f, fl) - 1, 0, len(f) - 2)
    a = np.clip((fl - f[j]) / (f[j + 1] - f[j]), 0.0, 1.0)[:, None]
    out = (Zs[j] * (1 - a) + Zs[j + 1] * a).astype(np.float32)
    idx = np.floor((np.log(np.maximum(f, 1e-6)) - lf[0]) / step).astype(int)
    ok = (idx >= 0) & (idx < len(lf))
    mx = np.full_like(out, -np.inf)
    np.maximum.at(mx, idx[ok], Zs[ok])
    return np.where(np.isfinite(mx), np.maximum(mx, out), out)


def _bank(dt, log_step, c=343.0, span_tau=1.5, min_half=8.0, max_half=150.0, tail_gap_s=4.0):
    shapes = []
    for v in MF_V_GRID:
        for tau in MF_TAU_GRID:
            d = v * tau
            if not MF_D_RANGE[0] <= d <= MF_D_RANGE[1]:
                continue
            half = float(np.clip(span_tau * tau, min_half, max_half))
            kk = np.arange(-int(round(half / dt)), int(round(half / dt)) + 1)
            dl = np.rint(np.log(straight_pass_curve(kk * dt, 0.0, d, v, 1.0, c)) / log_step).astype(int)
            # Keep a cell where the curve moves to a new log bin, and every tail_gap_s on flat parts.
            keep, last_dl, last_k = [], None, None
            for i, (k_, l_) in enumerate(zip(kk, dl)):
                if last_dl is None or l_ != last_dl or (k_ - last_k) * dt >= tail_gap_s or i == len(kk) - 1:
                    keep.append(i)
                    last_dl, last_k = l_, k_
            keep = np.array(keep)
            shapes.append(_Shape(v=float(v), d=float(d), k=kk[keep], dl=dl[keep]))
    return shapes


def _shape_response(img, s):
    """Mean of img along shape s anchored at every (row, col): out[j, n] = mean_i img[j + dl_i, n + k_i],
    standardised per frequency row (constant false-alarm rate)."""
    L, N = img.shape
    pr, pc = int(np.abs(s.dl).max()), int(np.abs(s.k).max())
    pad = np.zeros((L + 2 * pr, N + 2 * pc), np.float32)
    pad[pr:pr + L, pc:pc + N] = img
    inside = np.zeros_like(pad)
    inside[pr:pr + L, pc:pc + N] = 1.0
    acc = np.zeros((L, N), np.float32)
    cnt = np.zeros((L, N), np.float32)
    for k_, l_ in zip(s.k, s.dl):
        acc += pad[pr + l_:pr + l_ + L, pc + k_:pc + k_ + N]
        cnt += inside[pr + l_:pr + l_ + L, pc + k_:pc + k_ + N]
    # Mean over the cells inside the image (review B1: dividing by all cells diluted long shapes
    # near the edges, e.g. on short clips); too little of the shape inside -> no response.
    acc = np.where(cnt >= 0.4 * len(s.k), acc / np.maximum(cnt, 1.0), 0.0)
    # Mean/std, not median/MAD: the clipped image is mostly exact zeros, so the MAD is tiny and
    # noise blips would score as high as real curves. Relative floor on sd (review B12).
    mu = acc.mean(axis=1, keepdims=True)
    sd = acc.std(axis=1, keepdims=True)
    sd = sd + 1e-6 + 0.05 * float(np.median(sd))
    return (acc - mu) / sd


def _trace(Zs, t, f, t_c, d, v, fc, half, tol_rel=0.015, min_z=1.5, min_cover=0.3):
    """Points (t, f) of the strongest bin within +-tol_rel of the predicted curve, per frame.
    None unless points cover min_cover of the frames within +-max(10 s, d/v) of t_c."""
    m = (t >= t_c - half) & (t <= t_c + half)
    if m.sum() < 8:
        return None
    tt = t[m]
    cols = np.flatnonzero(m)
    fp = fc * straight_pass_curve(tt, t_c, d, v, 1.0)
    df = f[1] - f[0]
    lo = np.searchsorted(f, fp * (1 - tol_rel) - df)
    hi = np.searchsorted(f, fp * (1 + tol_rel) + df)
    ok = (hi - lo >= 1) & (lo > 0) & (hi < len(f))
    if ok.sum() < 8:
        return None
    lo, hi, cols, tt = lo[ok], hi[ok], cols[ok], tt[ok]
    width = int((hi - lo).max())
    rows = lo[:, None] + np.arange(width)[None, :]                       # (frames, width)
    inband = rows < hi[:, None]
    vals = np.where(inband, Zs[np.minimum(rows, len(f) - 1), cols[:, None]], -np.inf)
    k = np.argmax(vals, axis=1)
    peak = vals[np.arange(len(k)), k]
    keep = peak >= min_z
    if keep.sum() < 8:
        return None
    kk = (lo + k)[keep]
    cc_ = cols[keep]
    a, b, c3 = Zs[kk - 1, cc_], Zs[kk, cc_], Zs[np.minimum(kk + 1, len(f) - 1), cc_]
    den = a - 2 * b + c3
    sh = np.where(den != 0, 0.5 * (a - c3) / np.where(den != 0, den, 1.0), 0.0)
    T = tt[keep]
    F = f[kk] + np.clip(sh, -0.5, 0.5) * df
    core = max(10.0, d / max(v, 1.0))
    n_core = int(np.sum(np.abs(tt - t_c) <= core))
    if n_core == 0 or np.sum(np.abs(T - t_c) <= core) < min_cover * n_core:
        return None
    return T, F


def _mask_family(Zw, t, f, t_c, d, v, f0, span, fmax, tol_rel=0.02):
    """Zero every harmonic of a fitted curve in Zw, so later candidates cannot re-trace it."""
    m = (t >= span[0] - 5.0) & (t <= span[1] + 5.0)
    cols = np.flatnonzero(m)
    if cols.size == 0:
        return
    fp = straight_pass_curve(t[m], t_c, d, v, 1.0)
    df = f[1] - f[0]
    for h in range(1, int(fmax * 1.03 // max(f0, 1.0)) + 1):
        fh = h * f0 * fp
        lo = np.searchsorted(f, fh * (1 - tol_rel) - df)
        hi = np.searchsorted(f, fh * (1 + tol_rel) + df)
        for col, a, b in zip(cols, lo, hi):
            Zw[a:b, col] = 0.0


def _fit_seeded(T, F, x0, c=343.0):
    """Straight-pass fit from a seed; sigmas inflated for correlated residuals (wobble)."""
    lo = np.array([T[0] - 60, 30.0, 10.0, F.min() * 0.7])
    hi = np.array([T[-1] + 60, 20000.0, 260.0, F.max() * 1.3])
    x0 = np.clip(np.asarray(x0, float), lo + 1e-6, hi - 1e-6)
    try:
        res = least_squares(lambda p: straight_pass_curve(T, *p, c=c) - F, x0, bounds=(lo, hi),
                            loss="soft_l1", f_scale=max(0.5, 0.005 * float(np.median(F))))
    except ValueError:
        return None
    r = res.fun
    rms = float(np.sqrt(np.mean(r ** 2)))
    n = len(T)
    rho = float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], 0.0, 0.99)) if n > 3 and np.std(r) > 0 else 0.0
    n_eff = max(n * (1 - rho) / (1 + rho), 5.0)
    # Plain finite-difference Jacobian: under a robust loss scipy returns a loss-modified one
    # (review B8), which does not pair with the raw residual variance.
    J = np.empty((n, 4))
    for i in range(4):
        h = 1e-4 * max(abs(res.x[i]), 1.0)
        xp, xm = res.x.copy(), res.x.copy()
        xp[i] += h
        xm[i] -= h
        J[:, i] = (straight_pass_curve(T, *xp, c=c) - straight_pass_curve(T, *xm, c=c)) / (2 * h)
    try:
        # Residual variance times n / n_eff: correlated residuals carry fewer independent samples.
        cov = np.linalg.inv(J.T @ J) * (np.sum(r ** 2) / max(n - 4, 1)) * (n / n_eff)
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        return None
    return res.x, sig, rms


def _contrast(Zs, t, f, t_c, d, v, tones, span, offsets=(0.04, 0.07)):
    """Mean level along the fitted curve(s) minus the mean along parallel curves offset by
    +-offsets (relative): tone evidence that a free fit to noise inside the trace window cannot fake."""
    m = (t >= span[0]) & (t <= span[1])
    cols = np.flatnonzero(m)
    if cols.size < 8:
        return 0.0
    fp = straight_pass_curve(t[m], t_c, d, v, 1.0)
    df = f[1] - f[0]

    def level(fr):
        i = np.rint((fr - f[0]) / df).astype(int)
        ok = (i >= 1) & (i < len(f) - 1)
        vals = np.maximum(np.maximum(Zs[np.clip(i - 1, 0, len(f) - 1), cols], Zs[np.clip(i, 0, len(f) - 1), cols]),
                          Zs[np.clip(i + 1, 0, len(f) - 1), cols])
        return vals[ok]

    on, off = [], []
    for ft in tones:
        on.append(level(ft * fp))
        for o in offsets:
            off.append(level(ft * fp * (1 + o)))
            off.append(level(ft * fp * (1 - o)))
    on = np.concatenate(on) if on else np.zeros(0)
    off = np.concatenate(off) if off else np.zeros(0)
    if on.size == 0 or off.size == 0:
        return 0.0
    return float(np.mean(on) - np.mean(off))


def _shape_scores_at(img, bank, j, n, shifts, w):
    """Harmonic-summed mean response of every shape anchored at one (row, col); for seeding."""
    L, N = img.shape
    out = np.zeros(len(bank))
    for si, s in enumerate(bank):
        tot = 0.0
        for wh, sh in zip(w, shifts):
            r = j + sh + s.dl
            c = n + s.k
            ok = (r >= 0) & (r < L) & (c >= 0) & (c < N)
            if ok.any():
                tot += wh * float(img[r[ok], c[ok]].sum()) / len(s.k)
        out[si] = tot
    return out


def _harmonic_ratio(fa, fb, tol=0.03, hmax=6):
    """(a, b) with fa/fb close to a/b, or None."""
    for a in range(1, hmax + 1):
        for b in range(1, hmax + 1):
            if abs(fa * b / (fb * a) - 1.0) <= tol:
                return a, b
    return None


def _detect_matched(y, fs=ANALYSIS_FS, fmin=30.0, fmax=1000.0, log_step=0.008, dt=1.0, block_s=120.0,
                    local_hz=15.0, threshold=4.5, max_candidates=200, n_seeds=3, max_harmonic=4,
                    trace_tols=(0.015, 0.03, 0.05, 0.015), reseed_v=(0.6, 0.8, 1.25, 1.6, 2.0),
                    harmonic_weights=(1.0, 0.7, 0.5), weak_tone_z=2.5,
                    high_rel_rms=0.006, medium_rel_rms=0.012, min_contrast=1.0, high_contrast=2.0, min_listed_contrast=0.3, min_side_s=5.0,
                    max_rel_sigma_d=0.5, silence_min_s=30.0, silence_margin_s=10.0, loud_db=6.0):
    duration = len(y) / fs
    spec = spectrogram(y, fs)                       # unchanged meaning: loudness, silences, downstream
    t_lin, f_lin, Zs = _detection_image(y, fs, fmin, fmax, block_s, local_hz)
    hop_s = t_lin[1] - t_lin[0]

    # Coarse log image for the shape search.
    lf = np.arange(np.log(fmin), np.log(fmax), log_step)
    img = _to_log(Zs, f_lin, lf, log_step)
    pool = max(1, int(round(dt / hop_s)))
    n_c = img.shape[1] // pool
    img = img[:, :n_c * pool].reshape(len(lf), n_c, pool).max(axis=2)
    t_c_grid = t_lin[:n_c * pool].reshape(n_c, pool).mean(axis=1)
    img = maximum_filter1d(img, 3, axis=0)          # +-1 log bin tolerance for curve mismatch

    bank = _bank(dt, log_step)
    best = np.full(img.shape, -np.inf, np.float32)
    which = np.zeros(img.shape, np.int32)
    w = np.asarray(harmonic_weights, float)
    shifts = [int(round(np.log(h) / log_step)) for h in range(1, len(w) + 1)]
    norm = float(np.sqrt(np.sum(w ** 2)))
    for si, s in enumerate(bank):
        R = _shape_response(img, s)
        H = np.zeros_like(R)
        for wh, sh in zip(w, shifts):
            if sh == 0:
                H += wh * R
            elif sh < R.shape[0]:
                H[:-sh] += wh * R[sh:]
        H /= norm
        upd = H > best
        best[upd] = H[upd]
        which[upd] = si

    # Candidates: local maxima above threshold.
    peak = (best == maximum_filter(best, size=(7, 9))) & (best >= threshold)
    rows, cols = np.nonzero(peak)
    order = np.argsort(-best[rows, cols])[:max_candidates]

    ridges = _ridges(spec, 8.0, 8, 20.0)           # free ridge tracks: refiner here, and silence evidence
    found = []
    accepted = []                                   # (t_c, d, v, f0, span) of families already explained
    Zw = Zs.copy()                                  # tracing image; accepted curves are masked out of it
    for o in order:
        j, n = rows[o], cols[o]
        t_c0, fc0, score = float(t_c_grid[n]), float(np.exp(lf[j])), float(best[j, n])
        # A loud pass lights up many anchors along its harmonics: skip those already explained. The
        # test stays wide (6 %, ratios a/b <= 12): real tracks wander off the straight-pass curve, and
        # anchors on that residue otherwise become false events. At this width the ratios cover nearly
        # every frequency, so in effect no second aircraft is sought inside an accepted pass's span
        # (G3b: ~8 close passes hidden this way). Tested on the tuning sets: 1.5 %/h<=6 gave G3b 60
        # matched / 43 usable at 18.7 FP/h, 6 %/h<=6 59 / 41 at 10.2, this 55 / 44 at 3.7. Integer
        # multiples only, with fits inside a stronger event's span held to contrast >= 2.0, gave
        # 61 / 49 at 6.9 FP/h but located 18/24 G3b runs against 19/24 here, and the New York
        # searches ran 2-40x slower: the extra passes did not help the search. A whole-curve test was
        # also rejected: it compares the seed shape, which for these anchors is usually wrong.
        if any(sp[0] - 10 <= t_c0 <= sp[1] + 10 and
               _harmonic_ratio(fc0, f0a * float(straight_pass_curve(t_c0, tca, da, va, 1.0)), tol=0.06, hmax=12)
               is not None for tca, da, va, f0a, sp in accepted):
            continue
        # Gentle (distant) curves match several shapes almost equally: seed from the best few.
        alt = _shape_scores_at(img, bank, j, n, shifts, w)
        seeds = [which[j, n]] + [si for si in np.argsort(-alt)[:n_seeds] if si != which[j, n]][:n_seeds - 1]
        def refine(x, h):
            tr = out = None
            # Trace -> fit -> re-trace on the fitted curve, with a widening band; then one tight pass.
            # A failed trace keeps the current curve and tries the next band (review B3).
            for tol in trace_tols:
                # Window from the current curve: the flat tails carry the speed information.
                half = float(np.clip(3.0 * x[1] / max(x[2], 1.0), 20.0, 200.0))
                tr_new = _trace(Zw, t_lin, f_lin, x[0], x[1], x[2], x[3], half, tol_rel=tol)
                if tr_new is None:
                    continue
                out_new = _fit_seeded(tr_new[0], tr_new[1], x)
                if out_new is None:
                    continue
                tr, out = tr_new, out_new
                x = out[0]
            if out is None:
                return None
            (tc, d, v, f0), sig, rms = out
            return {"h": h, "t_c": tc, "d": d, "v": v, "f0": f0, "sig": sig, "rms": rms, "rel": rms / f0,
                    "n": len(tr[0]), "side": min(tc - tr[0][0], tr[0][-1] - tc),
                    "span": (float(tr[0][0]), float(tr[0][-1]))}

        def quality(g):                              # smaller is better: tight fit through many points
            return g["rel"] / np.sqrt(g["n"])

        def support(g):
            # Mean of Zs along the fit's tone in a window common to all fits of this candidate, so long
            # and short fits are compared on the same frames (review B5).
            m = (t_lin >= t_c0 - 150.0) & (t_lin <= t_c0 + 150.0)
            fr = g["f0"] * straight_pass_curve(t_lin[m], g["t_c"], g["d"], g["v"], 1.0)
            i = np.rint((fr - f_lin[0]) / (f_lin[1] - f_lin[0])).astype(int)
            ok = (i >= 1) & (i < len(f_lin) - 1)
            cols = np.flatnonzero(m)[ok]
            i = i[ok]
            if i.size == 0:
                return 0.0
            v = np.maximum(np.maximum(Zs[i - 1, cols], Zs[i, cols]), Zs[i + 1, cols])
            return float(v.sum()) / np.count_nonzero(m)

        fits = []
        for si, h in [(si, h) for si in seeds for h in range(1, max_harmonic + 1)]:
            if h * fc0 > fmax:
                continue
            g = refine(np.array([t_c0, bank[si].d, bank[si].v, h * fc0]), h)
            if g is not None:
                fits.append(g)
        if not fits:
            continue
        # Only candidates with tone evidence get the costly re-seeds and ridge refits: most anchors
        # are noise, and these steps cannot turn noise into a pass.
        g_best = max(fits, key=support)
        if _contrast(Zw, t_lin, f_lin, g_best["t_c"], g_best["d"], g_best["v"], [g_best["f0"]],
                     g_best["span"]) < min_listed_contrast:
            continue
        # Near closest approach only v^2/d is measured, so a seed that is too steep settles on a curve
        # through only part of the tone. Re-seed along that degeneracy line (same v^2/d) from every
        # good fit, not just the best (review B2/B11); the support comparison below picks the curve
        # that follows the most tone, tails included.
        for g0 in sorted(fits, key=quality)[:3]:
            for vm in reseed_v:
                g = refine(np.array([g0["t_c"], g0["d"] * vm ** 2, g0["v"] * vm, g0["f0"]]), g0["h"])
                if g is not None:
                    fits.append(g)
        # Ridge refiner: a freely tracked ridge lying along the fitted curve (any harmonic) is not
        # biased by the seed shape, so it recovers the tails of distant passes. Refit on it.
        b0 = max(fits, key=support)
        for r in ridges:
            # Compare near closest approach only: a too-steep fit and the true curve agree there.
            core = max(15.0, 0.5 * b0["d"] / max(b0["v"], 1.0))
            ov = np.abs(r["t"] - b0["t_c"]) <= core
            if ov.sum() < 12 or r["t"][-1] - r["t"][0] < 30:
                continue
            fb = straight_pass_curve(r["t"][ov], b0["t_c"], b0["d"], b0["v"], 1.0) * b0["f0"]
            k = r["f"][ov] / fb
            ratio = float(np.median(k))
            if ratio < 0.15:
                continue
            hk = None
            for a in range(1, 7):
                for bb in range(1, 7):
                    if abs(ratio - a / bb) < 0.02 * a / bb:
                        hk = a / bb
                        break
                if hk is not None:
                    break
            if hk is None or np.median(np.abs(k / hk - 1.0)) > 0.02:
                continue
            out = _fit_seeded(r["t"], r["f"], np.array([b0["t_c"], b0["d"], b0["v"], b0["f0"] * hk]))
            if out is None:
                continue
            (tc, d, v, f0), sig, rms = out
            # The ridge must be the same pass along its whole length (review B5): it fits the model
            # well, and its closest approach agrees with the candidate's.
            if rms / f0 > medium_rel_rms or abs(tc - b0["t_c"]) > max(10.0, 0.25 * b0["d"] / max(b0["v"], 1.0)):
                continue
            fits.append({"h": b0["h"], "t_c": tc, "d": d, "v": v, "f0": f0, "sig": sig, "rms": rms, "rel": rms / f0,
                         "n": len(r["t"]), "side": min(tc - r["t"][0], r["t"][-1] - tc),
                         "span": (float(r["t"][0]), float(r["t"][-1])), "ridge": True})
        good = [g for g in fits if g["rel"] <= medium_rel_rms and g["side"] >= min_side_s]
        ref = max(good or fits, key=support)
        seen = sorted({g["f0"] for g in good if abs(g["t_c"] - ref["t_c"]) < 5 and abs(g["v"] - ref["v"]) < 0.2 * ref["v"]
                       and _harmonic_ratio(g["f0"], ref["f0"]) is not None}) or [ref["f0"]]
        # Contrast on the masked image: on the unmasked one a fit beside a loud accepted pass borrows its
        # tones (tested: false events next to real ones). The cost, a crossing aircraft losing the few
        # masked cells where it crosses, is small because contrast is a mean over the whole span.
        con = _contrast(Zw, t_lin, f_lin, ref["t_c"], ref["d"], ref["v"], seen, ref["span"])
        found.append({**ref, "score": score, "tones": seen, "contrast": con})
        if good and con >= min_contrast:
            _mask_family(Zw, t_lin, f_lin, ref["t_c"], ref["d"], ref["v"], min(seen), ref["span"], fmax)
            accepted.append((ref["t_c"], ref["d"], ref["v"], min(seen), ref["span"]))

    # Merge candidates that describe the same aircraft (same time, speed, distance; harmonic tones).
    found.sort(key=lambda e: -e["score"])
    merged = []
    for e in found:
        for m in merged:
            near_t = abs(e["t_c"] - m["t_c"]) < max(3.0, 0.1 * m["d"] / max(m["v"], 1.0))
            same_geom = abs(e["v"] - m["v"]) < 0.2 * m["v"] and abs(e["d"] - m["d"]) < 0.35 * m["d"]
            # Two fits of the same tone (exact harmonic relation) at the same closest-approach time are
            # one aircraft even if their distances disagree: one fit is wrong; keep the better-evidenced.
            same_tone = _harmonic_ratio(e["f0"], m["f0"], tol=0.015) is not None
            if near_t and ((same_geom and _harmonic_ratio(e["f0"], m["f0"]) is not None) or same_tone):
                m["tones"] = sorted(set(m["tones"]) | set(e["tones"])) if same_geom else m["tones"]
                if (e["contrast"], -e["rel"]) > (m["contrast"], -m["rel"]) if not same_geom else e["rel"] < m["rel"]:
                    m.update({k: e[k] for k in ("t_c", "d", "v", "f0", "sig", "rms", "rel", "n", "side")})
                m["contrast"] = max(m["contrast"], e["contrast"])
                m["span"] = (min(m["span"][0], e["span"][0]), max(m["span"][1], e["span"][1]))
                break
        else:
            merged.append(dict(e))

    events, candidates = [], []
    for m in merged:
        # Tones within 3 % of each other are the same tone fitted twice.
        tones = []
        for ft in sorted(m["tones"]):
            if not tones or ft > tones[-1] * 1.03:
                tones.append(ft)
        f_low = tones[0]
        harmonics = sorted({int(round(ft / f_low)) for ft in tones})
        sig = m["sig"]
        if m["contrast"] < min_listed_contrast:
            continue                                # no tone evidence: a fit to noise, not a candidate
        if m["d"] < GROUND_D or m["v"] < GROUND_V:
            q = "ground"
        elif (m["contrast"] >= min_contrast and m["rel"] <= medium_rel_rms and m["side"] >= min_side_s
              and sig[1] / max(m["d"], 1.0) <= max_rel_sigma_d and 10.5 < m["v"] < 259.5 and 30.5 < m["d"] < 19900.0):
            q = "high" if (m["rel"] <= high_rel_rms and m["contrast"] >= high_contrast) else "medium"
        else:
            q = "low"
        ev = Event(t_c=float(m["t_c"]), d=float(m["d"]), v=float(m["v"]), f0=float(f_low),
                   sigma_t=max(float(sig[0]), SIGMA_T_FLOOR),
                   sigma_d=float(np.hypot(sig[1], SIGMA_D_MODEL * m["d"])),
                   sigma_v=float(np.hypot(sig[2], SIGMA_V_MODEL * m["v"])),
                   harmonics=harmonics, span=(float(m["span"][0]), float(m["span"][1])), quality=q,
                   score=float(m["score"]), contrast=float(m["contrast"]), fit_rel_rms=float(m["rel"]))
        (events if q in ("high", "medium") else candidates).append(ev)
    events.sort(key=lambda e: e.t_c)
    candidates.sort(key=lambda e: e.t_c)

    # Silence: no loud broadband sound, no event, no ridge in the 30-500 Hz spectrogram, and no weak
    # tone track anywhere in the 30-1000 Hz detection image (review B9: silence must be blocked by the
    # raw evidence, however weak, not by the accepted-event or candidate lists; a faint real pass
    # below the event threshold must not leave a false silence).
    weak = _ridges(Spectrogram(t=t_lin, f=f_lin, S=Zs, loud=np.zeros(len(t_lin))), weak_tone_z, 8, 10.0)
    busy = spec.loud > loud_db
    for a, b in [(r["t"][0], r["t"][-1]) for r in ridges + weak] + [e.span for e in events]:
        busy |= (spec.t >= a - silence_margin_s) & (spec.t <= b + silence_margin_s)
    silences, start = [], None
    for tj, bz in zip(spec.t, busy):
        if not bz and start is None:
            start = tj
        if bz and start is not None:
            if tj - start >= silence_min_s:
                silences.append((float(start), float(tj)))
            start = None
    if start is not None and spec.t[-1] - start >= silence_min_s:
        silences.append((float(start), float(spec.t[-1])))
    return Perception(duration=duration, events=events, silences=silences, spec=spec, candidates=candidates)
