"""Modules M3-M6: coarse screening, forward scoring and posterior heat map.

Fresh implementation (2026-11-07), written from FRAMEWORK.md and
``Geolocation strategy.md`` -- not a continuation of
``search_partial_reference.py`` (which stops mid-function and was never run;
it is kept only as a reference).

Time bases -- read this first, every time is labelled by its base:
  * flight time: Unix seconds (ADS-B ``t`` column, ``Track.t``).
  * audio time: seconds from the start of the recording
    (``Event.t_c``, ``Spectrogram.t``, silences).
  * T0: Unix time of the recording start.  ``t_audio = t_arr_unix - T0``,
    where ``t_arr_unix`` is when a sound is heard.  T0 is unknown unless
    the caller passes ``t0_known``; with two events it cancels in the
    difference ``dt = t_c2 - t_c1 = t_arr2 - t_arr1`` (FRAMEWORK section 1).

Method (FRAMEWORK section 3 funnel F3-F9, section 4 posterior;
strategy sections 2-5):
  F3/F4  For every grid cell, the closest approach (CPA) of every flight
         segment, from the piecewise-linear track through its ADS-B
         samples (edges, not the spline -- the spline is used for the
         final forward curve in M4).  A CPA is kept if its slant distance
         is below ``rmax_m``; it is *marked* for an event if its distance
         and speed lie in that event's gate (FRAMEWORK section 9.3:
         +/-20% around the M1 estimate, widened to 3*sigma).
  F5     Pair gate: at one cell, CPAs for different events whose implied
         T0 values agree within ``dt_tol_s`` (or, with known T0, whose
         predicted audio CPA times agree with the detected ones).
         One event alone is a mirror line pair, not a point (FRAMEWORK
         section 1); two non-parallel events give <=4 intersections and
         dt selects among them.  Hypotheses are therefore tried for
         every event subset of size >= 2 (pairs, triples, ...; size 1
         with known T0), so a stray detection cannot hide the true
         triple.  Extra detected events (spurious harmonics are common
         in M1) do not veto a subset: every subset is tried.
  F6     Silence gate, SOFT (FRAMEWORK section 3 improvement 3, section 4
         "silence likelihood"): no hard distance cutoff.  For each
         silence window and each flight whose CPA sound would arrive in
         it, add ``log(1 - P(audible))``.  A faint 5.9 km pass costs a
         little, a loud pass costs a lot, and one unlucky miss cannot
         eliminate the true cell.
         P(audible) starts from the same model as
         ``synth.render_scene`` -- spherical spreading plus
         ``excess_db_per_km`` -- so G3 tests the code, not a mismatch
         between two audibility models (decision 2026-11-07).  T3 is
         expected to replace both with measured values.
  F7     Collapse surviving cells by flight pair into blobs.
  F8     Quick score at each blob centroid: forward-model the Doppler
         curve with ``physics`` (M4) -- see below -- sweep f0 per event
         and a *common* dt (locked offset, strategy section 5), score
         the harmonics on the whitened spectrogram.
  F9     Refine 100 -> 25 -> 5 m in a small window around each surviving
         blob, with the sweep parameters locked from F8, then a small
         re-sweep at the final peak.
  M6     Posterior: per cell, logsumexp over the hypotheses covering it
         of (event log-likelihood + silence log-likelihood), plus a
         uniform prior for G3.  Both a max-over-sweep and a
         marginalised (logsumexp-over-sweep) version are computed and
         kept side by side (decision 2026-11-07): the peak is usually
         the same, the widths differ, and that is exactly what the
         T4 calibration test measures.  The marginalised map is a
         TRUE per-cell full sweep (Fix 6a, 2026-11-07), not the max
         map shifted by a per-blob constant.

Doppler form (decision 2026-11-07): use ``physics.py``,
``f_heard = f0 / (1 - v_r / c_eff)`` (FRAMEWORK section 7), NOT the
strategy doc's approximate ``f0 * (1 - v_r / c)``.  All forward curves
go through ``physics.emission_to_arrival`` / ``heard_curve`` semantics:
emission time -> arrival time -> ratio, never a shift of a finished
curve.

No network calls and no hard-coded data paths in this module.  Ground
height is supplied as a callable (tests use a constant; production
passes ``terrain.Terrain.height``).  ``telemetry`` is not imported
(``telemetry.py`` imports ``adsblol``, which is a separate module);
``load_segments`` reads the Parquet tables directly with DuckDB.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .geodesy import LocalFrame
from .physics import Track, emission_to_arrival

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - fallback path, exercised only without numba
    _HAS_NUMBA = False

    def njit(*a, **k):
        def wrap(f):
            return f

        return wrap

    prange = range

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None

C_DEFAULT = 343.0
EARTH_M_PER_DEG_LAT = 111_320.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """Tunable parameters (FRAMEWORK section 9.3, starting values)."""

    grid_spacing_m: float = 200.0
    rmax_m: float = 10_000.0
    # F4 gates: +/-20% around the M1 estimate, widened to 3*sigma (see event_gates).
    d_rel_tol: float = 0.20
    v_rel_tol: float = 0.20
    # F4 distance gate is ASYMMETRIC on the high side only (see event_gates):
    # the low bound stays ev.d - d_tol, but the high bound is
    # ev.d + d_gate_high_factor * d_tol.  A real aircraft that turns or
    # changes speed near its CPA produces a Doppler curve that matches a
    # straight pass at a CLOSER distance than its geometric CPA, so M1's
    # sound-implied distance is systematically below the geometric CPA
    # the gate compares against.  On the G3b tuning set (24 renders of
    # real 2026-11-06 Hudson/New York ADS-B tracks, 55 expected passes
    # M1 matched) the ratio geometric d / sound-implied d is 1.05 at the
    # median, 1.36 at the 75th percentile, 2.48 at the 90th and 5.5 at
    # most; only 6/55 are below 0.8.  Widening the high side by 2.0
    # recovers 8 of the 11 passes the symmetric gate lost (44/55 -> 52/55);
    # x3 adds nothing (52/55).  1.0 restores the old symmetric gate.
    # The fuller fix -- comparing against the straight-pass fit of the
    # flight's forward Doppler curve instead of the geometric CPA -- is
    # deferred: it costs a curve fit per surviving CPA.
    d_gate_high_factor: float = 2.0
    dt_tol_s: float = 5.0  # section 9.3: epsilon for the pair gate
    # F8 sweep (section 9.3): f0 +/-3 Hz, dt +/-5 s.
    f0_sweep_hz: float = 3.0
    f0_step_hz: float = 0.5
    dt_sweep_s: float = 5.0
    dt_step_s: float = 0.5
    refine_spacings_m: tuple = (100.0, 25.0, 5.0)
    refine_window_half_m: tuple = (500.0, 150.0, 50.0)
    c: float = C_DEFAULT
    excess_db_per_km: float = 4.0  # synth.render_scene assumption, for G3
    noise_rms: float = 0.02  # synth.render_scene default
    level_by_class: dict = field(
        default_factory=lambda: {"prop": 1.0, "turboprop": 1.0, "rotor": 1.0,
                                 "jet": 0.3, "unknown": 0.5}
    )
    audibility_scale_db: float = 3.0  # logistic scale for P(audible)
    score_scale_db: float = 10.0  # dB of mean spectrogram score per loglik unit
    # Effective independent frames per event, capped: spectrogram frames
    # are correlated (STFT overlap, smooth Doppler curve, wobble) and
    # the curve shape is set by ~4 parameters, so span/1.02 s (up to
    # ~300) overcounts the evidence by an order of magnitude.  The cap
    # keeps log-likelihoods in the hundreds, not thousands; the
    # remaining overconfidence from systematic errors (track, sound
    # speed, M1 fit) is handled by posterior smoothing below and by
    # the temperature in T4.
    n_eff_cap: float = 50.0
    # Gaussian smoothing of the posterior, in grid cells, before
    # credible regions are computed.  Accounts for systematic location
    # error and for grid discretisation (a 500 m cell centre can be
    # ~350 m from the truth even when it is the correct cell).
    posterior_smooth_sigma_cells: float = 1.0
    temperature: float = 1.0  # section 9.3: likelihood temperature, tuned in T4
    prior: str = "uniform"
    quick_score_min_db: float = 2.0  # drop a blob if any event scores below this
    max_blobs_to_score: int = 20
    # Density-scaled scoring budget (2026-11-09, G3b New York): the
    # number of blobs quick-scored is NOT fixed at max_blobs_to_score.
    # Blobs are ranked by min dt_err (see collapse_blobs), and dt_err
    # is a near-tie at city density -- most blobs are within 0.1 s --
    # so a fixed top-20 cut is near-arbitrary there.  On the G3b
    # New York runs with d_gate_high_factor=2.0 the blobs that won
    # at factor 1.0 were cut: run_01 (43 -> 46 blobs, cut 23 -> 26)
    # lost the single-event blob that located truth (220 m -> 8.9 km),
    # run_04 (84 -> 103 blobs, cut 64 -> 83) lost its winning blob
    # (132 m -> 5.2 km).  The effective budget is
    #   min(max_blobs_to_score_cap,
    #       max(max_blobs_to_score, ceil(blobs_to_score_frac * n_blobs)))
    # so sparse scenes are unchanged (G3a: 6 blobs -> 20) and dense
    # scenes score up to all blobs, bounded by the cap.  Runtime:
    # quick scoring is ~0.02 s/blob, but the marginalised sweep is
    # ~10-20 ms per scored cell, so the cap is what bounds the worst
    # case (run_04-class: ~100 blobs, est. +100-400 s sweep on top of
    # ~500 s of pair gate).  The cap also bounds bench_large, where
    # blobs reach ~4,254 at 50M entries.
    blobs_to_score_frac: float = 1.0
    max_blobs_to_score_cap: int = 100
    max_modes: int = 5
    score_frame_step: int = 2  # subsample spectrogram frames when scoring
    dt_emit_s: float = 0.5  # emission-time step for forward curves in scoring
    # --- Fixes 2026-11-07 (9 bugs, REFERENCE.md section 6) ---
    # Fix 3: candidates per event, closest-first (sorted by CPA distance).
    max_candidates_per_event: int = 12
    # Fix 3: deterministic budget for pair-gate combinations per (cell, subset).
    # NOT density-scaled (2026-11-09): candidate lists are capped at
    # max_candidates_per_event=12 first, so a 2-event combination is at
    # most 12*12=144 < 500 -- the cap only bites subsets of >=3 events,
    # and the G3b run_01/run_04 failures were single-event blobs lost
    # at the blob cut, not pairs lost here.  Pair gate is also ~83%
    # of run_04's 602 s, so scaling this would blow the runtime ceiling.
    max_pair_combos: int = 500
    # Fix 4: enumerate all event subsets up to this many events; above it
    # the event list is pruned (by span) and the pruning is recorded.
    max_subset_events: int = 6
    # Fix 6d: refinement / scale limits are config, not hard-coded slices.
    refine_top_n: int = 3
    # Fix 6b: baro-altitude fallback for missing GPS altitude is OFF by
    # default (different datum).  Set True only with the recorded
    # vertical-uncertainty caveat in load_segments.
    allow_baro_fallback: bool = False
    # Fix 6c/6d: scale guard.  A grid above max_grid_cells, or a CPA
    # estimate above max_cpa_entries, is refused unless allow_large_box
    # is set explicitly -- tile / use the G4 per-day CPA cache instead.
    max_grid_cells: int = 500_000
    max_cpa_entries: int = 5_000_000
    allow_large_box: bool = False
    # Fix 5/6c: mode clustering separation, in ENU metres (not degrees).
    mode_min_separation_m: float = 1000.0
    # --- Jet deferral (2026-11-08, user proposal) ---
    # High jets are ~99% of the early CPA work in the New York 1 h
    # box (~30k entries for props vs ~3.1M for jets, user measured),
    # yet jets are only used for the silence check, which runs on
    # the ~450 surviving cells.  With defer_jets=True, high jets
    # are kept OUT of the early CPA table / pair gate / scoring and
    # their silence is computed on demand, per surviving cell only
    # (see _deferred_jet_silence in run_search).  defer_jets=False
    # reproduces the old behaviour exactly.
    defer_jets: bool = True
    # A jet is "low" (kept early, a genuine candidate) if any of
    # its points is at or below this ellipsoid height.  FRAMEWORK
    # F1 already says jets below ~3 km AGL are candidates.  This
    # is a conservative proxy, not true AGL: it uses ellipsoid
    # height, so in high terrain it keeps more jets early (safe
    # direction); terrain-based AGL is a G4 refinement.
    jet_early_max_alt_m: float = 3000.0

    def to_dict(self):
        d = {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in self.__dict__.items()}
        return d


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    icao: str
    seg: int
    cls: str
    t: np.ndarray  # Unix seconds, strictly increasing
    enu: np.ndarray  # (N, 3) ENU metres in the search frame
    track: Track
    # G3b fix: spatial trim can split one (icao, seg) track into
    # disjoint runs (it leaves the box and re-enters).  Runs are
    # separate Segment objects with the same icao/seg and a
    # sub-segment index; 0 is the whole / first run.  Downstream code
    # indexes segments by list position, so duplicates are safe.
    # Check 3: the unique segment identity is (icao, seg, subseg),
    # reported as "piece" in report.json.  A circling / out-and-back
    # aircraft can genuinely be heard twice -- one piece per event --
    # and every gate (pair_gate dedup, collapse_blobs, silence
    # exclusion) works per piece, i.e. per list index, not per
    # (icao, seg).
    subseg: int = 0


def segments_from_tracks(items):
    """Build Segments from in-memory tracks.

    ``items``: iterable of (icao, seg, cls, Track-in-search-frame-ENU).
    Used by tests and by scripts/g3_synthetic.py, which have no Parquet.
    """
    out = []
    for icao, seg, cls, track in items:
        t = np.asarray(track.t, dtype=float)
        enu = np.asarray(track.position(t), dtype=float)
        out.append(Segment(str(icao), int(seg), str(cls), t, enu, track))
    return out


def trim_segments(segments, frame, lat_min, lat_max, lon_min, lon_max,
                  rmax_m=10_000.0, min_points=5, stats=None, trim_extra_m=3000.0):
    """Trim in-memory segments to the search box expanded by ``rmax_m``.

    G3b fix: a flight crossing the box used to come in whole -- up to
    hundreds of km of it -- even though only the part within ``rmax_m``
    of a grid cell can ever be that cell's CPA.  That inflated every
    segment's bbox (and the old product guard) for nothing.

    For each segment, keep the runs of points inside the expanded box
    (ENU rectangle from the box corners, expanded by ``rmax_m``),
    plus ONE bracketing point either side of each run so the spline
    at the edge is anchored, not extrapolated.  A run that leaves and
    re-enters is split into separate Segment objects (same icao/seg,
    ``subseg`` = run index, reported as ``piece``) rather than joined
    across the gap with a straight line -- joining would invent track
    where there is none.
    Runs shorter than ``min_points`` (after the bracketing points)
    are dropped.  Returns (trimmed_segments, stats); ``stats`` is
    updated in place if given.  Stats keys: points_trimmed_spatially,
    segments_split, runs_dropped_too_short, segments_dropped_by_spatial_trim.

    Trim-edge safety (user check 5): the box is expanded by
    ``rmax_m + trim_extra_m`` (default 3000 m), NOT by rmax alone.
    A CPA right at the trim edge must not be lost: a flight passing
    just outside the 10 km corridor still has its closest point
    within ~rmax of a cell near the box edge, and the track edge
    that contains that CPA can span a reporting gap.  3000 m is
    about one reporting interval at cruise speed (15 s x ~150-200
    m/s) plus margin, so a CPA at or just inside the rmax boundary
    keeps its bracketing points.  The bracketing points are the
    NEAREST points outside the expanded box on each side (the point
    immediately before the run starts / after it ends), not just
    the first kept point -- the spline and the CPA on the edge
    between them are therefore anchored on both sides.  A CPA
    genuinely > rmax from every cell is still excluded by the CPA
    gate itself (see _best_cpa / build_cpa_table), not by trimming.
    """
    st = {"points_trimmed_spatially": 0, "segments_split": 0,
          "runs_dropped_too_short": 0, "segments_dropped_by_spatial_trim": 0}
    if stats is not None:
        stats.update(st)
    if not segments:
        return [], st
    corners = frame.to_enu(
        np.array([lat_min, lat_min, lat_max, lat_max], dtype=float),
        np.array([lon_min, lon_max, lon_min, lon_max], dtype=float),
        np.zeros(4))
    _expand = float(rmax_m) + float(trim_extra_m)
    xmin, xmax = float(corners[:, 0].min()) - _expand, float(corners[:, 0].max()) + _expand
    ymin, ymax = float(corners[:, 1].min()) - _expand, float(corners[:, 1].max()) + _expand
    out = []
    for s in segments:
        enu = np.asarray(s.enu, dtype=float)
        t = np.asarray(s.t, dtype=float)
        inside = ((enu[:, 0] >= xmin) & (enu[:, 0] <= xmax)
                  & (enu[:, 1] >= ymin) & (enu[:, 1] <= ymax))
        # Contiguous runs of inside points.
        runs = []
        i = 0
        n = len(inside)
        while i < n:
            if not inside[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and inside[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        if not runs:
            st["segments_dropped_by_spatial_trim"] += 1
            st["points_trimmed_spatially"] += int(n)
            continue
        if len(runs) > 1:
            st["segments_split"] += int(len(runs) - 1)
        kept_idx = set()
        for run_idx, (a, b) in enumerate(runs):
            # One bracketing point either side, if it exists.
            a2 = max(0, a - 1)
            b2 = min(n - 1, b + 1)
            tk, ek = t[a2:b2 + 1], enu[a2:b2 + 1]
            if len(tk) < min_points:
                st["runs_dropped_too_short"] += 1
                continue
            kept_idx.update(range(a2, b2 + 1))
            out.append(Segment(s.icao, int(s.seg), s.cls, tk, ek,
                               Track(tk, ek), subseg=int(run_idx)))
        st["points_trimmed_spatially"] += int(n - len(kept_idx))
    if stats is not None:
        stats.update(st)
    return out, st


def load_segments_with_stats(points_path, frame, lat_min, lat_max, lon_min, lon_max,
                             min_points=5, allow_baro_fallback=None, config=None,
                             t_start=None, t_end=None, time_margin_s=0.0,
                             trim_spatial=True, rmax_m=None, trim_extra_m=None):
    """Like load_segments, but also returns a stats dict (see load_segments).

    Fix 6b -- GPS (ellipsoidal) altitude vs pressure altitude:
    ``alt_geom_m`` (WGS84 ellipsoid) and ``alt_baro_m`` (pressure altitude,
    standard atmosphere) are different datums; mixing them silently gives
    a vertical error of tens to hundreds of metres, which matters for
    slant distance at ~0.5-2 km.  So:
      * NaN ``alt_geom_m`` *inside* a segment is interpolated in time
        from the finite ``alt_geom_m`` values of the same segment
        (0.8% of points are missing, gaps are small) -- the primary fix.
      * ``alt_baro_m`` is used ONLY if a segment has no finite
        ``alt_geom_m`` at all AND ``allow_baro_fallback`` is True
        (from ``config.allow_baro_fallback`` if set, else False).
        That use is counted in the stats and warned about.
      * Otherwise such a segment is dropped, and counted in the stats
        (``segments_dropped_no_alt``) -- never silently mixed.
      * Points still non-finite after that are dropped and counted.

    G3b fix -- time filter (``t_start`` / ``t_end``, Unix seconds,
    plus ``time_margin_s``): the old loader returned every segment
    of the whole day that touched the box (3,274 near New York for
    one 50 km box), even if it flew 12 hours from the recording.
    Now the SQL hit CTE *and* the main select filter points to
    [t_start - margin, t_end + margin]; a segment that touches the
    box at a different time of day is not loaded at all.  A segment
    partly overlapping the window is kept, trimmed to the window,
    plus ONE bracketing point either side (via LAG/LEAD in SQL) so
    the Track spline at the window edge is anchored.  If both are
    None, behaviour is unchanged (whole day).

    Margin rule for unknown T0: the window must cover the T0
    uncertainty + the recording duration + sound delay (~d/c, up
    to ~30 s at 10 km) + the Δt between events.  Worked example:
    T0 known to ±600 s, recording 1600 s long, events up to 920 s
    apart -> call with t_start = T0 - 600, t_end = T0 + 1600 + 920,
    time_margin_s = 60 (or fold the 600 s into t_start directly).
    If T0 is known exactly, t_start = t0_known, t_end = t0_known
    + duration, margin ~60 s is enough.

    G3b fix -- spatial trim (``trim_spatial``, default True): after
    the time filter, each segment is trimmed to the box expanded by
    ``rmax_m`` (see trim_segments) before its Track is built, so a
    flight crossing the box no longer comes in whole.

    Returns (segments, stats).  Stats keys: points_read, points_kept,
    alt_geom_nan, alt_geom_interpolated, alt_baro_fallback_points,
    segments_dropped_no_alt, segments_dropped_too_short,
    points_dropped_nonfinite, segments_dropped_by_time,
    points_time_filtered_out, points_trimmed_spatially,
    segments_split, runs_dropped_too_short,
    segments_dropped_by_spatial_trim, segments_added_by_expanded_hit
    (user note 4: segments whose only points are within rmax of the
    box edge, picked up by the expanded hit -- 6 jets Hudson, 3
    Manhattan on the real 2026-11-06 day).
    """
    if duckdb is None:
        raise ImportError("duckdb is required to load points.parquet")
    import time as _time
    _t_load0 = _time.time()
    if allow_baro_fallback is None:
        allow_baro_fallback = bool(getattr(config, "allow_baro_fallback", False))
    if rmax_m is None:
        rmax_m = float(getattr(config, "rmax_m", 10_000.0)) if config is not None else 10_000.0
    if trim_extra_m is None:
        trim_extra_m = 3000.0
    stats = {"points_read": 0, "points_kept": 0, "alt_geom_nan": 0,
             "alt_geom_interpolated": 0, "alt_baro_fallback_points": 0,
             "segments_dropped_no_alt": 0, "segments_dropped_too_short": 0,
             "points_dropped_nonfinite": 0,
             "segments_dropped_by_time": 0, "points_time_filtered_out": 0,
             "points_trimmed_spatially": 0, "segments_split": 0,
             "runs_dropped_too_short": 0, "segments_dropped_by_spatial_trim": 0,
             "segments_added_by_expanded_hit": 0}
    path = str(Path(points_path))
    lit = path.replace("'", "''")
    # User note 4 (2026-11-08): the SQL hit used to select only
    # segments with a point INSIDE the box, so a flight passing
    # within rmax (10 km) of the edge -- close enough to be some
    # cell's CPA, and to matter for the silence check -- was left
    # out entirely (user measured 6 jets Hudson, 3 Manhattan).
    # The hit is now the box EXPANDED by rmax_m, in degrees
    # (cos(lat) for longitude).  The trim to box+rmax+3 km is
    # unchanged, so the extra segments are kept only where they
    # can actually be a CPA; a flight 20 km away is still not hit.
    _lat_mid = 0.5 * (float(lat_min) + float(lat_max))
    _dlat = float(rmax_m) / EARTH_M_PER_DEG_LAT
    _dlon = float(rmax_m) / (EARTH_M_PER_DEG_LAT
                             * max(0.2, math.cos(math.radians(_lat_mid))))
    _lat_e0, _lat_e1 = float(lat_min) - _dlat, float(lat_max) + _dlat
    _lon_e0, _lon_e1 = float(lon_min) - _dlon, float(lon_max) + _dlon
    try:
        _eh = duckdb.sql(f"""
            SELECT (SELECT COUNT(*) FROM (SELECT DISTINCT icao, seg FROM '{lit}'
                     WHERE lat BETWEEN {_lat_e0} AND {_lat_e1}
                       AND lon BETWEEN {_lon_e0} AND {_lon_e1}))
                   - (SELECT COUNT(*) FROM (SELECT DISTINCT icao, seg FROM '{lit}'
                     WHERE lat BETWEEN {lat_min} AND {lat_max}
                       AND lon BETWEEN {lon_min} AND {lon_max})) AS added
        """).df()
        stats["segments_added_by_expanded_hit"] = int(_eh["added"].iloc[0])
    except Exception:
        stats["segments_added_by_expanded_hit"] = 0
    # Time window (Unix), with margin.  None means open-ended / no filter.
    _w0 = float(t_start) - float(time_margin_s) if t_start is not None else None
    _w1 = float(t_end) + float(time_margin_s) if t_end is not None else None
    _time_filter = _w0 is not None or _w1 is not None

    def _t_cond(alias="t", w0=None, w1=None):
        parts = []
        if w0 is not None:
            parts.append(f"{alias} >= {w0!r}")
        if w1 is not None:
            parts.append(f"{alias} <= {w1!r}")
        return " AND ".join(parts) if parts else "TRUE"

    if _time_filter:
        # Stats first: how many box-touching segments / points fall
        # outside the window.  One extra columnar scan; the loader
        # runs once per day/box, so this is cheap next to the load.
        try:
            _st = duckdb.sql(f"""
                WITH box_hit AS (SELECT DISTINCT icao, seg FROM '{lit}'
                                 WHERE lat BETWEEN {_lat_e0} AND {_lat_e1}
                                   AND lon BETWEEN {_lon_e0} AND {_lon_e1}),
                     time_hit AS (SELECT DISTINCT icao, seg FROM '{lit}'
                                  WHERE lat BETWEEN {_lat_e0} AND {_lat_e1}
                                    AND lon BETWEEN {_lon_e0} AND {_lon_e1}
                                    AND {_t_cond('t', _w0, _w1)})
                SELECT (SELECT COUNT(*) FROM box_hit)
                       - (SELECT COUNT(*) FROM time_hit) AS seg_dropped,
                       (SELECT COUNT(*) FROM '{lit}' p JOIN box_hit USING (icao, seg)
                        WHERE NOT ({_t_cond('p.t', _w0, _w1)})) AS pts_out
            """).df()
            stats["segments_dropped_by_time"] = int(_st["seg_dropped"].iloc[0])
            _pts_out = int(_st["pts_out"].iloc[0])
        except Exception:
            _pts_out = 0
        # Select both altitudes separately -- the old code collapsed them in
        # SQL with isnan(), which hid whether the value was geometric or baro.
        # Bracketing: keep the one point just before the window (its
        # successor is in/after the window) and the one just after it.
        _keep = f"(t BETWEEN {_w0!r} AND {_w1!r})" if (_w0 is not None and _w1 is not None) else f"({_t_cond('t', _w0, _w1)})"
        _brack = []
        if _w0 is not None:
            _brack.append(f"(t < {_w0!r} AND next_t >= {_w0!r})")
        if _w1 is not None:
            _brack.append(f"(t > {_w1!r} AND prev_t <= {_w1!r})")
        if _brack:
            _keep = "(" + _keep + " OR " + " OR ".join(_brack) + ")"
        df = duckdb.sql(f"""
            WITH hit AS (SELECT DISTINCT icao, seg FROM '{lit}'
                         WHERE lat BETWEEN {_lat_e0} AND {_lat_e1}
                           AND lon BETWEEN {_lon_e0} AND {_lon_e1}
                           AND {_t_cond('t', _w0, _w1)}),
                 raw AS (SELECT p.icao, p.seg, p.t, p.lat, p.lon,
                                p.alt_geom_m AS alt_geom, p.alt_baro_m AS alt_baro,
                                CAST(p.cls AS VARCHAR) AS cls,
                                LAG(p.t) OVER (PARTITION BY p.icao, p.seg ORDER BY p.t) AS prev_t,
                                LEAD(p.t) OVER (PARTITION BY p.icao, p.seg ORDER BY p.t) AS next_t
                         FROM '{lit}' p JOIN hit USING (icao, seg))
            SELECT icao, seg, t, lat, lon, alt_geom, alt_baro, cls FROM raw
            WHERE {_keep}
            ORDER BY icao, seg, t
        """).df()
        # points_time_filtered_out, net of the bracketing points kept.
        if len(df):
            _kept_outside = int((((df.t.values < _w0) if _w0 is not None else False)
                                 | ((df.t.values > _w1) if _w1 is not None else False)).sum()) \
                if (_w0 is not None or _w1 is not None) else 0
            stats["points_time_filtered_out"] = int(max(0, _pts_out - _kept_outside))
    else:
        # Select both altitudes separately -- the old code collapsed them in
        # SQL with isnan(), which hid whether the value was geometric or baro.
        df = duckdb.sql(f"""
            WITH hit AS (SELECT DISTINCT icao, seg FROM '{lit}'
                         WHERE lat BETWEEN {_lat_e0} AND {_lat_e1}
                           AND lon BETWEEN {_lon_e0} AND {_lon_e1})
            SELECT p.icao, p.seg, p.t, p.lat, p.lon,
                   p.alt_geom_m AS alt_geom, p.alt_baro_m AS alt_baro,
                   CAST(p.cls AS VARCHAR) AS cls
            FROM '{lit}' p JOIN hit USING (icao, seg)
            ORDER BY p.icao, p.seg, p.t
        """).df()
    if len(df) == 0:
        stats["load_seconds"] = round(_time.time() - _t_load0, 3)
        return [], stats
    stats["points_read"] = int(len(df))
    alt_geom = np.asarray(df.alt_geom.values, dtype=float)
    alt_baro = np.asarray(df.alt_baro.values, dtype=float)
    stats["alt_geom_nan"] = int(np.isnan(alt_geom).sum())
    # Height to use, per row, before ENU conversion.
    h = alt_geom.copy()
    keys = np.array([f"{a}:{b}" for a, b in zip(df.icao.values, df.seg.values)])
    bounds = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    no_alt_segments = set()
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg_geom = alt_geom[a:b]
        finite = np.isfinite(seg_geom)
        if finite.any():
            if (~finite).any():
                # Interpolate in time (t is already sorted within a segment).
                t_seg = np.asarray(df.t.values[a:b], dtype=float)
                h[a:b] = np.interp(t_seg, t_seg[finite], seg_geom[finite])
                stats["alt_geom_interpolated"] += int((~finite).sum())
        else:
            # No geometric altitude anywhere in this segment.
            if allow_baro_fallback:
                import warnings
                warnings.warn(
                    f"load_segments: segment {df.icao.values[a]}:{df.seg.values[a]} "
                    "has no alt_geom_m; using alt_baro_m (pressure altitude, "
                    "different datum) as a last resort")
                h[a:b] = alt_baro[a:b]
                stats["alt_baro_fallback_points"] += int(b - a)
            else:
                # Mark for dropping below: h stays NaN.
                h[a:b] = np.nan
                stats["segments_dropped_no_alt"] += 1
                no_alt_segments.add((int(a), int(b)))
    enu = frame.to_enu(df.lat.values, df.lon.values, h)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (int(a), int(b)) in no_alt_segments:
            continue  # already counted in segments_dropped_no_alt
        t = np.asarray(df.t.values[a:b], dtype=float)
        keep = np.r_[True, np.diff(t) > 0.1]
        if int(keep.sum()) < min_points:
            stats["segments_dropped_too_short"] += 1
            continue
        tk, ek = t[keep], enu[a:b][keep]
        ok = np.isfinite(ek).all(axis=1) & np.isfinite(tk)
        stats["points_dropped_nonfinite"] += int((~ok).sum())
        if int(ok.sum()) < min_points:
            stats["segments_dropped_too_short"] += 1
            continue
        out.append(Segment(str(df.icao.values[a]), int(df.seg.values[a]),
                           str(df.cls.values[a]), tk[ok], ek[ok],
                           Track(tk[ok], ek[ok])))
    # G3b fix: trim long segments to the box + rmax + trim_extra_m
    # BEFORE their Tracks are used for CPA work (see trim_segments).
    # A flight crossing the box no longer comes in whole.
    if trim_spatial and out:
        out, _trim_st = trim_segments(out, frame, lat_min, lat_max, lon_min, lon_max,
                                      rmax_m=float(rmax_m), min_points=min_points,
                                      stats=stats, trim_extra_m=float(trim_extra_m))
    stats["points_kept"] = int(sum(len(s.t) for s in out))
    stats["load_seconds"] = round(_time.time() - _t_load0, 3)
    return out, stats


def load_segments(points_path, frame, lat_min, lat_max, lon_min, lon_max,
                  min_points=5, allow_baro_fallback=None, config=None, stats=None,
                  t_start=None, t_end=None, time_margin_s=0.0,
                  trim_spatial=True, rmax_m=None, trim_extra_m=None):
    """Flight segments with any point in the box, in the local ENU frame.

    Reads ``points.parquet`` (M2a output) with DuckDB.  NOTE on the
    NaN-altitude bug: ``alt_geom_m`` is stored as NaN, not NULL, so
    ``coalesce(alt_geom_m, alt_baro_m)`` does NOT replace it.  Use
    ``isnan()`` explicitly (see also the codebase reference, section 4).

    Fix 6b: missing ``alt_geom_m`` is interpolated in time within its
    segment; ``alt_baro_m`` is NOT silently substituted (different
    datum) -- see load_segments_with_stats.  Pass ``stats={}`` to get
    the counts back (interpolated / baro-fallback / dropped).

    G3b: ``t_start`` / ``t_end`` (Unix) + ``time_margin_s`` filter
    segments to the recording window (see load_segments_with_stats
    for the margin rule); ``trim_spatial`` trims long segments to
    the box + ``rmax_m`` + ``trim_extra_m`` (default 3000 m safety
    margin, see trim_segments).  Both default to the old behaviour
    when off (t_start/t_end None, trim_spatial True is the new
    default -- it only removes points that cannot be any cell's CPA).
    """
    segs, st = load_segments_with_stats(
        points_path, frame, lat_min, lat_max, lon_min, lon_max,
        min_points=min_points, allow_baro_fallback=allow_baro_fallback,
        config=config, t_start=t_start, t_end=t_end,
        time_margin_s=time_margin_s, trim_spatial=trim_spatial,
        rmax_m=rmax_m, trim_extra_m=trim_extra_m)
    if stats is not None:
        stats.update(st)
    return segs


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    enu: np.ndarray  # (N, 3), row-major: north-to-south rows, west-to-east cols
    lats: np.ndarray  # (N,) cell-centre latitudes
    lons: np.ndarray  # (N,) cell-centre longitudes
    shape: tuple  # (ny, nx)
    spacing_m: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    # Fix 6b: count of ground heights that were NaN (ocean / missing tile)
    # and were replaced (see make_grid).  0 in the normal case.
    height_nan_count: int = 0


def _height_array(height_fn, lats, lons):
    if height_fn is None:
        return np.zeros_like(lats, dtype=float)
    if isinstance(height_fn, (int, float)):
        return np.full_like(lats, float(height_fn), dtype=float)
    h = np.asarray(height_fn(lats, lons), dtype=float)
    return np.broadcast_to(h, lats.shape).astype(float)


def make_grid(lat_min, lat_max, lon_min, lon_max, spacing_m, frame, height_fn=None):
    """Regular grid in ENU metres, converted back to lat/lon per cell.

    Cells are laid out on a local ENU rectangle centred on the box centre,
    so spacing is in true metres (not degrees); lat/lon per cell comes from
    ``frame.to_llh`` and is therefore exact for the frame.
    """
    lat0 = 0.5 * (lat_min + lat_max)
    lon0 = 0.5 * (lon_min + lon_max)
    # Box size in metres (spherical approximation is fine for sizing only).
    width_m = (lon_max - lon_min) * EARTH_M_PER_DEG_LAT * math.cos(math.radians(lat0))
    height_m = (lat_max - lat_min) * EARTH_M_PER_DEG_LAT
    nx = max(1, int(round(width_m / spacing_m)))
    ny = max(1, int(round(height_m / spacing_m)))
    xs = (np.arange(nx) + 0.5) * (width_m / nx) - width_m / 2.0
    ys = height_m / 2.0 - (np.arange(ny) + 0.5) * (height_m / ny)
    xx, yy = np.meshgrid(xs, ys)  # (ny, nx), y north, x east, origin = box centre
    centre_enu = frame.to_enu(np.array([lat0]), np.array([lon0]),
                              np.array([0.0]))[0]
    enu = np.column_stack([xx.ravel() + centre_enu[0],
                           yy.ravel() + centre_enu[1],
                           np.zeros(xx.size)])
    lats, lons, _h = frame.to_llh(enu)
    ground = _height_array(height_fn, lats, lons)
    # Fix 6b: a NaN ground height (ocean / missing DEM tile) must not
    # produce NaN ENU cells silently.  Replace NaNs with the median
    # finite height in this grid (0 m if there is none) and record how
    # many were replaced on the Grid, so the caller can see it.
    nan_count = int(np.isnan(ground).sum())
    if nan_count:
        finite = ground[np.isfinite(ground)]
        fill = float(np.median(finite)) if finite.size else 0.0
        ground = np.where(np.isfinite(ground), ground, fill)
        import warnings
        warnings.warn(f"make_grid: {nan_count}/{ground.size} ground heights "
                      f"were NaN; filled with {fill:.1f} m")
    # Ground height in ENU: recompute ENU with the true ground height.
    enu = frame.to_enu(lats, lons, ground)
    return Grid(enu=enu, lats=np.asarray(lats), lons=np.asarray(lons),
                shape=(ny, nx), spacing_m=float(spacing_m),
                lat_min=float(lat_min), lat_max=float(lat_max),
                lon_min=float(lon_min), lon_max=float(lon_max),
                height_nan_count=nan_count)


def grid_cell_latlon(grid, idx):
    return float(grid.lats[idx]), float(grid.lons[idx])


# ---------------------------------------------------------------------------
# F3/F4: CPA table
# ---------------------------------------------------------------------------

def event_gates(perception, config):
    """Per-event (d_lo, d_hi, v_lo, v_hi) from the M1 estimates.

    FRAMEWORK section 9.3: bounds +/-20% around the estimate, widened to
    3*sigma when the fit uncertainty is larger than that.

    The distance gate is asymmetric on the high side only.  M1 fits a
    straight-pass Doppler model, but a real aircraft that turns or
    changes speed near its CPA sounds like a straight pass at a CLOSER
    distance than its geometric CPA, so the sound-implied distance M1
    reports is systematically below the geometric CPA distance this
    gate compares against (G3b: geometric / sound-implied ratio 1.05
    median, 2.48 at the 90th percentile).  The low bound therefore
    stays ev.d - d_tol, while the high bound is
    ev.d + config.d_gate_high_factor * d_tol (default factor 2.0;
    1.0 restores the symmetric gate).  The speed gate stays symmetric:
    speed is not biased this way (M1 v tracks the label v closely on
    G3b).  Cost: a wider high side marks more CPAs per event, so F5
    and scoring see more hypotheses; M4 forward-curve scoring uses the
    real track and should still reject wrong cells.
    """
    n = len(perception.events)
    d_lo = np.zeros(n); d_hi = np.zeros(n); v_lo = np.zeros(n); v_hi = np.zeros(n)
    for k, ev in enumerate(perception.events):
        d_tol = max(config.d_rel_tol * ev.d, 3.0 * max(ev.sigma_d, 0.0))
        v_tol = max(config.v_rel_tol * ev.v, 3.0 * max(ev.sigma_v, 0.0))
        d_lo[k] = max(30.0, ev.d - d_tol)
        d_hi[k] = ev.d + config.d_gate_high_factor * d_tol
        v_lo[k] = max(10.0, ev.v - v_tol)
        v_hi[k] = ev.v + v_tol
    return d_lo, d_hi, v_lo, v_hi


@dataclass
class CpaTable:
    """Closest approaches, in CSR form per grid cell.

    For cell ``g``, entries ``cell_ptr[g]:cell_ptr[g+1]`` are the segments
    whose (single, global) CPA to that cell is within ``rmax_m``.
    ``mask`` is a bitmask over detected events: bit k set when this CPA
    could be event k (distance and speed in event k's gate).  ``mask==0``
    entries are silence-pool only: they can penalise a cell in F6 but
    never match an event.
    """
    cell_ptr: np.ndarray
    seg: np.ndarray
    t_cpa: np.ndarray  # Unix seconds, CPA (emission) time
    t_arr: np.ndarray  # Unix seconds, t_cpa + d/c
    d: np.ndarray
    v: np.ndarray
    mask: np.ndarray
    n_cells: int
    n_events: int

    def entries_for_cell(self, g):
        a, b = int(self.cell_ptr[g]), int(self.cell_ptr[g + 1])
        return a, b


def _segment_bboxes(segments, rmax):
    sbox = np.zeros((len(segments), 4))
    for i, s in enumerate(segments):
        sbox[i] = [s.enu[:, 0].min() - rmax, s.enu[:, 0].max() + rmax,
                   s.enu[:, 1].min() - rmax, s.enu[:, 1].max() + rmax]
    return sbox


def split_early_deferred(segments, config):
    """Split segments into (early, deferred) for jet deferral.

    Early = prop / turboprop / rotor / unknown, PLUS jets with any
    point at or below ``config.jet_early_max_alt_m`` (low jets are
    genuine candidates -- FRAMEWORK F1: jets below ~3 km AGL can
    be heard as an event).  Deferred = jets with every point above
    the threshold (high jets: silence-pool only).  Unknown-class
    aircraft stay early (many are small planes; the FAA registry
    is incomplete).  With ``config.defer_jets`` False, everything
    is early and nothing is deferred (old behaviour).

    The altitude proxy is ellipsoid height (``Segment.enu[:, 2]``),
    NOT height above ground: in high terrain it keeps more jets
    early than true AGL would -- conservative, safe direction.
    It is also frame-dependent: enu z equals ellipsoid height
    only at the frame origin, and falls by ~d^2/(2R) away from it
    (up to ~80 m across a 50 km box, ~200 m at 50 km from the
    origin), so a jet near the threshold can be kept early in one
    frame and deferred in another.  That only ever keeps MORE
    jets early, never defers a truly low jet -- safe direction.
    Terrain-based AGL is a G4 refinement, not built here.
    """
    if config is None or not getattr(config, "defer_jets", False):
        return list(segments), []
    threshold = float(getattr(config, "jet_early_max_alt_m", 3000.0))
    early, deferred = [], []
    for s in segments:
        if s.cls == "jet" and float(np.min(np.asarray(s.enu)[:, 2])) > threshold:
            deferred.append(s)
        else:
            early.append(s)
    return early, deferred


def _best_cpa_python(gx, gy, gz, pts, tms, a, b, rmax2):
    """Pure-python _best_cpa (same semantics as the njit version):
    global CPA of one segment to one point, piecewise-linear track.
    Returns (t_cpa, d, v, found).  Used for deferred-jet silence
    when numba is unavailable."""
    best_d2 = rmax2
    best_t = 0.0
    best_v = 0.0
    found = False
    for k in range(a, b - 1):
        ex = pts[k + 1, 0] - pts[k, 0]
        ey = pts[k + 1, 1] - pts[k, 1]
        ez = pts[k + 1, 2] - pts[k, 2]
        L2 = ex * ex + ey * ey + ez * ez
        if L2 <= 0.0:
            continue
        dt = tms[k + 1] - tms[k]
        if dt <= 0.0:
            continue
        u = ((gx - pts[k, 0]) * ex + (gy - pts[k, 1]) * ey
             + (gz - pts[k, 2]) * ez) / L2
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        px = pts[k, 0] + u * ex
        py = pts[k, 1] + u * ey
        pz = pts[k, 2] + u * ez
        d2 = (px - gx) ** 2 + (py - gy) ** 2 + (pz - gz) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_t = tms[k] + u * dt
            best_v = math.sqrt(L2) / dt
            found = True
    if found:
        return best_t, math.sqrt(best_d2), best_v, True
    return 0.0, 0.0, 0.0, False


def _jet_cpa_entries(cell_enu, deferred_segments, packed, config):
    """Deferred-jet CPAs to one cell: [(t_arr, d, cls), ...].

    One entry per deferred jet whose global CPA to the cell is
    within rmax -- exactly the entry that jet would have had in
    the full CPA table for this cell, computed with the same
    _best_cpa.  ``packed`` is _pack_segments(deferred_segments).
    """
    if not deferred_segments:
        return []
    pts, tms, ptr = packed
    gx, gy, gz = float(cell_enu[0]), float(cell_enu[1]), float(cell_enu[2])
    rmax2 = float(config.rmax_m) ** 2
    out = []
    for j, seg in enumerate(deferred_segments):
        a, b = int(ptr[j]), int(ptr[j + 1])
        if b - a < 2:
            continue
        if _HAS_NUMBA:
            t_cpa, d, _v, found = _best_cpa(gx, gy, gz, pts, tms, a, b, rmax2)
        else:
            t_cpa, d, _v, found = _best_cpa_python(gx, gy, gz, pts, tms, a, b, rmax2)
        if found:
            out.append((float(t_cpa) + float(d) / float(config.c),
                        float(d), seg.cls))
    return out


def corridor_cpa_estimate(segments, grid, rmax):
    """Corridor-based CPA work estimate (G3b fix).

    The old guard estimated ``n_cells * n_segments`` -- the cost of
    a brute-force grid x segments scan, i.e. exactly the O(N*M) the
    funnel was designed to avoid.  The real work in
    ``_cpa_count`` / ``_cpa_fill`` is different: for each cell it
    checks each segment's bbox and only calls ``_best_cpa`` when the
    cell lies inside that segment's bbox expanded by ``rmax``.  So
    the work per segment is (cells in its corridor bbox, clipped
    to the grid), not (all cells).

    For each segment this counts the grid cell centres whose x lies
    in [xmin-rmax, xmax+rmax] and whose y lies in
    [ymin-rmax, ymax+rmax], using the grid's own x/y centre
    coordinates (row-major, north-to-south rows).  It is an upper
    bound on the number of ``_best_cpa`` calls (cells in the bbox
    whose CPA is beyond ``rmax`` still cost a call, cells outside
    cost only the bbox check).  Returns (total, per_segment list,
    breakdown dict).

    Jets are counted (user check 4): jets are not candidates, but
    they still sit in the CPA table for the silence check -- in the
    user's New York box 2,884 of 3,274 segments are jets, most of
    the work there.  FRAMEWORK F1 already says jets feed only the
    silence check.  So the estimate counts EVERY segment at full
    cost and reports the split separately: ``breakdown`` has
    ``candidate`` (prop / turboprop / rotor, the classes that can
    match an event), ``silence_only`` (jet only), and ``unknown``
    (any other class, in practice ``"unknown"``) corridor cells,
    plus the segment counts for each.  Unknown-class segments are
    NOT silence-only: the search treats them as possible matches
    (``level_by_class["unknown"] = 0.5`` in audibility_prob, and
    pair_gate / scoring do not filter by class, so an unknown
    segment can match an event exactly like a prop can).  The
    three buckets sum to the total -- candidate + silence_only +
    unknown == total, and the three segment counts sum to
    n_segments.  The split is reporting only -- the guard uses
    the TOTAL, so the estimator never silently assumes jets are free.
    No slimming of jet table entries was implemented: slimming
    (fewer fields / coarser gates for silence-only rows) was
    considered and deliberately NOT done -- it would change CPA
    semantics for the silence check, and the full-entry estimate
    already passes the guard for the real boxes, so there is
    nothing to gain by risking it.  If G4 ever needs it, it must
    be measured against the full-entry baseline first.

    Example (user's G3b numbers, 50 km box at 200 m = 62,500 cells):
    311 segments -> product 19.4M, corridor typically <1M after
    time filter + trim; 3,274 segments -> product 204.6M, corridor
    typically a few M after time filter + trim.  Measured on the
    real 2026-11-06 day, 1 h window 11:30-12:30 UTC (user, 2026-11-08):
    Manhattan 152 segments, corridor estimate 3.1M, actual table
    2.2M; Teterboro 124 segments, 2.9M / 1.8M.  A 1 h window
    already uses ~3.1M of the 5M cap near New York, and ~99% of
    that is jets for the silence check -- with only ~1 h time
    knowledge those boxes have little headroom.  (An earlier
    comment here said "~100k" for that case; that was ~30x too
    low and is corrected above.)  The guard in run_search uses
    this estimate, not the product.
    """
    if not segments:
        return 0, [], {"candidate": 0, "silence_only": 0, "unknown": 0,
                       "n_candidate_segments": 0,
                       "n_silence_only_segments": 0,
                       "n_unknown_segments": 0,
                       "n_segments": 0}
    ny, nx = grid.shape
    # Grid x centres (west -> east, first row) and y centres
    # (north -> south, first column), as actually laid out.
    enu2 = np.asarray(grid.enu).reshape(ny, nx, 3)
    xs = np.sort(np.asarray(enu2[0, :, 0], dtype=float))
    ys = np.sort(np.asarray(enu2[:, 0, 1], dtype=float))
    sbox = _segment_bboxes(segments, float(rmax))
    per_seg = []
    for i in range(len(segments)):
        x0, x1, y0, y1 = sbox[i]
        nx_in = int(np.searchsorted(xs, x1, side="right")
                    - np.searchsorted(xs, x0, side="left"))
        ny_in = int(np.searchsorted(ys, y1, side="right")
                    - np.searchsorted(ys, y0, side="left"))
        per_seg.append(int(max(0, nx_in) * max(0, ny_in)))
    # Candidate vs silence-only vs unknown split (reporting only;
    # all count at full cost in the total).  Jets below ~3 km AGL
    # can in principle be heard (FRAMEWORK F1), so this split is a
    # reporting convention, not a gate -- nothing is excluded here.
    # Unknown-class segments are their own bucket: they are NOT
    # silence-only, because the search treats them as possible
    # matches (level_by_class["unknown"] = 0.5, and no gate filters
    # by class), so counting them under silence_only understated
    # the candidate pool (user note 3, 2026-11-08).
    _cand_cls = {"prop", "turboprop", "rotor"}
    _jet_cls = {"jet"}
    cand = sum(p for p, s in zip(per_seg, segments) if s.cls in _cand_cls)
    sil = sum(p for p, s in zip(per_seg, segments) if s.cls in _jet_cls)
    unk = int(sum(per_seg) - cand - sil)
    breakdown = {"candidate": int(cand), "silence_only": int(sil),
                 "unknown": int(unk),
                 "n_candidate_segments": int(sum(
                     1 for s in segments if s.cls in _cand_cls)),
                 "n_silence_only_segments": int(sum(
                     1 for s in segments if s.cls in _jet_cls)),
                 "n_unknown_segments": int(sum(
                     1 for s in segments
                     if s.cls not in _cand_cls and s.cls not in _jet_cls)),
                 "n_segments": int(len(segments))}
    return int(sum(per_seg)), per_seg, breakdown


def _pack_segments(segments):
    if not segments:
        return (np.zeros((0, 3)), np.zeros(0), np.zeros(1, dtype=np.int64))
    pts = np.concatenate([s.enu for s in segments]).astype(float)
    tms = np.concatenate([s.t for s in segments]).astype(float)
    ptr = np.zeros(len(segments) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(s.t) for s in segments])
    return pts, tms, ptr


if _HAS_NUMBA:
    @njit(cache=True)
    def _best_cpa(gx, gy, gz, pts, tms, a, b, rmax2):
        """Global CPA of one segment to one point, piecewise-linear track.

        Returns (t_cpa, d, v, found).  The closest point on each edge is
        found by projection; the minimum over edges is the CPA.  This is
        the gating approximation -- the spline Track is used for scoring.
        """
        best_d2 = rmax2
        best_t = 0.0
        best_v = 0.0
        found = False
        for k in range(a, b - 1):
            ex = pts[k + 1, 0] - pts[k, 0]
            ey = pts[k + 1, 1] - pts[k, 1]
            ez = pts[k + 1, 2] - pts[k, 2]
            L2 = ex * ex + ey * ey + ez * ez
            if L2 <= 0.0:
                continue
            dt = tms[k + 1] - tms[k]
            if dt <= 0.0:
                continue
            u = ((gx - pts[k, 0]) * ex + (gy - pts[k, 1]) * ey
                 + (gz - pts[k, 2]) * ez) / L2
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            px = pts[k, 0] + u * ex
            py = pts[k, 1] + u * ey
            pz = pts[k, 2] + u * ez
            d2 = (px - gx) ** 2 + (py - gy) ** 2 + (pz - gz) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_t = tms[k] + u * dt
                best_v = np.sqrt(L2) / dt
                found = True
        if found:
            return best_t, np.sqrt(best_d2), best_v, True
        return 0.0, 0.0, 0.0, False

    @njit(parallel=True, cache=True)
    def _cpa_count(grid, pts, tms, ptr, sbox, rmax2, counts):
        for g in prange(grid.shape[0]):
            gx, gy, gz = grid[g, 0], grid[g, 1], grid[g, 2]
            c = 0
            for s in range(ptr.shape[0] - 1):
                if gx < sbox[s, 0] or gx > sbox[s, 1] or gy < sbox[s, 2] or gy > sbox[s, 3]:
                    continue
                a, b = ptr[s], ptr[s + 1]
                if b - a < 2:
                    continue
                _t, _d, _v, found = _best_cpa(gx, gy, gz, pts, tms, a, b, rmax2)
                if found:
                    c += 1
            counts[g] = c

    @njit(parallel=True, cache=True)
    def _cpa_fill(grid, pts, tms, ptr, sbox, rmax2, d_lo, d_hi, v_lo, v_hi,
                  c, offs, o_seg, o_t, o_d, o_v, o_mask):
        n_ev = d_lo.shape[0]
        for g in prange(grid.shape[0]):
            gx, gy, gz = grid[g, 0], grid[g, 1], grid[g, 2]
            w = offs[g]
            for s in range(ptr.shape[0] - 1):
                if gx < sbox[s, 0] or gx > sbox[s, 1] or gy < sbox[s, 2] or gy > sbox[s, 3]:
                    continue
                a, b = ptr[s], ptr[s + 1]
                if b - a < 2:
                    continue
                t_cpa, d, v, found = _best_cpa(gx, gy, gz, pts, tms, a, b, rmax2)
                if not found:
                    continue
                mask = 0
                for k in range(n_ev):
                    if d_lo[k] <= d <= d_hi[k] and v_lo[k] <= v <= v_hi[k]:
                        mask |= (1 << k)
                o_seg[w] = s
                o_t[w] = t_cpa
                o_d[w] = d
                o_v[w] = v
                o_mask[w] = mask
                w += 1


def _cpa_table_python(grid, segments, d_lo, d_hi, v_lo, v_hi, config):
    """Pure-python fallback (same semantics as the numba path)."""
    pts, tms, ptr = _pack_segments(segments)
    sbox = _segment_bboxes(segments, config.rmax_m)
    rmax2 = config.rmax_m ** 2
    rows = [[] for _ in range(len(grid.enu))]
    for g, (gx, gy, gz) in enumerate(grid.enu):
        for s in range(len(segments)):
            if gx < sbox[s, 0] or gx > sbox[s, 1] or gy < sbox[s, 2] or gy > sbox[s, 3]:
                continue
            a, b = int(ptr[s]), int(ptr[s + 1])
            best_d2, best_t, best_v, found = rmax2, 0.0, 0.0, False
            for k in range(a, b - 1):
                ex, ey, ez = pts[k + 1] - pts[k]
                L2 = float(ex * ex + ey * ey + ez * ez)
                if L2 <= 0:
                    continue
                dt = tms[k + 1] - tms[k]
                if dt <= 0:
                    continue
                u = float(np.clip(((gx - pts[k, 0]) * (pts[k + 1, 0] - pts[k, 0])
                                   + (gy - pts[k, 1]) * (pts[k + 1, 1] - pts[k, 1])
                                   + (gz - pts[k, 2]) * (pts[k + 1, 2] - pts[k, 2])) / L2, 0, 1))
                px = pts[k, 0] + u * (pts[k + 1, 0] - pts[k, 0])
                py = pts[k, 1] + u * (pts[k + 1, 1] - pts[k, 1])
                pz = pts[k, 2] + u * (pts[k + 1, 2] - pts[k, 2])
                d2 = (px - gx) ** 2 + (py - gy) ** 2 + (pz - gz) ** 2
                if d2 < best_d2:
                    best_d2, best_t, best_v, found = d2, tms[k] + u * dt, math.sqrt(L2) / dt, True
            if found:
                d = math.sqrt(best_d2)
                mask = 0
                for k in range(len(d_lo)):
                    if d_lo[k] <= d <= d_hi[k] and v_lo[k] <= best_v <= v_hi[k]:
                        mask |= (1 << k)
                rows[g].append((s, best_t, d, best_v, mask))
    cell_ptr = np.zeros(len(grid.enu) + 1, dtype=np.int64)
    for g, r in enumerate(rows):
        cell_ptr[g + 1] = cell_ptr[g] + len(r)
    n = int(cell_ptr[-1])
    seg = np.zeros(n, dtype=np.int64); t_cpa = np.zeros(n); d_a = np.zeros(n)
    v_a = np.zeros(n); mask_a = np.zeros(n, dtype=np.int64)
    for g, r in enumerate(rows):
        for j, (s, tc, d, v, m) in enumerate(r):
            i = int(cell_ptr[g]) + j
            seg[i] = s; t_cpa[i] = tc; d_a[i] = d; v_a[i] = v; mask_a[i] = m
    return CpaTable(cell_ptr=cell_ptr, seg=seg, t_cpa=t_cpa,
                    t_arr=t_cpa + d_a / config.c, d=d_a, v=v_a, mask=mask_a,
                    n_cells=len(grid.enu), n_events=len(d_lo))


def build_cpa_table(grid, segments, perception, config):
    """F3/F4: CPA table for one grid (see module docstring)."""
    d_lo, d_hi, v_lo, v_hi = event_gates(perception, config)
    if not segments:
        return CpaTable(cell_ptr=np.zeros(len(grid.enu) + 1, dtype=np.int64),
                        seg=np.zeros(0, dtype=np.int64), t_cpa=np.zeros(0),
                        t_arr=np.zeros(0), d=np.zeros(0), v=np.zeros(0),
                        mask=np.zeros(0, dtype=np.int64),
                        n_cells=len(grid.enu), n_events=len(perception.events))
    if not _HAS_NUMBA:
        return _cpa_table_python(grid, segments, d_lo, d_hi, v_lo, v_hi, config)
    pts, tms, ptr = _pack_segments(segments)
    sbox = _segment_bboxes(segments, config.rmax_m)
    counts = np.zeros(len(grid.enu), dtype=np.int64)
    _cpa_count(grid.enu, pts, tms, ptr, sbox, config.rmax_m ** 2, counts)
    cell_ptr = np.zeros(len(grid.enu) + 1, dtype=np.int64)
    np.cumsum(counts, out=cell_ptr[1:])
    n = int(cell_ptr[-1])
    o_seg = np.zeros(n, dtype=np.int64); o_t = np.zeros(n); o_d = np.zeros(n)
    o_v = np.zeros(n); o_mask = np.zeros(n, dtype=np.int64)
    if n:
        _cpa_fill(grid.enu, pts, tms, ptr, sbox, config.rmax_m ** 2,
                  d_lo, d_hi, v_lo, v_hi, config.c, cell_ptr,
                  o_seg, o_t, o_d, o_v, o_mask)
    return CpaTable(cell_ptr=cell_ptr, seg=o_seg, t_cpa=o_t,
                    t_arr=o_t + o_d / config.c, d=o_d, v=o_v, mask=o_mask,
                    n_cells=len(grid.enu), n_events=len(perception.events))


# ---------------------------------------------------------------------------
# F5: pair gate
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    cell: int
    event_idx: tuple  # detected-event indices, in time order
    seg_idx: tuple  # segment indices, aligned with event_idx
    cpa_idx: tuple  # indices into the CpaTable arrays
    t0: float  # implied Unix recording start (or t0_known)
    dt_err: float  # max |implied T0_k - t0| over matched events, seconds


def _candidates_for_cell(cpa, g, event_idx):
    """CPA indices in cell g matching event event_idx (mask bit set),
    sorted by CPA distance ascending (closest first).

    Fix 3: the old code returned them in CPA-table (segment) order and
    then capped at 12, so the "closest 12" were not the closest and
    the true flight could be dropped.  Sorting here makes the cap
    a closest-first cap.
    """
    a, b = cpa.entries_for_cell(g)
    out = [i for i in range(a, b) if cpa.mask[i] & (1 << event_idx)]
    out.sort(key=lambda i: float(cpa.d[i]))
    return out


def pair_gate(cpa, perception, config, t0_known=None, counts=None):
    """F5: hypotheses -- one cell, one flight per event, one shared T0.

    With known T0, each event is checked against the absolute time.
    Without it, the implied T0 values must agree within ``dt_tol_s``.
    Returns hypotheses for every event subset of size >= 2 (and size 1
    when ``t0_known`` is given), not just pairs + the full set.

    Fix 4: with 4 detected events (3 real + 1 stray, as in G3), the
    true triple was never tried -- only pairs and the 4-set -- so the
    combined three-flight answer was hidden.  All subsets are now
    enumerated, up to ``config.max_subset_events`` events; above that
    the event list is pruned by span (longest first) and the pruning
    is recorded in ``counts`` (if supplied).

    Fix 3: candidate lists are sorted by distance before the cap
    (see _candidates_for_cell), and a combination budget overrun is
    NOT silently skipped: combos are evaluated in product order
    (closest-first, since the lists are distance-sorted) up to
    ``config.max_pair_combos`` per (cell, subset), and the truncation
    is recorded in ``counts`` (``pair_gate_truncated_groups``,
    ``pair_gate_combos_evaluated``).
    """
    events = perception.events
    n_ev = len(events)
    hyps = []
    if counts is None:
        counts = {}
    counts.setdefault("pair_gate_combos_evaluated", 0)
    counts.setdefault("pair_gate_truncated_groups", 0)
    counts.setdefault("pair_gate_events_pruned", 0)
    if n_ev == 0:
        return hyps
    # --- Event pruning for very many events (Fix 4) ---
    ev_idx_all = list(range(n_ev))
    if n_ev > config.max_subset_events:
        # Keep the longest-span events (most evidence); record it.
        order = sorted(ev_idx_all,
                       key=lambda k: -(events[k].span[1] - events[k].span[0]))
        kept = sorted(order[: config.max_subset_events])
        counts["pair_gate_events_pruned"] = int(n_ev - len(kept))
        import warnings
        warnings.warn(f"pair_gate: {n_ev} events, keeping "
                      f"{config.max_subset_events} longest-span events "
                      f"{kept}; the rest are not tried")
        ev_idx_all = kept
        events = [perception.events[k] for k in kept]
        n_ev = len(ev_idx_all)
    # Event subsets to try: ALL subsets of size >= 2, plus size 1 when
    # T0 is known (a single event is then a testable hypothesis).
    subsets = []
    for r in range(2, n_ev + 1):
        subsets.extend(tuple(c) for c in combinations(range(n_ev), r))
    if t0_known is not None:
        subsets.extend((k,) for k in range(n_ev))
    # Map subset indices (into the pruned event list) back to original
    # event indices for the Hypothesis objects.
    for g in range(cpa.n_cells):
        a, b = cpa.entries_for_cell(g)
        if b - a < 1:
            continue
        for subset in subsets:
            cand_lists = [_candidates_for_cell(cpa, g, ev_idx_all[k])
                          for k in subset]
            if any(not c for c in cand_lists):
                continue
            # Cap the Cartesian product: closest CPAs per event first.
            cand_lists = [c[: config.max_candidates_per_event]
                          for c in cand_lists]
            n_combo = 1
            for c in cand_lists:
                n_combo *= len(c)
            budget = config.max_pair_combos
            truncated = n_combo > budget
            evaluated = 0
            for combo in product(*cand_lists):
                if evaluated >= budget:
                    break
                evaluated += 1
                seg_idx = tuple(int(cpa.seg[i]) for i in combo)
                # Check 3 audit: seg_idx are INDICES into the
                # segments list, not (icao, seg).  Two pieces of
                # one aircraft (same icao/seg, different subseg /
                # piece, from trim_segments) are two different list
                # entries, so this dedup does NOT merge them: a
                # circling / out-and-back aircraft can genuinely
                # be heard twice, one piece per event.  The same
                # holds for collapse_blobs (groups by seg_idx),
                # the silence exclusion (exclude_seg_idx) and
                # score_hypothesis -- all per-piece.  The only
                # thing that was missing was the piece identity
                # in the REPORT (now "piece" in every flight
                # dict), which made two pieces look identical.
                if len(set(seg_idx)) != len(seg_idx):
                    continue  # same flight piece cannot be two different events
                t0s = []
                for j, k in enumerate(subset):
                    t0s.append(float(cpa.t_arr[combo[j]]
                                     - events[k].t_c))
                if t0_known is not None:
                    dt_err = max(abs(t - t0_known) for t in t0s)
                    if dt_err > config.dt_tol_s:
                        continue
                    t0 = float(t0_known)
                else:
                    if len(subset) < 2:
                        continue
                    t0 = float(np.mean(t0s))
                    dt_err = max(abs(t - t0) for t in t0s)
                    if dt_err > config.dt_tol_s:
                        continue
                orig_subset = tuple(ev_idx_all[k] for k in subset)
                hyps.append(Hypothesis(cell=int(g), event_idx=orig_subset,
                                       seg_idx=seg_idx,
                                       cpa_idx=tuple(int(i) for i in combo),
                                       t0=t0, dt_err=float(dt_err)))
            counts["pair_gate_combos_evaluated"] += int(evaluated)
            if truncated:
                counts["pair_gate_truncated_groups"] += 1
    # De-duplicate: same cell, same events, same flights -- keep smallest dt_err.
    best = {}
    for h in hyps:
        key = (h.cell, h.event_idx, h.seg_idx)
        if key not in best or h.dt_err < best[key].dt_err:
            best[key] = h
    return sorted(best.values(), key=lambda h: h.dt_err)


def single_event_gate(cpa, perception, config, t0_known):
    """F5 for the 1-event + known-T0 case (strategy, single-event variant)."""
    if len(perception.events) != 1 or t0_known is None:
        return []
    # Reuse pair_gate logic by pretending the single event is a pair with
    # itself is wrong; instead do the absolute check directly.
    out = []
    ev = perception.events[0]
    for g in range(cpa.n_cells):
        for i in _candidates_for_cell(cpa, g, 0):
            dt_err = abs(float(cpa.t_arr[i] - t0_known - ev.t_c))
            if dt_err <= config.dt_tol_s:
                out.append(Hypothesis(cell=int(g), event_idx=(0,),
                                      seg_idx=(int(cpa.seg[i]),),
                                      cpa_idx=(int(i),), t0=float(t0_known),
                                      dt_err=float(dt_err)))
    return out


# ---------------------------------------------------------------------------
# F6: soft silence (audibility)
# ---------------------------------------------------------------------------

def audibility_prob(d_m, cls, config):
    """P(audible | slant distance, class), from the synth model.

    ``render_scene`` amplitude at distance d (level = amplitude at 1 km):
    ``amp = level * 1000/d * 10**(-excess*d/1000/20)``.  SNR in dB against
    ``noise_rms``, then a logistic with scale ``audibility_scale_db``.
    At 2 km (prop, level 1): amp~0.20, SNR~20 dB, P~0.999.  At 6 km:
    amp~0.011, SNR~-5.6 dB, P~0.13.  At 10 km: P~0.001.  This is the
    G3 starting model; T3 replaces it with measured values.
    """
    d = np.asarray(d_m, dtype=float)
    level = config.level_by_class.get(cls, 0.5) if isinstance(cls, str) else 0.5
    d_safe = np.maximum(d, 1.0)
    amp = level * 1000.0 / d_safe * 10 ** (-config.excess_db_per_km * d_safe / 1000.0 / 20.0)
    snr_db = 20.0 * np.log10(np.maximum(amp, 1e-12) / max(config.noise_rms, 1e-12))
    p = 1.0 / (1.0 + np.exp(-snr_db / config.audibility_scale_db))
    return float(np.clip(p, 1e-6, 0.999)) if np.ndim(d) == 0 else np.clip(p, 1e-6, 0.999)


def silence_loglik(cpa, cell, segments, perception, t0, config,
                   exclude_seg_idx=()):
    """F6: sum over silence windows of log(1 - P(audible)).

    For each silence window and each segment (excluding the matched
    flights), take the closest CPA whose sound arrives inside the
    window (audio time = t_arr - T0) and penalise by its audibility.
    Flights far away contribute ~log(1) = 0; a loud flight in a
    silence costs several log units.  No hard cutoff anywhere.
    """
    if not perception.silences:
        return 0.0
    a, b = cpa.entries_for_cell(cell)
    if b - a == 0:
        return 0.0
    total = 0.0
    exclude = set(int(s) for s in exclude_seg_idx)
    for (s0, s1) in perception.silences:
        # Closest CPA per segment inside this window.
        best_d = {}
        for i in range(a, b):
            si = int(cpa.seg[i])
            if si in exclude:
                continue
            t_audio = float(cpa.t_arr[i] - t0)
            if s0 <= t_audio <= s1:
                if si not in best_d or float(cpa.d[i]) < best_d[si]:
                    best_d[si] = float(cpa.d[i])
        for si, d in best_d.items():
            p = audibility_prob(d, segments[si].cls, config)
            total += math.log1p(-p)
    return float(total)


def _silence_for_hypothesis(sil_cache, cpa, cell, segments, perception,
                            t0, config, exclude_seg_idx=()):
    """Silence loglik for one hypothesis, with a correctly-keyed cache.

    Fix 1: the old run_search cached by cell only, but
    silence_loglik depends on (a) t0 -- different hypotheses at the
    same cell have different implied T0, hence different audio times
    for the same CPA -- and (b) the excluded (matched) flights.
    A second hypothesis at the same cell therefore got the first
    one's silence score.  The cache key is now
    (cell, round(t0, 3), tuple(sorted(exclude_seg_idx))).
    ``sil_cache`` may be None (no caching, e.g. in tests).
    """
    key = (int(cell), round(float(t0), 3),
           tuple(sorted(int(s) for s in exclude_seg_idx)))
    if sil_cache is not None and key in sil_cache:
        return sil_cache[key]
    val = silence_loglik(cpa, cell, segments, perception, t0, config,
                         exclude_seg_idx=exclude_seg_idx)
    if sil_cache is not None:
        sil_cache[key] = val
    return val


# ---------------------------------------------------------------------------
# F7: collapse to flight-pair blobs
# ---------------------------------------------------------------------------

@dataclass
class Blob:
    event_idx: tuple
    seg_idx: tuple
    cells: list  # grid-cell indices
    hypotheses: list  # Hypothesis objects, one per cell
    centroid_cell: int


def collapse_blobs(hypotheses):
    """F7: group surviving cells by (events, flights)."""
    groups = defaultdict(list)
    for h in hypotheses:
        groups[(h.event_idx, h.seg_idx)].append(h)
    blobs = []
    for (ev_idx, seg_idx), hs in groups.items():
        cells = sorted({h.cell for h in hs})
        # Centroid cell: the hypothesis cell closest to the median cell ENU
        # is approximated here by the median hypothesis (smallest dt_err
        # among the median-ranked cells); the caller re-centres on the
        # scored centroid.  Use the lowest-dt_err cell as centroid.
        centroid = min(hs, key=lambda h: h.dt_err).cell
        blobs.append(Blob(event_idx=ev_idx, seg_idx=seg_idx, cells=cells,
                          hypotheses=hs, centroid_cell=int(centroid)))
    blobs.sort(key=lambda bl: min(h.dt_err for h in bl.hypotheses))
    return blobs


def effective_max_blobs_to_score(n_blobs, config):
    """Density-scaled scoring budget (2026-11-09, G3b New York).

    ``min(max_blobs_to_score_cap, max(max_blobs_to_score,
    ceil(blobs_to_score_frac * n_blobs)))``: the floor is today's
    fixed 20, so sparse scenes (G3a: 6 blobs) are unchanged; dense
    scenes score up to all blobs, bounded by the cap.
    """
    return int(min(config.max_blobs_to_score_cap,
                   max(config.max_blobs_to_score,
                       math.ceil(config.blobs_to_score_frac * n_blobs))))


# ---------------------------------------------------------------------------
# F8: forward scoring on the spectrogram
# ---------------------------------------------------------------------------

@dataclass
class ScoredHypothesis:
    hypothesis: Hypothesis
    observer_enu: np.ndarray
    best_dt: float
    best_f0: dict  # event index -> f0 (Hz) at the joint maximum
    mean_score: dict  # event index -> mean spectrogram score (dB) at max
    loglik_max: float
    loglik_marg: float
    per_event_max: dict
    per_event_marg: dict
    silence_loglik: float

    @property
    def loglik(self):
        return self.loglik_max


def _ratio_curve(segment, observer_enu, config):
    """(t_arr, ratio) for one segment/observer, from physics (M4)."""
    t_emit = np.arange(segment.track.t_start, segment.track.t_end,
                       config.dt_emit_s)
    if len(t_emit) < 2:
        return None
    t_arr, ratio, _dist = emission_to_arrival(
        segment.track, np.asarray(observer_enu, dtype=float), t_emit, c=config.c)
    order = np.argsort(t_arr)
    return t_arr[order], ratio[order]


def _score_event_matrix(segment, observer_enu, spec, event, t0, config):
    """Mean spectrogram score (dB) on a (dt, f0) grid for one event.

    Returns (dt_grid, f0_grid, matrix, n_eff) or None if the event span
    has too few spectrogram frames.  f0 is swept per event, dt is common
    (locked offset): the caller combines events at the same dt.
    Harmonics: ``event.harmonics`` (multiples of the lowest detected
    tone) with weight 1/k -- close to the synth amplitudes
    (prop 1, 0.6, 0.4, 0.25) and down-weighting the noisier high
    harmonics.  The base frequency is the lowest *detected* tone, which
    need not be the true blade-pass fundamental: the Doppler ratio is
    the same for every harmonic, so the curve shape is unchanged.
    """
    frames_all = spec.t[(spec.t >= event.span[0]) & (spec.t <= event.span[1])]
    if len(frames_all) < 8:
        # Fall back to frames around t_c if the span is unreliable.
        frames_all = spec.t[np.abs(spec.t - event.t_c) <= 60.0]
    if len(frames_all) < 8:
        return None
    frames = frames_all[:: max(1, config.score_frame_step)]
    curve = _ratio_curve(segment, observer_enu, config)
    if curve is None:
        return None
    t_arr, ratio = curve
    dt_grid = np.arange(-config.dt_sweep_s, config.dt_sweep_s + 1e-9,
                        config.dt_step_s)
    f0_grid = event.f0 + np.arange(-config.f0_sweep_hz,
                                   config.f0_sweep_hz + 1e-9, config.f0_step_hz)
    harmonics = list(event.harmonics) if event.harmonics else [1]
    weights = np.array([1.0 / max(1, int(k)) for k in harmonics], dtype=float)
    weights = weights / weights.sum()
    mat = np.full((len(dt_grid), len(f0_grid)), -5.0)
    t_target_base = frames + t0  # Unix arrival times before the dt shift
    for di, dt in enumerate(dt_grid):
        r = np.interp(t_target_base - dt, t_arr, ratio, left=np.nan, right=np.nan)
        valid = np.isfinite(r)
        if valid.sum() < 0.5 * len(frames):
            continue
        # Score per harmonic, weighted.
        total = np.zeros(len(f0_grid))
        for w, k in zip(weights, harmonics):
            f = (f0_grid[:, None] * float(k)) * r[None, :]
            # spec.value expects t and f of the same shape.
            tt = np.broadcast_to(frames[None, :], f.shape)
            vals = spec.value(tt, f)
            with np.errstate(all="ignore"):
                m = np.nanmean(np.where(np.isfinite(vals), vals, np.nan), axis=1)
            total += w * np.where(np.isfinite(m), m, -5.0)
        mat[di, :] = total
    span_s = max(float(event.span[1] - event.span[0]), 1.0)
    n_eff = min(config.n_eff_cap, max(5.0, span_s / 1.02))  # capped, see SearchConfig
    return dt_grid, f0_grid, mat, float(n_eff)


def _loglik_from_matrix(mat, n_eff, config):
    """Convert a mean-score (dB) matrix to log-likelihoods.

    Per sweep point: ``loglik = mean_score * N_eff / score_scale_db``.
    Max: the best sweep point.  Marginalised: logsumexp over the sweep
    grid minus log(N_points) (uniform prior over the sweep), so the two
    are on the same scale and marginalised <= max.
    """
    ll = mat * (n_eff / config.score_scale_db)
    best = float(np.max(ll))
    marg = float(logsumexp(ll.ravel()) - math.log(ll.size))
    return best, marg


def score_hypothesis(segments, observer_enu, hypothesis, perception, config,
                     silence_ll=None, cpa=None):
    """F8 scoring for one hypothesis at one observer position.

    Returns a ScoredHypothesis, or None if any event cannot be scored.
    The joint maximum is over (common dt, per-event f0); the joint
    marginalised value is logsumexp over the same grid (see _loglik...).
    """
    spec = perception.spec
    mats = {}
    n_effs = {}
    for j, ev_idx in enumerate(hypothesis.event_idx):
        seg = segments[hypothesis.seg_idx[j]]
        ev = perception.events[ev_idx]
        res = _score_event_matrix(seg, observer_enu, spec, ev, hypothesis.t0, config)
        if res is None:
            return None
        dt_grid, f0_grid, mat, n_eff = res
        mats[ev_idx] = (dt_grid, f0_grid, mat)
        n_effs[ev_idx] = n_eff
    # Joint max: for each dt, sum over events of the best f0 at that dt.
    dt_grid = mats[hypothesis.event_idx[0]][0]
    f0_grids = {k: mats[k][1] for k in hypothesis.event_idx}
    per_dt_max = np.zeros(len(dt_grid))
    per_dt_marg_terms = np.zeros(len(dt_grid))
    best_f0 = {}
    for di in range(len(dt_grid)):
        s = 0.0
        for k in hypothesis.event_idx:
            _dg, fg, mat = mats[k]
            ll = mat[di, :] * (n_effs[k] / config.score_scale_db)
            s += float(np.max(ll))
            per_dt_marg_terms[di] += float(logsumexp(ll) - math.log(len(fg)))
        per_dt_max[di] = s
    di_best = int(np.argmax(per_dt_max))
    loglik_max = float(per_dt_max[di_best])
    # Joint marginalised: logsumexp over dt of (sum_k logsumexp_f0) ,
    # minus log(N_dt) for the uniform dt prior (f0 normalisation is
    # already in per_dt_marg_terms).
    loglik_marg = float(logsumexp(per_dt_marg_terms) - math.log(len(dt_grid)))
    mean_score = {}
    for k in hypothesis.event_idx:
        _dg, fg, mat = mats[k]
        ll = mat[di_best, :] * (n_effs[k] / config.score_scale_db)
        fi = int(np.argmax(ll))
        best_f0[k] = float(fg[fi])
        mean_score[k] = float(mat[di_best, fi])
    per_event_max = {}
    per_event_marg = {}
    for k in hypothesis.event_idx:
        _dg, _fg, mat = mats[k]
        b, m = _loglik_from_matrix(mat, n_effs[k], config)
        per_event_max[k] = b
        per_event_marg[k] = m
    if silence_ll is None:
        silence_ll = 0.0
    return ScoredHypothesis(
        hypothesis=hypothesis, observer_enu=np.asarray(observer_enu, dtype=float),
        best_dt=float(dt_grid[di_best]), best_f0=best_f0, mean_score=mean_score,
        loglik_max=loglik_max, loglik_marg=loglik_marg,
        per_event_max=per_event_max, per_event_marg=per_event_marg,
        silence_loglik=float(silence_ll))


def _score_single_point(segments, observer_enu, hypothesis, perception, config):
    """Mean score / loglik at the locked (best_dt, best_f0) from F8.

    Used in F9: per-cell scoring with one sweep point, not a full sweep.
    Returns (loglik_max_locked, mean_score_dict) or None.
    """
    # Reuse the matrix machinery with a degenerate sweep by temporarily
    # narrowing the config -- simpler: compute the score directly.
    spec = perception.spec
    total_ll = 0.0
    mean_score = {}
    # The locked parameters are carried on the hypothesis object by the
    # caller via attributes set in run_search (see _attach_locked).
    dt = getattr(hypothesis, "_locked_dt", 0.0)
    f0s = getattr(hypothesis, "_locked_f0", {})
    for j, ev_idx in enumerate(hypothesis.event_idx):
        seg = segments[hypothesis.seg_idx[j]]
        ev = perception.events[ev_idx]
        f0 = float(f0s.get(ev_idx, ev.f0))
        curve = _ratio_curve(seg, observer_enu, config)
        if curve is None:
            return None
        t_arr, ratio = curve
        frames_all = spec.t[(spec.t >= ev.span[0]) & (spec.t <= ev.span[1])]
        if len(frames_all) < 8:
            frames_all = spec.t[np.abs(spec.t - ev.t_c) <= 60.0]
        if len(frames_all) < 8:
            return None
        frames = frames_all[:: max(1, config.score_frame_step)]
        r = np.interp(frames + hypothesis.t0 - dt, t_arr, ratio,
                      left=np.nan, right=np.nan)
        valid = np.isfinite(r)
        if valid.sum() < 0.5 * len(frames):
            return None
        harmonics = list(ev.harmonics) if ev.harmonics else [1]
        weights = np.array([1.0 / max(1, int(k)) for k in harmonics], dtype=float)
        weights = weights / weights.sum()
        score = 0.0
        for w, k in zip(weights, harmonics):
            f = (f0 * float(k)) * r
            vals = spec.value(frames, f)
            m = float(np.nanmean(np.where(np.isfinite(vals), vals, np.nan)))
            score += w * (m if np.isfinite(m) else -5.0)
        mean_score[ev_idx] = float(score)
        span_s = max(float(ev.span[1] - ev.span[0]), 1.0)
        n_eff = min(config.n_eff_cap, max(5.0, span_s / 1.02))
        total_ll += score * n_eff / config.score_scale_db
    return float(total_ll), mean_score


def _attach_locked(hypothesis, scored):
    hypothesis._locked_dt = scored.best_dt
    hypothesis._locked_f0 = dict(scored.best_f0)
    return hypothesis


# ---------------------------------------------------------------------------
# F9: refinement
# ---------------------------------------------------------------------------

@dataclass
class RefinedResult:
    blob_key: tuple
    best_lat: float
    best_lon: float
    best_enu: np.ndarray
    best_loglik: float
    best_dt: float
    best_f0: dict
    grids: list  # list of (spacing, Grid, loglik array) per level
    best_t0: float = float("nan")  # Fix 2: implied/known T0 at the refined peak


def _box_around_latlon(lat, lon, half_m):
    dlat = half_m / EARTH_M_PER_DEG_LAT
    dlon = half_m / (EARTH_M_PER_DEG_LAT * max(0.2, math.cos(math.radians(lat))))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def refine_blob(segments, blob, perception, config, frame, height_fn,
                scored_centroid, t0_known=None, silence_fn=None):
    """F9: 100 -> 25 -> 5 m refinement around one blob.

    Each level builds a fresh CPA table on a small window, keeps only
    cells matching this blob's flight pair, scores them with the locked
    (dt, f0) from the previous level, and recentres on the best cell.
    A small re-sweep at the final peak updates (dt, f0).  The window
    half-widths (500/150/50 m) are >> the coarse-grid error, so the
    true peak stays inside the window if the coarse peak was right.

    Fix 2: ``t0_known`` is threaded through -- with a known recording
    start the gate uses it (pair_gate with t0_known, or
    single_event_gate for the 1-event case) instead of re-deriving
    T0 from pairs, which could drift / pick a wrong pair.  The
    refined T0 therefore stays at t0_known.  A level with no
    hypotheses, or no scorable cell, now RAISES RuntimeError instead
    of silently breaking -- the caller (run_search) records the
    failure in funnel_counts/report; a silently coarse peak is worse
    than a recorded failure.  The re-sweep silence term is also fixed:
    it previously used scored.silence_loglik, which is 0.0 when no
    silence_ll was passed, instead of the silence at the refined cell.

    ``silence_fn`` (jet deferral): if given, it is called as
    silence_fn(cpa, cell, cell_enu, t0, exclude_seg_idx) and must
    return the TOTAL silence loglik for that cell, including any
    deferred-jet contribution; if None, plain silence_loglik is
    used (early segments only -- the pre-deferral behaviour).
    """
    def _sil(cpa, cell, t0, exclude_seg_idx):
        if silence_fn is not None:
            return float(silence_fn(cpa, cell, grid.enu[cell], t0, exclude_seg_idx))
        return float(silence_loglik(cpa, cell, segments, perception, t0,
                                    config, exclude_seg_idx=exclude_seg_idx))
    best_enu = np.asarray(scored_centroid.observer_enu, dtype=float)
    best_dt = scored_centroid.best_dt
    best_f0 = dict(scored_centroid.best_f0)
    best_ll = scored_centroid.loglik_max + scored_centroid.silence_loglik
    best_t0 = float(t0_known) if t0_known is not None else float(
        scored_centroid.hypothesis.t0)
    grids_out = []
    for spacing, half in zip(config.refine_spacings_m, config.refine_window_half_m):
        lat_c, lon_c, _h = frame.to_llh(best_enu[None, :])
        lat_c, lon_c = float(lat_c[0]), float(lon_c[0])
        la_min, la_max, lo_min, lo_max = _box_around_latlon(lat_c, lon_c, half)
        grid = make_grid(la_min, la_max, lo_min, lo_max, spacing, frame, height_fn)
        cpa = build_cpa_table(grid, segments, perception, config)
        if len(perception.events) == 1 and t0_known is not None:
            gate_hyps = single_event_gate(cpa, perception, config, t0_known)
        else:
            gate_hyps = pair_gate(cpa, perception, config, t0_known=t0_known)
        hyps = [h for h in gate_hyps
                if h.event_idx == blob.event_idx and h.seg_idx == blob.seg_idx]
        if not hyps:
            raise RuntimeError(
                f"refine_blob {blob.event_idx}/{blob.seg_idx}: no hypotheses "
                f"at spacing {spacing} m (window half {half} m)")
        # Score each surviving cell at the locked parameters.
        logliks = np.full(len(grid.enu), -np.inf)
        best_cell_ll = -np.inf
        best_cell = None
        best_hyp = None
        for h in hyps:
            _attach_locked(h, scored_centroid)
            # Locked dt/f0 may need updating per level; use current best.
            h._locked_dt = best_dt
            h._locked_f0 = dict(best_f0)
            res = _score_single_point(segments, grid.enu[h.cell], h, perception, config)
            if res is None:
                continue
            ll_locked, _ms = res
            sil = _sil(cpa, h.cell, h.t0, h.seg_idx)
            total = ll_locked + sil
            # Keep the best hypothesis per cell (several T0 variants can occur).
            if total > logliks[h.cell]:
                logliks[h.cell] = total
            if total > best_cell_ll:
                best_cell_ll = total
                best_cell = h.cell
                best_hyp = h
        grids_out.append((float(spacing), grid, logliks))
        if best_cell is None:
            raise RuntimeError(
                f"refine_blob {blob.event_idx}/{blob.seg_idx}: no scorable "
                f"cell at spacing {spacing} m")
        best_enu = np.asarray(grid.enu[best_cell], dtype=float)
        best_ll = float(best_cell_ll)
        best_t0 = float(best_hyp.t0)
        # Small re-sweep at the new peak, to update (dt, f0) for the next level.
        if best_hyp is not None:
            sil_best = _sil(cpa, best_cell, best_hyp.t0, best_hyp.seg_idx)
            scored = score_hypothesis(segments, best_enu, best_hyp, perception,
                                      config, silence_ll=sil_best)
            if scored is not None:
                best_dt = scored.best_dt
                best_f0 = dict(scored.best_f0)
                best_ll = scored.loglik_max + scored.silence_loglik
                best_t0 = float(best_hyp.t0)
    lat_f, lon_f, _h = frame.to_llh(best_enu[None, :])
    result = RefinedResult(blob_key=(blob.event_idx, blob.seg_idx),
                           best_lat=float(lat_f[0]), best_lon=float(lon_f[0]),
                           best_enu=best_enu, best_loglik=float(best_ll),
                           best_dt=float(best_dt), best_f0=dict(best_f0),
                           grids=grids_out)
    # Keep the (known or implied) T0 on the result for the caller/tests.
    result.best_t0 = float(best_t0)
    return result


# ---------------------------------------------------------------------------
# M6: posterior, modes, outputs
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    grid: Grid
    logpost_max: np.ndarray  # (N,) log posterior, max-over-sweep scoring
    logpost_marg: np.ndarray  # (N,) log posterior, marginalised scoring
    prob_max: np.ndarray
    prob_marg: np.ndarray
    modes: list  # dicts: lat, lon, mass (marg prob), loglik_max/marg, flights
    hypotheses: list
    blobs: list
    scored: list  # ScoredHypothesis objects (blob centroids)
    refined: list  # RefinedResult objects
    funnel_counts: dict
    config: SearchConfig
    frame: LocalFrame
    box: tuple
    t0_known: object
    report: dict

    def peak(self, which="marg"):
        arr = self.prob_marg if which == "marg" else self.prob_max
        i = int(np.argmax(arr))
        return float(self.grid.lats[i]), float(self.grid.lons[i]), float(arr[i])

    def truth_in_region(self, lat, lon, level=0.90, which="marg"):
        """Is (lat, lon) inside the level credible region? (nearest cell).

        Fix 6c: nearest cell is found in ENU metres, not by
        (lat diff)^2 + (lon diff)^2 in degrees.
        """
        regions = credible_masks(self, which=which)
        truth_enu = self.frame.to_enu(np.array([lat]), np.array([lon]),
                                      np.array([0.0]))[0]
        d2 = ((self.grid.enu[:, :2] - truth_enu[:2]) ** 2).sum(axis=1)
        i = int(np.argmin(d2))
        return bool(regions[level][i])


def _normalise_logpost(logpost):
    logpost = np.asarray(logpost, dtype=float)
    finite = np.isfinite(logpost)
    if not finite.any():
        p = np.full_like(logpost, 1.0 / len(logpost))
        return np.zeros_like(logpost), p
    m = float(np.max(logpost[finite]))
    p = np.exp(np.where(finite, logpost - m, -np.inf))
    p = p / p.sum()
    logp = np.where(p > 0, np.log(np.maximum(p, 1e-300)), -np.inf)
    return logp, p


def _smooth_logpost(logpost, shape, sigma_cells):
    """Gaussian-smooth a log-posterior grid by ``sigma_cells``.

    Done in probability space (exp, convolve, renormalise) so the
    result is still a proper posterior.  ``sigma_cells=0`` is a no-op.
    This is the systematic-error allowance discussed in SearchConfig:
    without it a 500 m grid can put 98% of the mass on one cell whose
    centre is ~350 m from the truth, and the 90% region then misses
    the truth even though the peak is the correct cell.
    """
    if not sigma_cells or sigma_cells <= 0:
        return logpost
    from scipy.ndimage import gaussian_filter
    ny, nx = shape
    finite = np.isfinite(logpost)
    if not finite.any():
        return logpost
    m = float(np.max(logpost[finite]))
    p = np.exp(np.where(finite, logpost - m, -np.inf)).reshape(ny, nx)
    smoothed = gaussian_filter(p, sigma=float(sigma_cells), mode="nearest")
    total = float(smoothed.sum())
    if total <= 0:
        return logpost
    smoothed = smoothed / total
    out = np.where(smoothed > 0, np.log(np.maximum(smoothed, 1e-300)), -np.inf)
    return out.ravel()


def credible_masks(result, which="marg"):
    """Binary masks for the 50/90/95% highest-posterior-density regions."""
    prob = result.prob_marg if which == "marg" else result.prob_max
    order = np.argsort(-prob)
    cum = np.cumsum(prob[order])
    masks = {}
    for level in (0.50, 0.90, 0.95):
        thr_idx = int(np.searchsorted(cum, level))
        keep = order[: thr_idx + 1]
        mask = np.zeros(len(prob), dtype=bool)
        mask[keep] = True
        masks[level] = mask
    return masks


def _find_modes(grid, prob, logpost, config, segments, blobs):
    """Local maxima of the posterior, clustered within 1 km.

    Fix 6c: clustering is in ENU metres, not degrees -- the old code
    compared (lat diff)^2 + (lon diff)^2 in degrees, which is wrong
    units and anisotropic (1 deg lon != 1 deg lat).

    Fix 5c: ``mass`` is the whole peak's mass, not one cell's
    probability.  Each grid cell is assigned to its nearest mode
    (in ENU metres, Voronoi assignment), and the mode's mass is the
    sum of the posterior probability over its assigned cells.  The
    masses therefore sum to ~1 over the modes (plus the unassigned
    tail is negligible: every cell is assigned to some mode).
    """
    ny, nx = grid.shape
    p2 = prob.reshape(ny, nx)
    cand = []
    for iy in range(ny):
        for ix in range(nx):
            v = p2[iy, ix]
            if v <= 0:
                continue
            lo_y, hi_y = max(0, iy - 1), min(ny, iy + 2)
            lo_x, hi_x = max(0, ix - 1), min(nx, ix + 2)
            if v >= p2[lo_y:hi_y, lo_x:hi_x].max():
                cand.append((float(v), iy * nx + ix))
    cand.sort(reverse=True)
    modes = []
    sep_m = float(getattr(config, "mode_min_separation_m", 1000.0))
    for v, idx in cand:
        if any(float(np.hypot(*(grid.enu[idx, :2] - grid.enu[m["cell"], :2])))
               < sep_m for m in modes):
            continue
        modes.append({"lat": float(grid.lats[idx]), "lon": float(grid.lons[idx]),
                      "mass": float(v), "cell": int(idx)})
        if len(modes) >= config.max_modes:
            break
    # Voronoi mass: assign every cell to its nearest mode in ENU metres.
    if modes:
        mode_enu = np.array([grid.enu[m["cell"], :2] for m in modes])
        # Chunked to bound memory on large grids.
        mass = np.zeros(len(modes))
        for a in range(0, len(grid.enu), 10000):
            chunk = grid.enu[a:a + 10000, :2]
            d2 = ((chunk[:, None, :] - mode_enu[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argmin(d2, axis=1)
            np.add.at(mass, nearest, prob[a:a + 10000])
        for m, mm in zip(modes, mass):
            m["mass"] = float(mm)
    return modes


def _projected_crs_and_transform(result):
    """Projected CRS + Affine transform in which the ENU grid is regular.

    Fix 6c: the old code wrote an EPSG:4326 GeoTIFF with a single
    mean dlat/dlon Affine, but dlon varies with cos(lat) and the ENU
    tangent plane distorts over a large box, so the heat map /
    credible polygons were misplaced.  The grid IS regular in ENU
    metres about the frame origin, so write it in azimuthal
    equidistant centred on the frame origin (``+proj=aeqd``) -- the
    natural projected CRS for a local ENU grid -- with the transform
    in metres.  Returns (crs, transform, proj4_string).
    """
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    frame = result.frame
    lat0 = float(getattr(frame, "lat0", float(np.mean(result.grid.lats))))
    lon0 = float(getattr(frame, "lon0", float(np.mean(result.grid.lons))))
    proj4 = (f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
             "+datum=WGS84 +units=m +no_defs")
    crs = CRS.from_proj4(proj4)
    spacing = float(result.grid.spacing_m)
    west = float(np.min(result.grid.enu[:, 0]) - spacing / 2.0)
    north = float(np.max(result.grid.enu[:, 1]) + spacing / 2.0)
    transform = Affine(spacing, 0.0, west, 0.0, -spacing, north)
    return crs, transform, proj4


def _grid_distortion(grid):
    """Distortion diagnostics for the uniform-dlon assumption.

    Returns {"dlon_rel_variation": ..., "box_width_m": ..., "box_height_m": ...}.
    dlon varies with cos(lat); over a 50 km box the variation is
    ~0.1%, over CONUS it is tens of percent -- which is why the old
    mean-dlon EPSG:4326 transform was wrong at scale (Fix 6c).
    """
    ny, nx = grid.shape
    lons = grid.lons.reshape(ny, nx)
    if nx > 1:
        dlon_row = np.mean(np.diff(lons, axis=1), axis=1)
        mean_dlon = float(np.mean(dlon_row)) if len(dlon_row) else 0.0
        var = (float(np.max(dlon_row) - np.min(dlon_row)) / abs(mean_dlon)
               if mean_dlon else 0.0)
    else:
        var = 0.0
    width_m = float((grid.lon_max - grid.lon_min) * EARTH_M_PER_DEG_LAT
                    * math.cos(math.radians(0.5 * (grid.lat_min + grid.lat_max))))
    height_m = float((grid.lat_max - grid.lat_min) * EARTH_M_PER_DEG_LAT)
    return {"dlon_rel_variation": float(var),
            "box_width_m": width_m, "box_height_m": height_m}


def write_outputs(result, out_dir):
    """M6 outputs (FRAMEWORK section 4): GeoTIFF, GeoJSON, JSON, PNG.

    Returns a dict of paths.  rasterio / shapely / matplotlib are
    imported here so that importing search.py does not require them.

    Fix 6c: ``heatmap.tif`` is now written in azimuthal-equidistant
    (+proj=aeqd, centred on the search frame origin), in which the
    ENU grid is regular in metres -- NOT EPSG:4326 with a mean
    dlat/dlon transform.  The CRS is stored in the report
    (``report["output_crs"]``).  ``credible_regions.geojson`` is
    still EPSG:4326 (GeoJSON convention): its polygons are built in
    the projected CRS and reprojected to lat/lon.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    grid = result.grid
    ny, nx = grid.shape
    distortion = _grid_distortion(grid)
    result.report["grid_distortion"] = distortion
    if distortion["dlon_rel_variation"] > 0.01 or distortion["box_width_m"] > 200_000:
        result.report.setdefault("warnings", []).append(
            f"large-box distortion: dlon variation "
            f"{distortion['dlon_rel_variation']:.3f}, box "
            f"{distortion['box_width_m'] / 1000:.0f}x"
            f"{distortion['box_height_m'] / 1000:.0f} km -- "
            "heat map is in the projected (aeqd) CRS; tile the search "
            "for very large boxes (see G4)")
    # --- GeoTIFF (marginalised posterior; band 1 = probability, band 2 = log posterior)
    try:
        import rasterio
        crs, transform, proj4 = _projected_crs_and_transform(result)
        result.report["output_crs"] = proj4
        result.report["output_crs_note"] = (
            "heatmap.tif is in azimuthal-equidistant (+proj=aeqd) centred "
            "on the search frame origin, in metres; the ENU grid is "
            "regular in this CRS.  credible_regions.geojson is EPSG:4326.")
        prob = result.prob_marg.reshape(ny, nx).astype(np.float32)
        logp = result.logpost_marg.reshape(ny, nx).astype(np.float32)
        logp = np.where(np.isfinite(logp), logp, -30.0).astype(np.float32)
        tif_path = out_dir / "heatmap.tif"
        with rasterio.open(tif_path, "w", driver="GTiff", height=ny, width=nx,
                           count=2, dtype="float32", crs=crs,
                           transform=transform) as dst:
            dst.write(prob, 1)
            dst.write(logp, 2)
            dst.set_band_description(1, "posterior_probability_marginalised")
            dst.set_band_description(2, "log_posterior_marginalised")
        paths["heatmap_tif"] = str(tif_path)
        paths["heatmap_crs"] = proj4
    except Exception as exc:  # pragma: no cover - reported in the JSON instead
        paths["heatmap_tif_error"] = str(exc)
    # --- GeoJSON credible regions (marginalised, EPSG:4326)
    try:
        import rasterio
        from rasterio.features import shapes
        from shapely.geometry import shape as shp_shape, mapping as shp_mapping
        from shapely.ops import unary_union
        try:
            from shapely import transform as shp_transform
        except ImportError:  # shapely < 2
            from shapely.ops import transform as shp_transform
        from pyproj import Transformer
        crs, transform, proj4 = _projected_crs_and_transform(result)
        # Reproject aeqd-metre polygons to lat/lon for GeoJSON.
        to_ll = Transformer.from_crs(proj4, "EPSG:4326", always_xy=True)
        masks = credible_masks(result, which="marg")
        features = []
        for level in (0.50, 0.90, 0.95):
            mask = masks[level].reshape(ny, nx).astype(np.uint8)
            geoms = []
            for geom, val in shapes(mask, mask=mask, transform=transform):
                if val == 1:
                    geoms.append(shp_shape(geom))
            if not geoms:
                continue
            merged = unary_union(geoms)
            # shapely.transform: transformation takes an (N, 2) coordinate
            # array and returns the transformed array.
            def _to_ll(coords):
                lon, lat = to_ll.transform(coords[:, 0], coords[:, 1])
                return np.column_stack([lon, lat])

            merged_ll = shp_transform(merged, _to_ll)
            features.append({"type": "Feature",
                             "properties": {"level": level,
                                            "cells": int(mask.sum())},
                             "geometry": shp_mapping(merged_ll)})
        geo_path = out_dir / "credible_regions.geojson"
        geo_path.write_text(json.dumps(
            {"type": "FeatureCollection", "features": features}, indent=2))
        paths["credible_geojson"] = str(geo_path)
    except Exception as exc:  # pragma: no cover
        paths["credible_geojson_error"] = str(exc)
    # --- JSON report
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(result.report, indent=2, default=str))
    paths["report_json"] = str(report_path)
    # --- Diagnostic plot
    try:
        png_path = write_diagnostic_plot(result, out_dir / "diagnostic.png")
        paths["diagnostic_png"] = str(png_path)
    except Exception as exc:  # pragma: no cover
        paths["diagnostic_png_error"] = str(exc)
    return paths


def write_diagnostic_plot(result, path):
    """Spectrogram with the best mode's modelled curves overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Rebuild the perception from the report's stored reference: the
    # SearchResult does not carry the Perception, so the caller (run_search)
    # attaches it as ``result._perception`` before calling write_outputs.
    perc = getattr(result, "_perception", None)
    if perc is None:
        raise ValueError("diagnostic plot needs result._perception")
    spec = perc.spec
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(spec.S, aspect="auto", origin="lower",
              extent=[spec.t[0], spec.t[-1], spec.f[0], spec.f[-1]],
              cmap="viridis", vmin=-5, vmax=20)
    ax.set_xlabel("time from recording start (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram with modelled curves (best mode, marginalised posterior)")
    # Best mode: use the first refined result if present, else the top mode.
    if result.refined:
        best = result.refined[0]
        obs = best.best_enu
        dt = best.best_dt
        f0s = best.best_f0
        # Find the scored hypothesis for this blob to get event/segment mapping.
        scored = None
        for s in result.scored:
            if (s.hypothesis.event_idx, s.hypothesis.seg_idx) == best.blob_key:
                scored = s
                break
        if scored is not None:
            from .physics import heard_curve
            # Segments are attached to the result for plotting.
            segs = getattr(result, "_segments", [])
            seg_by_idx = {i: s for i, s in enumerate(segs)}
            for j, ev_idx in enumerate(scored.hypothesis.event_idx):
                ev = perc.events[ev_idx]
                seg = seg_by_idx.get(scored.hypothesis.seg_idx[j])
                if seg is None:
                    continue
                t_frames = spec.t[(spec.t >= ev.span[0]) & (spec.t <= ev.span[1])]
                if len(t_frames) == 0:
                    continue
                f0 = f0s.get(ev_idx, ev.f0)
                # heard_curve expects Unix arrival times.
                t_unix = t_frames + scored.hypothesis.t0 - dt
                for k in (ev.harmonics or [1]):
                    f = heard_curve(seg.track, obs, t_unix, f0 * float(k),
                                    c=result.config.c)
                    ok = np.isfinite(f)
                    ax.plot(t_frames[ok], f[ok], color="red", lw=1.2,
                            label=f"event {ev_idx} x{k}" if k == 1 else None)
    fig.colorbar(ax.images[0], ax=ax, label="whitened level (dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def run_search(perception, segments, box, height_fn=None, t0_known=None,
               config=None, out_dir=None, frame=None,
               t_start=None, t_end=None, time_margin_s=None,
               margin_already_in_window=False):
    """Run M3-M6 for one recording.

    ``perception``: ``audio.Perception`` (events, silences, spectrogram).
    ``segments``: list of ``Segment`` in the search frame.
    ``box``: (lat_min, lat_max, lon_min, lon_max).
    ``height_fn``: callable(lat, lon) -> ground height (m, ellipsoid),
        a constant, or None (flat, 0 m).  Production: ``Terrain.height``.
    ``t0_known``: Unix recording start, or None.
    ``t_start`` / ``t_end`` / ``time_margin_s``: the time window the
        segments were loaded with (see load_segments).  They are NOT
        re-applied here -- they are RECORDED in the report
        (``report["time_filter"]``) because they are an assumption.
        ``margin_already_in_window`` (user note 5, 2026-11-08): False
        (default) means ``time_margin_s`` is ADDITIONAL uncertainty
        beyond ``t_start``/``t_end`` -- the loader's window is
        [t_start-margin, t_end+margin].  True means the caller
        already folded the T0 uncertainty into ``t_start``/``t_end``
        itself, so the margin is only extra padding, not additional
        start-time uncertainty -- the report says so instead of
        claiming the margin IS the uncertainty.  With no filter
        (all None) the report says "no time filter (whole day)"
        and records ``margin_s`` as None, never "±0 s".
        If only the date is known, pass all three as None:
        the loader then keeps the whole day and the size question
        comes back (the corridor guard is the only thing standing
        between a whole-day load and the 5M cap).
    Returns a ``SearchResult`` (and writes outputs if ``out_dir`` given).
    """
    import time as _time
    _t_run0 = _time.time()
    config = config or SearchConfig()
    lat_min, lat_max, lon_min, lon_max = [float(v) for v in box]
    if frame is None:
        frame = LocalFrame(0.5 * (lat_min + lat_max), 0.5 * (lon_min + lon_max), 0.0)
    counts = {"events": len(perception.events), "segments": len(segments),
              "silences": len(perception.silences),
              "pair_gate_combos_evaluated": 0,
              "pair_gate_truncated_groups": 0,
              "pair_gate_events_pruned": 0,
              "max_blobs_to_score_effective": 0,
              "blobs_truncated_by_max_blobs_to_score": 0,
              "blobs_truncated_by_refine_top_n": 0,
              "refine_errors": 0}
    # Fix 6d / G3b: scale guard -- estimate the grid / CPA size BEFORE
    # allocating the CPA table.  The grid size is checked first
    # (exact, from the box dimensions); the CPA estimate is
    # CORRIDOR-based (see corridor_cpa_estimate), NOT the old
    # product n_cells * n_segments.  That product is the cost of a
    # brute-force scan -- exactly the O(N*M) the funnel (FRAMEWORK
    # section 3, strategy section 2) was designed to avoid: filter
    # by type, grid only in a 10 km corridor, CPA first, spectrogram
    # scoring only on survivors.  Estimating as if doing brute
    # force refused a normal 50 km box (62,500 cells x 3,274
    # segments = 204.6M vs a 5M cap) even though the corridor work
    # after time filter + trim is orders of magnitude smaller.
    # The grid itself is built before the corridor guard: the grid
    # is bounded by the cell guard above (<=500k cells by default),
    # and the corridor estimate needs the grid's actual x/y centres.
    _w_m = (lon_max - lon_min) * EARTH_M_PER_DEG_LAT * math.cos(
        math.radians(0.5 * (lat_min + lat_max)))
    _h_m = (lat_max - lat_min) * EARTH_M_PER_DEG_LAT
    _n_cells_est = max(1, int(round(_w_m / config.grid_spacing_m))) * max(
        1, int(round(_h_m / config.grid_spacing_m)))
    if not config.allow_large_box and _n_cells_est > config.max_grid_cells:
        raise ValueError(
            f"run_search: box would need ~{_n_cells_est:,} grid cells "
            f"(limit: {config.max_grid_cells:,}).  Tile the search "
            "into regional boxes, or use the G4 per-day CPA cache; "
            "set config.allow_large_box=True only if you really mean "
            "to run it in memory.")
    grid = make_grid(lat_min, lat_max, lon_min, lon_max,
                     config.grid_spacing_m, frame, height_fn)
    # Jet deferral (2026-11-08): split BEFORE the guard and the
    # CPA table.  Early segments (prop/turboprop/rotor/unknown +
    # low jets) are the only ones in the early CPA table, pair
    # gate and scoring; high jets are deferred to a per-cell
    # silence pass (see _deferred_jet_silence below) and must NOT
    # be counted in the guard -- otherwise busy boxes still
    # refuse.  ``segments`` below is the early list from here on;
    # the deferred list is kept separately for the silence pass
    # and recorded in the report.
    _all_segments = list(segments)
    segments, deferred_segments = split_early_deferred(segments, config)
    _deferred_packed = _pack_segments(deferred_segments) if deferred_segments else None
    _deferred_est, _deferred_per_seg, _deferred_breakdown = corridor_cpa_estimate(
        deferred_segments, grid, config.rmax_m) if deferred_segments else (0, [], None)
    counts["segments"] = int(len(_all_segments))
    counts["segments_early"] = int(len(segments))
    counts["segments_deferred_jets"] = int(len(deferred_segments))
    counts["cpa_entries_estimated_deferred_jets"] = int(_deferred_est)
    counts["cpa_entries_deferred"] = 0  # filled in at the end, below
    counts["time_deferred_jet_silence_s"] = 0.0  # filled in below
    _corridor_est, _corridor_per_seg, _corridor_breakdown = corridor_cpa_estimate(
        segments, grid, config.rmax_m)
    _cpa_product = len(grid.enu) * max(1, len(segments))
    counts["grid_cells_estimated"] = int(len(grid.enu))
    counts["cpa_entries_estimated"] = int(_corridor_est)
    counts["cpa_entries_estimated_corridor"] = int(_corridor_est)
    counts["cpa_entries_estimated_product"] = int(_cpa_product)
    # Check 4: candidate vs silence-only vs unknown split of the
    # estimate.  Jets are counted at full cost in the total above;
    # this is only the reporting split.  Unknown-class segments
    # are their own bucket, not silence-only (user note 3).
    counts["cpa_entries_estimated_candidate"] = int(
        _corridor_breakdown["candidate"])
    counts["cpa_entries_estimated_silence_only"] = int(
        _corridor_breakdown["silence_only"])
    counts["cpa_entries_estimated_unknown"] = int(
        _corridor_breakdown["unknown"])
    counts["segments_candidate"] = int(
        _corridor_breakdown["n_candidate_segments"])
    counts["segments_silence_only"] = int(
        _corridor_breakdown["n_silence_only_segments"])
    counts["segments_unknown"] = int(
        _corridor_breakdown["n_unknown_segments"])
    counts["corridor_cells_max_per_segment"] = int(
        max(_corridor_per_seg) if _corridor_per_seg else 0)
    counts["corridor_cells_median_per_segment"] = float(
        np.median(_corridor_per_seg) if _corridor_per_seg else 0.0)
    if not config.allow_large_box and _corridor_est > config.max_cpa_entries:
        raise ValueError(
            f"run_search: corridor CPA estimate ~{_corridor_est:,} "
            f"(product estimate would be ~{_cpa_product:,}) exceeds "
            f"the limit {config.max_cpa_entries:,}.  This usually "
            "means too many untrimmed / unfiltered segments: load "
            "with load_segments(t_start=..., t_end=..., "
            "time_margin_s=...) for the recording window, trim "
            "long segments to the box (default in the loader), tile "
            "the search into regional boxes, or use the G4 per-day "
            "CPA cache; set config.allow_large_box=True only if you "
            "really mean to run it in memory.")
    counts["height_nan_count"] = int(getattr(grid, "height_nan_count", 0))
    # Check 6: per-stage timing (seconds), in funnel_counts.
    # load_segments time is NOT here -- the loader is outside
    # run_search; it is recorded in the loader stats as
    # "load_seconds" and printed by the G3b timing harness.
    # Silence time is a SUBSET of the scoring / posterior / refine
    # times below (it is called inside those stages), so it must
    # NOT be added to the stage sum -- see the test.  The same
    # applies to the deferred-jet silence time.
    _sil_time = [0.0]
    _jet_sil_time = [0.0]
    _jet_cpa_cache = {}  # cell_enu (rounded m) -> [(t_arr, d, cls)]
    _jet_cpa_entry_count = [0]

    def _deferred_jet_silence(cell_enu, t0):
        """Silence loglik of the deferred (high-jet) segments at
        one cell, for one implied T0.  The CPA of each deferred
        jet to this cell is computed on demand (same _best_cpa as
        the full table) and cached per cell -- the CPA itself does
        not depend on t0, only the silence-window check does.
        A jet kept early as a low jet is NOT in deferred_segments,
        so it is never double-counted: the early silence pass
        handles it (and excludes it when matched), this pass only
        ever sees high jets, which are never matched as events.
        """
        if not deferred_segments:
            return 0.0
        _j0 = _time.time()
        try:
            key = (round(float(cell_enu[0])), round(float(cell_enu[1])),
                   round(float(cell_enu[2])))
            entries = _jet_cpa_cache.get(key)
            if entries is None:
                entries = _jet_cpa_entries(cell_enu, deferred_segments,
                                           _deferred_packed, config)
                _jet_cpa_cache[key] = entries
                _jet_cpa_entry_count[0] += len(entries)
            total = 0.0
            for (s0, s1) in perception.silences:
                for (t_arr, d, cls) in entries:
                    if s0 <= float(t_arr) - float(t0) <= s1:
                        total += math.log1p(
                            -audibility_prob(float(d), cls, config))
            return float(total)
        finally:
            _jet_sil_time[0] += _time.time() - _j0

    def _sil_timed(sil_cache, cpa, cell, cell_enu, perception_, t0,
                   config_, exclude_seg_idx=()):
        """TOTAL silence loglik for a hypothesis: early table
        (cached, keyed by cell + T0 + matched flights) PLUS the
        deferred-jet pass for this cell.  With deferral off this
        is exactly the old _silence_for_hypothesis result."""
        _s0 = _time.time()
        try:
            early = _silence_for_hypothesis(
                sil_cache, cpa, cell, segments, perception_, t0,
                config_, exclude_seg_idx=exclude_seg_idx)
        finally:
            _sil_time[0] += _time.time() - _s0
        return float(early + _deferred_jet_silence(cell_enu, t0))

    def _silence_for_refine(cpa, cell, cell_enu, t0, exclude_seg_idx=()):
        """Same total silence, for refine_blob's local grids
        (no cache: refine cells are visited once per level)."""
        early = silence_loglik(cpa, cell, segments, perception, t0,
                               config, exclude_seg_idx=exclude_seg_idx)
        return float(early + _deferred_jet_silence(cell_enu, t0))

    # F3/F4
    _t0 = _time.time()
    cpa = build_cpa_table(grid, segments, perception, config)
    counts["time_cpa_table_s"] = round(_time.time() - _t0, 3)
    counts["cpa_entries"] = int(cpa.cell_ptr[-1])
    counts["cells_with_cpa"] = int(np.sum(np.diff(cpa.cell_ptr) > 0))
    # F5
    _t0 = _time.time()
    if len(perception.events) == 1 and t0_known is not None:
        hyps = single_event_gate(cpa, perception, config, t0_known)
    else:
        hyps = pair_gate(cpa, perception, config, t0_known=t0_known,
                         counts=counts)
    counts["time_pair_gate_s"] = round(_time.time() - _t0, 3)
    counts["pair_hypotheses"] = len(hyps)
    # Silence loglik per hypothesis (Fix 1: keyed by cell + T0 + matched
    # flights -- see _silence_for_hypothesis -- not by cell alone).
    sil_cache = {}
    counts["cells_with_pair_hypothesis"] = len({h.cell for h in hyps})
    # F7
    blobs = collapse_blobs(hyps)
    counts["blobs"] = len(blobs)
    # F8: quick score blob centroids, keep the best few.
    # Fix 6d: the max_blobs_to_score truncation is recorded, not silent.
    # The budget is density-scaled (see effective_max_blobs_to_score):
    # at city density the fixed top-20 cut by dt_err is near-arbitrary
    # and drops the winning blob (G3b NY run_01/run_04, 2026-11-09).
    n_blobs_to_score = effective_max_blobs_to_score(len(blobs), config)
    counts["max_blobs_to_score_effective"] = int(n_blobs_to_score)
    counts["blobs_truncated_by_max_blobs_to_score"] = int(
        max(0, len(blobs) - n_blobs_to_score))
    scored = []
    _t0 = _time.time()
    for blob in blobs[:n_blobs_to_score]:
        # Centroid hypothesis: lowest dt_err in the blob.
        h0 = min(blob.hypotheses, key=lambda h: h.dt_err)
        obs = grid.enu[h0.cell]
        sil = _sil_timed(sil_cache, cpa, h0.cell, grid.enu[h0.cell],
                         perception, h0.t0, config,
                         exclude_seg_idx=h0.seg_idx)
        s = score_hypothesis(segments, obs, h0, perception, config, silence_ll=sil)
        if s is None:
            continue
        if any(v < config.quick_score_min_db for v in s.mean_score.values()):
            continue
        _attach_locked(h0, s)
        scored.append(s)
    counts["time_scoring_s"] = round(_time.time() - _t0, 3)
    scored.sort(key=lambda s: s.loglik_max + s.silence_loglik, reverse=True)
    counts["blobs_scored"] = len(scored)
    counts["blobs_kept_after_quick_score"] = len(scored)
    # F9: refine the top blobs (Fix 2: t0_known threaded through, and
    # single-event + known-T0 blobs are refined too -- they are in
    # `scored` just like pair blobs.  Failures are recorded, not
    # swallowed: see refine_errors below.)
    refined = []
    refine_errors = []
    counts["blobs_truncated_by_refine_top_n"] = int(
        max(0, len(scored) - config.refine_top_n))
    _t0 = _time.time()
    for s in scored[: config.refine_top_n]:
        key = (s.hypothesis.event_idx, s.hypothesis.seg_idx)
        blob = None
        for bl in blobs:
            if (bl.event_idx, bl.seg_idx) == key:
                blob = bl
                break
        if blob is None:
            continue
        try:
            refined.append(refine_blob(segments, blob, perception, config,
                                       frame, height_fn, s, t0_known=t0_known,
                                       silence_fn=_silence_for_refine))
        except Exception as exc:
            import warnings
            warnings.warn(f"refine_blob {key} failed: {exc}")
            refine_errors.append({"blob_key": [list(key[0]), list(key[1])],
                                  "error": str(exc)})
    counts["blobs_refined"] = len(refined)
    counts["refine_errors"] = len(refine_errors)
    counts["time_refine_s"] = round(_time.time() - _t0, 3)
    # M6: posterior on the coarse grid.
    # Fix 6a: TRUE per-cell marginalised loglik.  The old code scored
    # each cell once at the locked (dt, f0) and set
    # total_marg = total_max - gap, where gap was the centroid's
    # (loglik_max - loglik_marg) -- so the marginalised map was just
    # the max map shifted by a per-blob constant, same shape within
    # each blob.  Now each (blob, cell) is scored with a FULL
    # dt/f0 sweep (score_hypothesis), giving a genuine per-cell
    # loglik_max AND loglik_marg.  Cost: one full sweep per scored
    # cell (not per hypothesis -- only the smallest-dt_err hypothesis
    # per (blob, cell) is scored); measured on the G3 run, see
    # README.  Scope: scored-blob cells only, as before -- unscored
    # cells get the background level below.
    import time as _time
    _t_post = _time.time()
    logpost_max = np.full(len(grid.enu), -np.inf)
    logpost_marg = np.full(len(grid.enu), -np.inf)
    scored_by_key = {(s.hypothesis.event_idx, s.hypothesis.seg_idx): s
                     for s in scored}
    scored_keys = set(scored_by_key)
    posterior_cells_scored = 0
    for blob in blobs:
        key = (blob.event_idx, blob.seg_idx)
        if key not in scored_keys:
            continue
        # One hypothesis per cell: the most T0-consistent one.
        best_by_cell = {}
        for h in blob.hypotheses:
            if h.cell not in best_by_cell or h.dt_err < best_by_cell[h.cell].dt_err:
                best_by_cell[h.cell] = h
        for cell, h in best_by_cell.items():
            sil = _sil_timed(sil_cache, cpa, cell, grid.enu[cell],
                             perception, h.t0, config,
                             exclude_seg_idx=h.seg_idx)
            s_cell = score_hypothesis(segments, grid.enu[cell], h, perception,
                                      config, silence_ll=sil)
            if s_cell is None:
                continue
            posterior_cells_scored += 1
            total_max = float(s_cell.loglik_max + sil)
            total_marg = float(s_cell.loglik_marg + sil)
            # Marginalised must not exceed max for the same cell.
            if total_marg > total_max:
                total_marg = total_max
            if total_max > logpost_max[cell]:
                logpost_max[cell] = total_max
            if total_marg > logpost_marg[cell]:
                logpost_marg[cell] = total_marg
    counts["posterior_cells_scored"] = int(posterior_cells_scored)
    counts["posterior_scoring_seconds"] = round(_time.time() - _t_post, 1)
    # Check 6: same value under the stage-timing name, plus the
    # accumulated silence time (a subset of the stages above).
    counts["time_marginalised_sweep_s"] = counts["posterior_scoring_seconds"]
    counts["time_silence_s"] = round(_sil_time[0], 3)
    # Jet deferral: actual deferred-jet CPA entries computed on
    # demand (one per deferred jet per surviving cell evaluated),
    # and the time spent in the deferred-jet silence pass.
    counts["cpa_entries_deferred"] = int(_jet_cpa_entry_count[0])
    counts["time_deferred_jet_silence_s"] = round(_jet_sil_time[0], 3)
    # Cells with a pair hypothesis but no scored blob: a low background
    # level (they matched geometry but not the spectrogram).  This keeps
    # the posterior proper without letting unscored cells dominate.
    if np.isfinite(logpost_max).any():
        bg = float(np.max(logpost_max[np.isfinite(logpost_max)])) - 50.0
    else:
        bg = 0.0
    unscored_cells = set()
    for h in hyps:
        if not np.isfinite(logpost_max[h.cell]):
            unscored_cells.add(h.cell)
    for cell in unscored_cells:
        logpost_max[cell] = bg
        logpost_marg[cell] = bg
    # If nothing was scored at all, the posterior is flat (S3: say "not sure").
    if not np.isfinite(logpost_max).any() or len(scored) == 0:
        logpost_max = np.zeros(len(grid.enu))
        logpost_marg = np.zeros(len(grid.enu))
    # Temperature (FRAMEWORK section 9.3): divide log-likelihoods by T.
    # The background/flat case above is already in log-posterior units.
    if config.temperature != 1.0 and len(scored) > 0:
        # Only rescale the scored part: recompute from the stored values
        # is not possible here, so apply the temperature to the
        # (loglik + silence) values before normalisation by scaling the
        # spread around the maximum.
        for arr in (logpost_max, logpost_marg):
            m = float(np.max(arr[np.isfinite(arr)]))
            arr[:] = m + (arr - m) / config.temperature
    if len(scored) > 0:
        logpost_max = _smooth_logpost(logpost_max, grid.shape,
                                      config.posterior_smooth_sigma_cells)
        logpost_marg = _smooth_logpost(logpost_marg, grid.shape,
                                       config.posterior_smooth_sigma_cells)
    logpost_max_n, prob_max = _normalise_logpost(logpost_max)
    logpost_marg_n, prob_marg = _normalise_logpost(logpost_marg)
    modes = _find_modes(grid, prob_marg, logpost_marg_n, config, segments, blobs)
    # Fix 5: mode fields.  The old code stored the log POSTERIOR
    # probabilities in fields called loglik_max/loglik_marg, left
    # flights empty, and mass was one cell's probability.
    # Now: log_prob_* are the log posterior probabilities of the mode
    # cell (<= 0, equal to log(prob) of that cell); loglik_* are the
    # TRUE scored logliks of the best hypothesis for that mode's blob;
    # flights is populated from that hypothesis; mass is the whole
    # peak's mass (Voronoi sum, see _find_modes).
    # Helper: best scored hypothesis whose blob covers a mode cell,
    # else the closest scored blob (ENU metres).
    def _best_scored_for_mode(mode):
        cell = mode["cell"]
        best = None
        for s in scored:
            blob_cells = None
            for bl in blobs:
                if (bl.event_idx, bl.seg_idx) == (
                        s.hypothesis.event_idx, s.hypothesis.seg_idx):
                    blob_cells = set(bl.cells)
                    break
            if blob_cells is not None and cell in blob_cells:
                key = s.loglik_max + s.silence_loglik
                if best is None or key > best[0]:
                    best = (key, s)
        if best is not None:
            return best[1]
        if not scored:
            return None
        mode_enu = grid.enu[cell, :2]
        return min(scored, key=lambda s: float(
            np.hypot(*(grid.enu[s.hypothesis.cell, :2] - mode_enu))))

    for mode in modes:
        mode["log_prob_max"] = float(logpost_max_n[mode["cell"]])
        mode["log_prob_marg"] = float(logpost_marg_n[mode["cell"]])
        s_best = _best_scored_for_mode(mode)
        if s_best is not None:
            mode["loglik_max"] = float(s_best.loglik_max)
            mode["loglik_marg"] = float(s_best.loglik_marg)
            mode["silence_loglik"] = float(s_best.silence_loglik)
            # Flights: CPA time/distance come from the CPA table entry.
            flights = []
            hyp = s_best.hypothesis
            for j, ev_idx in enumerate(hyp.event_idx):
                seg = segments[hyp.seg_idx[j]]
                cpa_i = hyp.cpa_idx[j]
                flights.append({
                    "icao": seg.icao, "seg": int(seg.seg),
                    # Check 3: piece identity -- (icao, seg, piece)
                    # is the unique segment id.  Two pieces of one
                    # aircraft (split by trim_segments) are two
                    # different segments and can be two events.
                    "piece": int(getattr(seg, "subseg", 0)),
                    "cls": seg.cls,
                    "event_idx": int(ev_idx),
                    "t_cpa_unix": float(cpa.t_cpa[cpa_i]),
                    "t_arr_unix": float(cpa.t_arr[cpa_i]),
                    "d_m": float(cpa.d[cpa_i]),
                    "v_ms": float(cpa.v[cpa_i])})
            mode["flights"] = flights
            mode["blob_key"] = [list(hyp.event_idx), list(hyp.seg_idx)]
        else:
            mode["loglik_max"] = None
            mode["loglik_marg"] = None
            mode["silence_loglik"] = None
            mode["flights"] = []
    # Refined results replace the coarse mode coordinates when they are
    # for the same flight pair as a top mode (within 1 km, ENU metres
    # -- Fix 6c: the old code compared lat/lon differences in degrees).
    for ref in refined:
        ref_enu = np.asarray(ref.best_enu, dtype=float)
        for mode in modes:
            mode_enu = grid.enu[mode["cell"], :2]
            if float(np.hypot(*(mode_enu - ref_enu[:2]))) < 1000.0:
                # Keep the highest-loglik refined result per mode.
                if ("refined_loglik" not in mode
                        or ref.best_loglik > mode["refined_loglik"]):
                    mode["lat_refined"] = ref.best_lat
                    mode["lon_refined"] = ref.best_lon
                    mode["refined_loglik"] = ref.best_loglik
                break
    peak_max_lat, peak_max_lon, _p = _peak_from(grid, prob_max)
    peak_marg_lat, peak_marg_lon, _p2 = _peak_from(grid, prob_marg)
    # Check 6: total search time (excludes write_outputs below).
    counts["time_total_s"] = round(_time.time() - _t_run0, 3)
    # Check 2: record the time filter / margin assumption.
    # User note 5 (2026-11-08): (a) with no filter the report must
    # NOT say "±0 s" -- it says "no time filter (whole day)" and
    # margin_s is None; (b) the margin is described as the WINDOW
    # margin, not automatically as start-time uncertainty -- the
    # uncertainty may already be built into t_start/t_end (see
    # margin_already_in_window above), and the report says which.
    _filtered = t_start is not None or t_end is not None
    _margin = (float(time_margin_s)
               if (_filtered and time_margin_s is not None) else None)
    if not _filtered:
        _tf_assumption = (
            "No time filter (whole day): the loader kept the whole "
            "day, so no recording-start-time assumption is made and "
            "no margin applies.  The size question comes back -- "
            "the corridor guard is the only thing standing between "
            "a whole-day load and the 5M cap.  State that no time "
            "filter was used in every G6-G8 result.")
    elif margin_already_in_window:
        _tf_assumption = (
            "The time window [t_start, t_end] already includes the "
            "recording-start-time (T0) uncertainty -- the caller "
            "folded it into the window ends.  The margin "
            f"({_margin:g} s) is only extra padding applied on top "
            "of that window by the loader, NOT additional "
            "start-time uncertainty.  The window must cover T0 "
            "uncertainty + recording duration + sound delay + the "
            "Δt between events.  If only the date is known, use no "
            "filter: the loader then keeps the whole day.  State "
            "this window in every G6-G8 result.")
    else:
        _tf_assumption = (
            "The time window is [t_start, t_end] plus a margin of "
            f"±{_margin:g} s applied on top by the loader, so the "
            "method assumes the recording start time (T0) is known "
            f"to within ±{_margin:g} s beyond the window ends.  "
            "The window must cover T0 uncertainty + recording "
            "duration + sound delay + the Δt between events.  If "
            "only the date is known, use no filter: the loader "
            "then keeps the whole day.  State this margin in every "
            "G6-G8 result.")
    report = {
        "config": config.to_dict(),
        "box": [lat_min, lat_max, lon_min, lon_max],
        "t0_known": t0_known,
        "time_filter": {
            "t_start": (float(t_start) if t_start is not None else None),
            "t_end": (float(t_end) if t_end is not None else None),
            "margin_s": _margin,
            "filtered": bool(_filtered),
            "margin_already_in_window": bool(margin_already_in_window),
            "assumption": _tf_assumption},
        "funnel_counts": counts,
        "refine_errors": refine_errors,
        "warnings": [],
        "events": [{"t_c": float(e.t_c), "d": float(e.d), "v": float(e.v),
                    "f0": float(e.f0), "harmonics": list(e.harmonics),
                    "span": [float(e.span[0]), float(e.span[1])]}
                   for e in perception.events],
        "silences": [[float(a), float(b)] for a, b in perception.silences],
        "segments": [{"icao": s.icao, "seg": int(s.seg), "cls": s.cls,
                      # Check 3: piece identity -- (icao, seg, piece).
                      "piece": int(getattr(s, "subseg", 0)),
                      "t_start": float(s.t[0]), "t_end": float(s.t[-1]),
                      "n_points": int(len(s.t))} for s in segments],
        # Jet deferral: high jets are NOT in "segments" above (that
        # is the early list); they are listed here, with the design
        # and its accepted limitation, so nothing is hidden.
        "deferred_segments": [{"icao": s.icao, "seg": int(s.seg),
                               "cls": s.cls,
                               "piece": int(getattr(s, "subseg", 0)),
                               "t_start": float(s.t[0]),
                               "t_end": float(s.t[-1]),
                               "min_alt_m": float(np.min(np.asarray(s.enu)[:, 2])),
                               "n_points": int(len(s.t))}
                              for s in deferred_segments],
        "jet_deferral": {
            "defer_jets": bool(getattr(config, "defer_jets", False)),
            "jet_early_max_alt_m": float(getattr(
                config, "jet_early_max_alt_m", 3000.0)),
            "proxy": ("A jet is kept early if any point is at or "
                      "below jet_early_max_alt_m in ELLIPSOID height, "
                      "not height above ground; in high terrain this "
                      "keeps more jets early (safe direction). "
                      "Terrain-based AGL is a G4 refinement."),
            "accepted_limitation": (
                "A high jet's sound could in principle be mistaken "
                "for a prop's near an airport, and deferring high "
                "jets removes the chance of matching it as an "
                "event.  Accepted: a jet loud enough to be heard "
                "as an event is almost always low, and low jets "
                "stay early.")},
        "modes": modes,
        "peak_max": {"lat": peak_max_lat, "lon": peak_max_lon},
        "peak_marg": {"lat": peak_marg_lat, "lon": peak_marg_lon},
        "scored_hypotheses": [
            {"event_idx": list(s.hypothesis.event_idx),
             "seg_idx": list(s.hypothesis.seg_idx),
             "flights": [{"icao": segments[i].icao, "seg": int(segments[i].seg),
                          "piece": int(getattr(segments[i], "subseg", 0)),
                          "cls": segments[i].cls}
                         for i in s.hypothesis.seg_idx],
             "t0": float(s.hypothesis.t0), "dt_err": float(s.hypothesis.dt_err),
             "best_dt": float(s.best_dt),
             "best_f0": {str(k): float(v) for k, v in s.best_f0.items()},
             "mean_score_db": {str(k): float(v) for k, v in s.mean_score.items()},
             "loglik_max": float(s.loglik_max),
             "loglik_marg": float(s.loglik_marg),
             "silence_loglik": float(s.silence_loglik)}
            for s in scored],
        "refined": [{"flights": [{"icao": segments[i].icao, "seg": int(segments[i].seg),
                                  "piece": int(getattr(segments[i], "subseg", 0)),
                                  "cls": segments[i].cls}
                                 for i in ref.blob_key[1]],
                     "lat": ref.best_lat, "lon": ref.best_lon,
                     "loglik": float(ref.best_loglik),
                     "best_dt": float(ref.best_dt),
                     "best_f0": {str(k): float(v) for k, v in ref.best_f0.items()}}
                    for ref in refined],
    }
    result = SearchResult(grid=grid, logpost_max=logpost_max_n,
                          logpost_marg=logpost_marg_n, prob_max=prob_max,
                          prob_marg=prob_marg, modes=modes, hypotheses=hyps,
                          blobs=blobs, scored=scored, refined=refined,
                          funnel_counts=counts, config=config, frame=frame,
                          box=(lat_min, lat_max, lon_min, lon_max),
                          t0_known=t0_known, report=report)
    result._perception = perception
    result._segments = segments
    if out_dir is not None:
        _t_w0 = _time.time()
        result.report["outputs"] = write_outputs(result, out_dir)
        counts["time_write_outputs_s"] = round(_time.time() - _t_w0, 3)
    return result


def _peak_from(grid, prob):
    i = int(np.argmax(prob))
    return float(grid.lats[i]), float(grid.lons[i]), float(prob[i])


__all__ = [
    "SearchConfig", "Segment", "segments_from_tracks", "load_segments",
    "load_segments_with_stats", "trim_segments", "corridor_cpa_estimate",
    "split_early_deferred",
    "Grid", "make_grid", "CpaTable", "build_cpa_table", "event_gates",
    "Hypothesis", "pair_gate", "single_event_gate",
    "audibility_prob", "silence_loglik", "Blob", "collapse_blobs",
    "effective_max_blobs_to_score",
    "ScoredHypothesis", "score_hypothesis", "refine_blob",
    "SearchResult", "credible_masks", "write_outputs",
    "write_diagnostic_plot", "run_search",
]
