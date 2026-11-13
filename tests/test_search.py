"""Tests for geoloc/search.py (M3-M6).

Small, quick tests use constructed tracks / spectrograms (no data,
no network).  The one end-to-end test renders a synthetic scene with
synth.render_scene() and runs audio.detect() for real; it is the
slowest test in the file (~30 s, mostly rendering).
"""
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from geoloc import audio as audio_mod
from geoloc.geodesy import LocalFrame
from geoloc.physics import Track, heard_curve
from geoloc.search import (
    SearchConfig, build_cpa_table, collapse_blobs, credible_masks,
    effective_max_blobs_to_score, event_gates, make_grid, pair_gate,
    run_search, score_hypothesis, segments_from_tracks, silence_loglik,
    audibility_prob,
)
from geoloc.synth import render_scene
from geoloc.synthetic_scene import TRUTH_LAT, TRUTH_LON, T0, build_scene


def _fake_spec(t_max=700.0):
    t = np.arange(0.0, t_max, 0.25)
    f = np.arange(30.0, 501.0, 1.0)
    S = np.zeros((len(f), len(t)))
    return audio_mod.Spectrogram(t=t, f=f, S=S, loud=np.zeros(len(t)))


def _perc(events, silences=None, spec=None):
    return audio_mod.Perception(duration=700.0, events=events,
                                silences=silences or [], spec=spec or _fake_spec())


def _event(t_c, d, v, f0, harmonics=(1, 2, 3), span=None):
    return audio_mod.Event(t_c=t_c, d=d, v=v, f0=f0, sigma_t=0.5, sigma_d=50.0,
                           sigma_v=2.0, harmonics=list(harmonics),
                           span=span or (t_c - 100, t_c + 100))


def _straight_track(cpa_enu, heading_deg, speed, alt, t_cpa, t_span=450, dt=5.0):
    h = np.radians(heading_deg)
    direction = np.array([np.sin(h), np.cos(h), 0.0])
    t = np.arange(t_cpa - t_span, t_cpa + t_span + 1, dt, dtype=float)
    pos = np.asarray(cpa_enu, float)[None, :] + (t - t_cpa)[:, None] * speed * direction[None, :]
    pos[:, 2] = alt
    return Track(t, pos)


# --- F6 audibility -------------------------------------------------------

def test_audibility_soft_and_monotonic():
    cfg = SearchConfig()
    p2 = audibility_prob(2000.0, "prop", cfg)
    p6 = audibility_prob(6000.0, "prop", cfg)
    p10 = audibility_prob(10000.0, "prop", cfg)
    assert p2 > 0.99
    assert 0.05 < p6 < 0.30  # ~0.13 from the synth model
    assert p10 < 0.01
    assert p2 > p6 > p10
    # A jet at the same distance is quieter (level 0.3 in synth).
    assert audibility_prob(2000.0, "jet", cfg) < p2
    # A faint 5.9 km pass is a small penalty, never an elimination.
    assert 0.0 < -math.log1p(-audibility_prob(5900.0, "prop", cfg)) < 1.0


# --- F3/F4 CPA table -----------------------------------------------------

def test_cpa_table_finds_truth():
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    # One eastbound prop, CPA 1500 m north of truth at T0+250 (emission).
    tr = _straight_track([0.0, 1500.0, 1000.0], 90, 60.0, 1000.0, T0 + 250)
    segs = segments_from_tracks([("A", 0, "prop", tr)])
    # Truth observer is the frame origin; event as M1 would report it.
    ev = _event(255.25, 1803.0, 60.0, 90.0)
    perc = _perc([ev])
    grid = make_grid(TRUTH_LAT - 0.02, TRUTH_LAT + 0.02, TRUTH_LON - 0.02,
                     TRUTH_LON + 0.02, 1000.0, frame, height_fn=0.0)
    cpa = build_cpa_table(grid, segs, perc, SearchConfig())
    # Cell nearest the origin should have a CPA matching physics.
    # (The nearest 1000 m cell is ~500 m from the origin, so its CPA
    # distance is larger than the 1803 m at the origin itself.)
    from geoloc.physics import closest_approach
    dists = np.hypot(grid.enu[:, 0], grid.enu[:, 1])
    g = int(np.argmin(dists))
    a, b = cpa.entries_for_cell(g)
    assert b - a >= 1
    t_cpa_ref, d_ref, t_arr_ref = closest_approach(tr, grid.enu[g])
    ds = cpa.d[a:b]
    i = int(np.argmin(np.abs(ds - d_ref)))
    assert abs(ds[i] - d_ref) < 50.0, (ds[i], d_ref)
    assert abs(cpa.v[a + i] - 60.0) < 5.0
    assert abs((cpa.t_arr[a + i] - T0) - (t_arr_ref - T0)) < 2.0
    # The CPA is within rmax, so it is present in the silence pool
    # (mask may be 0 or 1 depending on the event gate for this cell).


def _turning_event():
    """An M1 event modelled on G3b newyork run_00 (rotor): M1 reports
    d = 890 m, v = 51 m/s, while the flight's geometric CPA is 2052 m
    at 56 m/s.  The aircraft turned near CPA, so the sound-implied
    distance is far below the geometric one; sigma_d is large because
    the straight-pass fit is uncertain on a turning track.
    d_tol = max(0.20 * 890, 3 * 200) = 600, so the old symmetric gate
    tops out at 1490 m and the new asymmetric gate at 2090 m."""
    return audio_mod.Event(t_c=256.0, d=890.0, v=51.0, f0=90.0,
                           sigma_t=0.5, sigma_d=200.0, sigma_v=2.0,
                           harmonics=[1, 2, 3], span=(156.0, 356.0))


def test_event_gates_asymmetric_high_side_only():
    """F4 distance gate (2026-11-09): the high side is widened by
    d_gate_high_factor (default 2.0); the low side and the speed gate
    are unchanged.  Factor 1.0 reproduces the old symmetric gate."""
    ev = _event(255.25, 1000.0, 60.0, 90.0)  # sigma_d=50, sigma_v=2
    perc = _perc([ev])
    # d_tol = max(0.20*1000, 3*50) = 200; v_tol = max(0.20*60, 3*2) = 12.
    d_lo, d_hi, v_lo, v_hi = event_gates(perc, SearchConfig())
    assert d_lo[0] == pytest.approx(800.0)
    assert d_hi[0] == pytest.approx(1000.0 + 2.0 * 200.0)
    assert v_lo[0] == pytest.approx(48.0)
    assert v_hi[0] == pytest.approx(72.0)
    cfg1 = SearchConfig(d_gate_high_factor=1.0)
    d_lo1, d_hi1, v_lo1, v_hi1 = event_gates(perc, cfg1)
    assert d_lo1[0] == pytest.approx(d_lo[0]), "low side must not move"
    assert d_hi1[0] == pytest.approx(1200.0), "factor 1.0 = old symmetric gate"
    assert v_lo1[0] == pytest.approx(v_lo[0])
    assert v_hi1[0] == pytest.approx(v_hi[0]), "speed gate ignores the factor"
    cfg3 = SearchConfig(d_gate_high_factor=3.0)
    _, d_hi3, _, _ = event_gates(perc, cfg3)
    assert d_hi3[0] == pytest.approx(1600.0)


def test_event_gates_turning_aircraft_accepted():
    """Turning aircraft (G3b newyork run_00): geometric CPA 2052 m at
    56 m/s against an M1 event of 890 m at 51 m/s.  The old symmetric
    gate rejects it on distance (2052 > 1490); the new asymmetric gate
    accepts it (2052 <= 2090), with the speed gate passing either way."""
    perc = _perc([_turning_event()])
    d_lo, d_hi, v_lo, v_hi = event_gates(perc, SearchConfig())
    assert d_lo[0] == pytest.approx(290.0)
    assert d_hi[0] == pytest.approx(2090.0)
    assert d_lo[0] <= 2052.0 <= d_hi[0]
    assert v_lo[0] <= 56.0 <= v_hi[0]
    d_lo1, d_hi1, _, _ = event_gates(perc, SearchConfig(d_gate_high_factor=1.0))
    assert d_hi1[0] == pytest.approx(1490.0)
    assert not (d_lo1[0] <= 2052.0 <= d_hi1[0])


def test_cpa_table_mask_turning_aircraft():
    """Same turning case at the CPA-table level: some cell's CPA for
    the track (slant ~2059 m at the origin, v = 56 m/s) must carry the
    event's mask bit under the new gate and not under the old one.
    The table geometry is identical in both runs; only the mask moves."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    tr = _straight_track([0.0, 1800.0, 1000.0], 90, 56.0, 1000.0, T0 + 250)
    segs = segments_from_tracks([("R", 0, "rotor", tr)])
    perc = _perc([_turning_event()])
    grid = make_grid(TRUTH_LAT - 0.02, TRUTH_LAT + 0.02, TRUTH_LON - 0.02,
                     TRUTH_LON + 0.02, 200.0, frame, height_fn=0.0)
    cpa_new = build_cpa_table(grid, segs, perc, SearchConfig())
    cpa_old = build_cpa_table(grid, segs, perc, SearchConfig(d_gate_high_factor=1.0))
    assert len(cpa_new.d) == len(cpa_old.d) > 0
    assert np.array_equal(cpa_new.seg, cpa_old.seg)
    assert np.allclose(cpa_new.d, cpa_old.d)
    # Entries whose distance is inside the new gate but outside the old.
    in_window = (cpa_new.d > 1490.0) & (cpa_new.d <= 2090.0)
    assert in_window.any(), "no CPA falls in the widened band"
    idx = np.nonzero(in_window)[0]
    assert (cpa_new.mask[idx] & 1).all(), "new gate must mark these CPAs"
    assert not (cpa_old.mask[idx] & 1).any(), "old gate must not mark them"


# --- F5 pair gate --------------------------------------------------------

def _pair_scene():
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    tr_a = _straight_track([0.0, 1500.0, 1000.0], 90, 60.0, 1000.0, T0 + 250)
    tr_b = _straight_track([2000.0, 0.0, 1200.0], 0, 55.0, 1200.0, T0 + 850)
    segs = segments_from_tracks([("A", 0, "prop", tr_a), ("B", 1, "prop", tr_b)])
    ev_a = _event(255.25, 1803.0, 60.0, 90.0)
    ev_b = _event(856.8, 2332.0, 55.0, 110.0)
    perc = _perc([ev_a, ev_b])
    grid = make_grid(TRUTH_LAT - 0.03, TRUTH_LAT + 0.03, TRUTH_LON - 0.03,
                     TRUTH_LON + 0.03, 1000.0, frame, height_fn=0.0)
    return frame, segs, perc, grid


def test_pair_gate_recovers_truth_and_t0():
    _frame, segs, perc, grid = _pair_scene()
    cpa = build_cpa_table(grid, segs, perc, SearchConfig())
    hyps = pair_gate(cpa, perc, SearchConfig())
    assert hyps, "pair gate found nothing"
    # Best hypothesis (smallest dt_err) should be at/near the origin.
    h = min(hyps, key=lambda x: x.dt_err)
    assert h.dt_err < 2.0
    assert abs(h.t0 - T0) < 3.0
    assert set(h.seg_idx) == {0, 1}
    lat, lon = grid.lats[h.cell], grid.lons[h.cell]
    assert abs(lat - TRUTH_LAT) < 0.02 and abs(lon - TRUTH_LON) < 0.02


def test_pair_gate_rejects_wrong_dt():
    _frame, segs, perc, grid = _pair_scene()
    # Shift event B by +30 s: no pair should agree within 5 s.
    perc.events[1].t_c += 30.0
    cpa = build_cpa_table(grid, segs, perc, SearchConfig())
    hyps = pair_gate(cpa, perc, SearchConfig())
    # Hypotheses may still exist at other cells by coincidence, but the
    # true flight pair at the true cell must not survive with small error.
    truth_cell = int(np.argmin(np.hypot(grid.enu[:, 0], grid.enu[:, 1])))
    good = [h for h in hyps if h.cell == truth_cell and set(h.seg_idx) == {0, 1}]
    assert not good


def test_silence_penalises_loud_but_not_faint():
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    # Flight D heading north, CPA 12000 m east of truth at T0+465.
    tr_d = _straight_track([12000.0, 0.0, 1000.0], 0, 60.0, 1000.0, T0 + 465)
    segs = segments_from_tracks([("D", 0, "prop", tr_d)])
    ev = _event(255.25, 1803.0, 60.0, 90.0)
    # Silence 450-500 s (audio): D arrives at truth at ~500 s, just
    # outside; at a cell 9000 m east it arrives in-window at ~3000 m.
    perc = _perc([ev], silences=[(450.0, 500.0)])
    # Wide box east so a cell 9000 m east of truth exists.
    grid = make_grid(TRUTH_LAT - 0.03, TRUTH_LAT + 0.03, TRUTH_LON - 0.12,
                     TRUTH_LON + 0.12, 1000.0, frame, height_fn=0.0)
    cpa = build_cpa_table(grid, segs, perc, SearchConfig())
    # Truth cell: D is ~12 km away, P~0, penalty ~0.
    g_truth = int(np.argmin(np.hypot(grid.enu[:, 0], grid.enu[:, 1])))
    ll_truth = silence_loglik(cpa, g_truth, segs, perc, float(T0), SearchConfig())
    assert ll_truth > -0.05
    # East cell (9000 m east): D passes ~3000 m away in the silence.
    g_east = int(np.argmin(np.hypot(grid.enu[:, 0] - 9000, grid.enu[:, 1])))
    ll_east = silence_loglik(cpa, g_east, segs, perc, float(T0), SearchConfig())
    assert ll_east < -0.5, f"east silence loglik {ll_east}"


# --- F8 scoring ----------------------------------------------------------

def test_scoring_true_curve_beats_shifted():
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    tr_a = _straight_track([0.0, 1500.0, 1000.0], 90, 60.0, 1000.0, T0 + 250)
    tr_b = _straight_track([2000.0, 0.0, 1200.0], 0, 55.0, 1200.0, T0 + 850)
    segs = segments_from_tracks([("A", 0, "prop", tr_a), ("B", 1, "prop", tr_b)])
    # Artificial spectrogram: +15 dB exactly on the true curves.
    spec = _fake_spec(1100.0)
    obs = np.array([0.0, 0.0, 0.0])
    for tr, f0, span in [(tr_a, 90.0, (125.0, 425.0)), (tr_b, 110.0, (700.0, 1010.0))]:
        t_frames = spec.t[(spec.t >= span[0]) & (spec.t <= span[1])]
        f_true = heard_curve(tr, obs, t_frames + T0, f0, c=343.0)
        for k in (1, 2, 3):
            fk = f_true * k
            fi = np.rint((fk - spec.f[0]) / (spec.f[1] - spec.f[0])).astype(int)
            ti = np.rint((t_frames - spec.t[0]) / (spec.t[1] - spec.t[0])).astype(int)
            ok = (fi >= 0) & (fi < len(spec.f)) & (ti >= 0) & (ti < len(spec.t))
            spec.S[fi[ok], ti[ok]] = 15.0
    ev_a = _event(255.25, 1803.0, 60.0, 90.0, span=(125.0, 425.0))
    ev_b = _event(856.8, 2332.0, 55.0, 110.0, span=(700.0, 1010.0))
    perc = audio_mod.Perception(duration=1100.0, events=[ev_a, ev_b],
                                silences=[], spec=spec)
    from geoloc.search import Hypothesis
    hyp = Hypothesis(cell=0, event_idx=(0, 1), seg_idx=(0, 1), cpa_idx=(0, 1),
                     t0=float(T0), dt_err=0.0)
    cfg = SearchConfig()
    s_true = score_hypothesis(segs, obs, hyp, perc, cfg)
    assert s_true is not None
    assert s_true.mean_score[0] > 8.0, s_true.mean_score
    assert s_true.mean_score[1] > 8.0, s_true.mean_score
    # Shifted observer 5 km east: curves no longer line up.
    obs_shift = np.array([5000.0, 0.0, 0.0])
    s_shift = score_hypothesis(segs, obs_shift, hyp, perc, cfg)
    assert s_shift is not None
    assert s_shift.loglik_max < s_true.loglik_max - 20.0
    # Marginalised <= max, and close when the peak is sharp.
    assert s_true.loglik_marg <= s_true.loglik_max + 1e-6
    assert s_true.loglik_max - s_true.loglik_marg < 30.0


# --- G3 end to end -------------------------------------------------------

def test_g3_end_to_end_small(tmp_path):
    # Search frame centred 8 km east / 6 km north of truth, so truth is
    # off-centre in a 20 km box.
    centre_lat = TRUTH_LAT + 6000.0 / 111320.0
    centre_lon = TRUTH_LON + 8000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT)))
    frame = LocalFrame(centre_lat, centre_lon, 0.0)
    scene = build_scene(frame, dt_between=600.0, duration=1100.0, seed=1)
    y, truth = render_scene(scene["sources"], np.array([0.0, 0.0, 0.0]),
                            scene["t0"], scene["duration"], fs=4000, seed=1)
    perc = audio_mod.detect(y, fs=4000)
    # The two target events must be among the detected events.
    assert len(perc.events) >= 2
    # Identify them by f0 near 90 / 110 (or 180, the 2nd harmonic of A).
    f0s = sorted(e.f0 for e in perc.events)
    assert any(abs(f - 90.0) < 3 or abs(f - 180.0) < 3 for f in f0s), f0s
    assert any(abs(f - 110.0) < 3 for f in f0s), f0s
    box = (centre_lat - 10000.0 / 111320.0, centre_lat + 10000.0 / 111320.0,
           centre_lon - 10000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT))),
           centre_lon + 10000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT))))
    cfg = SearchConfig(grid_spacing_m=500.0, refine_spacings_m=(100.0, 25.0, 5.0),
                       refine_window_half_m=(500.0, 150.0, 50.0))
    out = tmp_path / "g3"
    result = run_search(perc, scene["segments"], box, height_fn=0.0,
                        config=cfg, out_dir=out, frame=frame)
    assert result.funnel_counts["pair_hypotheses"] > 0
    assert len(result.scored) >= 1
    # Top mode (marginalised) within 1 km of truth.
    lat_m, lon_m, _p = result.peak(which="marg")
    frame_truth = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    d_m = float(np.hypot(*frame_truth.to_enu(np.array([lat_m]), np.array([lon_m]),
                                             np.array([0.0]))[0, :2]))
    assert d_m < 1000.0, f"peak error {d_m:.0f} m at {lat_m},{lon_m}"
    # Truth inside the 90% credible region.
    assert result.truth_in_region(TRUTH_LAT, TRUTH_LON, level=0.90, which="marg")
    # Outputs were written.
    assert (out / "heatmap.tif").exists()
    assert (out / "credible_regions.geojson").exists()
    assert (out / "report.json").exists()
    assert (out / "diagnostic.png").exists()
    # Max vs marginalised peaks agree (both in the report).
    assert abs(result.report["peak_max"]["lat"] - result.report["peak_marg"]["lat"]) < 0.02


# --- load_segments (DuckDB, NaN alt_geom) --------------------------------

def test_load_segments_nan_alt_interpolated(tmp_path):
    """Fix 6b: NaN alt_geom *inside* a segment is interpolated in time,
    NOT silently replaced by alt_baro (different datum)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from geoloc.search import load_segments
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    # Track A in lat/lon: eastbound at 41.65 N, alt_geom NaN on half the points.
    lats = np.full(20, TRUTH_LAT + 1500.0 / 111320.0)
    lons = TRUTH_LON + np.linspace(-0.05, 0.05, 20)
    alt_geom = np.full(20, np.nan, dtype=np.float32)
    alt_geom[::2] = 1000.0
    alt_baro = np.full(20, 950.0, dtype=np.float32)
    table = pa.table({
        "icao": pa.array(["ABC123"] * 20, type=pa.string()),
        "seg": pa.array([0] * 20, type=pa.int16()),
        "t": pa.array([float(T0 + i * 5) for i in range(20)], type=pa.float64()),
        "lat": pa.array(lats, type=pa.float64()),
        "lon": pa.array(lons, type=pa.float64()),
        "alt_geom_m": pa.array(alt_geom, type=pa.float32()),
        "alt_baro_m": pa.array(alt_baro, type=pa.float32()),
        "gs_ms": pa.array([60.0] * 20, type=pa.float32()),
        "track_deg": pa.array([90.0] * 20, type=pa.float32()),
        "source": pa.array(["adsb_icao"] * 20, type=pa.dictionary(pa.int8(), pa.string())),
        "cls": pa.array(["prop"] * 20, type=pa.dictionary(pa.int8(), pa.string())),
    })
    path = tmp_path / "points.parquet"
    pq.write_table(table, path)
    segs = load_segments(path, frame, TRUTH_LAT - 0.05, TRUTH_LAT + 0.05,
                         TRUTH_LON - 0.1, TRUTH_LON + 0.1, stats={})
    assert len(segs) == 1
    assert len(segs[0].t) == 20
    assert np.isfinite(segs[0].enu).all(), "NaN altitude leaked into ENU"
    # Interpolated NaN points should be ~1000 m (geometric), not 950 m (baro).
    assert abs(float(np.median(segs[0].enu[1::2, 2])) - 1000.0) < 50.0


def test_single_event_known_t0_gate():
    from geoloc.search import single_event_gate
    _frame, segs, perc, grid = _pair_scene()
    perc1 = _perc([perc.events[0]])
    cpa = build_cpa_table(grid, segs, perc1, SearchConfig())
    hyps = single_event_gate(cpa, perc1, SearchConfig(), float(T0))
    assert hyps, "single-event + known T0 found nothing"
    h = min(hyps, key=lambda x: x.dt_err)
    assert h.dt_err < 2.0
    # Mirror ambiguity (FRAMEWORK section 1): the best cell may be
    # near truth or its mirror across the track (3000 m north here).
    lat, lon = grid.lats[h.cell], grid.lons[h.cell]
    near_truth = abs(lat - TRUTH_LAT) < 0.02 and abs(lon - TRUTH_LON) < 0.02
    near_mirror = abs(lat - (TRUTH_LAT + 3000.0 / 111320.0)) < 0.02 and abs(lon - TRUTH_LON) < 0.02
    assert near_truth or near_mirror, (lat, lon)


# --- Regression tests: 9 fixes (2026-11-07, REFERENCE.md section 6) ---

# Fix 1: silence cache keyed by (cell, T0, matched flights), not cell alone.
def test_silence_cache_keyed_by_t0_and_flights():
    from geoloc.search import _silence_for_hypothesis
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    tr_d = _straight_track([12000.0, 0.0, 1000.0], 0, 60.0, 1000.0, T0 + 465)
    segs = segments_from_tracks([("D", 0, "prop", tr_d)])
    ev = _event(255.25, 1803.0, 60.0, 90.0)
    perc = _perc([ev], silences=[(450.0, 500.0)])
    grid = make_grid(TRUTH_LAT - 0.03, TRUTH_LAT + 0.03, TRUTH_LON - 0.12,
                     TRUTH_LON + 0.12, 1000.0, frame, height_fn=0.0)
    cpa = build_cpa_table(grid, segs, perc, SearchConfig())
    g_east = int(np.argmin(np.hypot(grid.enu[:, 0] - 9000, grid.enu[:, 1])))
    cfg = SearchConfig()
    cache = {}
    # Same cell, T0 shifted by 30 s: D's arrival moves out of the silence.
    ll1 = _silence_for_hypothesis(cache, cpa, g_east, segs, perc, float(T0), cfg)
    ll2 = _silence_for_hypothesis(cache, cpa, g_east, segs, perc, float(T0 + 30), cfg)
    assert ll1 < -0.5, ll1
    assert ll2 > -0.05, ll2
    assert ll1 != ll2, "same-cell hypotheses with different T0 got the same silence score"
    # Same cell + T0, but the flight excluded (matched): penalty must vanish.
    ll_excl = _silence_for_hypothesis(cache, cpa, g_east, segs, perc, float(T0), cfg,
                                      exclude_seg_idx=(0,))
    assert ll_excl == 0.0, ll_excl
    # Cache hit for an identical key returns the identical value.
    assert _silence_for_hypothesis(cache, cpa, g_east, segs, perc, float(T0), cfg) == ll1


# Fix 3: closest-first cap, and >500 combos recorded, not silently skipped.
def test_pair_gate_closest_12_survives_and_combos_recorded():
    from geoloc.search import CpaTable, pair_gate
    cfg = SearchConfig()
    # One cell, 15 candidate CPAs for event 0 (all with dt_err 0), one for event 1.
    # The true flight (seg 99) is LAST in table order but CLOSEST in distance.
    n0 = 15
    seg = np.array([i for i in range(n0)] + [100], dtype=np.int64)
    # Table order: distractors first (d 1600..2100), true flight last (d 1500).
    d = np.array([1600.0 + 35.0 * i for i in range(n0)] + [2332.0])
    d[n0 - 1] = 1500.0  # placeholder overwritten below; true entry is separate
    # Rebuild cleanly: entries 0..13 distractors, entry 14 = true, entry 15 = event 1.
    seg = np.array(list(range(14)) + [99, 100], dtype=np.int64)
    d = np.array([1600.0 + 35.0 * i for i in range(14)] + [1500.0, 2332.0])
    v = np.array([60.0] * 15 + [55.0])
    t_arr = np.array([float(T0 + 255.25)] * 15 + [float(T0 + 856.8)])
    mask = np.array([1] * 15 + [2], dtype=np.int64)
    cpa = CpaTable(cell_ptr=np.array([0, 16], dtype=np.int64), seg=seg,
                   t_cpa=t_arr - d / 343.0, t_arr=t_arr, d=d, v=v, mask=mask,
                   n_cells=1, n_events=2)
    ev_a = _event(255.25, 1803.0, 60.0, 90.0)
    ev_b = _event(856.8, 2332.0, 55.0, 110.0)
    perc = _perc([ev_a, ev_b])
    counts = {}
    hyps = pair_gate(cpa, perc, cfg, counts=counts)
    assert any(set(h.seg_idx) == {99, 100} for h in hyps), \
        "true (closest, last in table order) flight was dropped by the cap"
    # >500 combos: 12 x 12 x 12 for three events must be truncated, not skipped.
    ev_c = _event(573.5, 2000.0, 60.0, 95.0)
    perc3 = _perc([ev_a, ev_b, ev_c])
    seg3 = np.array(list(range(12)) * 3, dtype=np.int64)
    d3 = np.array([1800.0] * 12 + [2332.0] * 12 + [2000.0] * 12)
    v3 = np.array([60.0] * 12 + [55.0] * 12 + [60.0] * 12)
    t3 = np.array([float(T0 + 255.25)] * 12 + [float(T0 + 856.8)] * 12
                  + [float(T0 + 573.5)] * 12)
    mask3 = np.array([1] * 12 + [2] * 12 + [4] * 12, dtype=np.int64)
    cpa3 = CpaTable(cell_ptr=np.array([0, 36], dtype=np.int64), seg=seg3,
                    t_cpa=t3 - d3 / 343.0, t_arr=t3, d=d3, v=v3, mask=mask3,
                    n_cells=1, n_events=3)
    counts3 = {}
    hyps3 = pair_gate(cpa3, perc3, SearchConfig(max_pair_combos=500), counts=counts3)
    assert counts3["pair_gate_truncated_groups"] >= 1, counts3
    assert counts3["pair_gate_combos_evaluated"] > 0
    assert hyps3, "busy group was silently dropped instead of truncated"


# Fix 6a: marginalised map is a true per-cell sweep, not a constant shift.
def test_marginalised_map_is_true_per_cell_sweep():
    # Reuse the scoring fixture: true curve bright, shifted observer dim.
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    tr_a = _straight_track([0.0, 1500.0, 1000.0], 90, 60.0, 1000.0, T0 + 250)
    tr_b = _straight_track([2000.0, 0.0, 1200.0], 0, 55.0, 1200.0, T0 + 850)
    segs = segments_from_tracks([("A", 0, "prop", tr_a), ("B", 1, "prop", tr_b)])
    spec = _fake_spec(1100.0)
    obs = np.array([0.0, 0.0, 0.0])
    for tr, f0, span in [(tr_a, 90.0, (125.0, 425.0)), (tr_b, 110.0, (700.0, 1010.0))]:
        t_frames = spec.t[(spec.t >= span[0]) & (spec.t <= span[1])]
        f_true = heard_curve(tr, obs, t_frames + T0, f0, c=343.0)
        for k in (1, 2, 3):
            fk = f_true * k
            fi = np.rint((fk - spec.f[0]) / (spec.f[1] - spec.f[0])).astype(int)
            ti = np.rint((t_frames - spec.t[0]) / (spec.t[1] - spec.t[0])).astype(int)
            ok = (fi >= 0) & (fi < len(spec.f)) & (ti >= 0) & (ti < len(spec.t))
            spec.S[fi[ok], ti[ok]] = 15.0
    ev_a = _event(255.25, 1803.0, 60.0, 90.0, span=(125.0, 425.0))
    ev_b = _event(856.8, 2332.0, 55.0, 110.0, span=(700.0, 1010.0))
    perc = audio_mod.Perception(duration=1100.0, events=[ev_a, ev_b],
                                silences=[], spec=spec)
    from geoloc.search import Hypothesis
    hyp = Hypothesis(cell=0, event_idx=(0, 1), seg_idx=(0, 1), cpa_idx=(0, 1),
                     t0=float(T0), dt_err=0.0)
    cfg = SearchConfig()
    s_true = score_hypothesis(segs, obs, hyp, perc, cfg)
    s_shift = score_hypothesis(segs, np.array([5000.0, 0.0, 0.0]), hyp, perc, cfg)
    assert s_true.loglik_marg <= s_true.loglik_max + 1e-6
    assert s_shift.loglik_marg <= s_shift.loglik_max + 1e-6
    # The max-marg gap varies by cell (sweep width varies) -- it is not
    # a constant shift, so the two maps have different shapes.
    gap_true = s_true.loglik_max - s_true.loglik_marg
    gap_shift = s_shift.loglik_max - s_shift.loglik_marg
    assert abs(gap_true - gap_shift) > 0.5, (gap_true, gap_shift)


# Fix 6b (continued): no-alt segment dropped + counted; baro only when allowed.
def test_load_segments_no_alt_dropped_and_baro_only_when_allowed(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from geoloc.search import load_segments, load_segments_with_stats
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    lats = np.full(20, TRUTH_LAT + 1500.0 / 111320.0)
    lons = TRUTH_LON + np.linspace(-0.05, 0.05, 20)
    table = pa.table({
        "icao": pa.array(["ABC123"] * 20, type=pa.string()),
        "seg": pa.array([0] * 20, type=pa.int16()),
        "t": pa.array([float(T0 + i * 5) for i in range(20)], type=pa.float64()),
        "lat": pa.array(lats, type=pa.float64()),
        "lon": pa.array(lons, type=pa.float64()),
        "alt_geom_m": pa.array([np.nan] * 20, type=pa.float32()),
        "alt_baro_m": pa.array([950.0] * 20, type=pa.float32()),
        "gs_ms": pa.array([60.0] * 20, type=pa.float32()),
        "track_deg": pa.array([90.0] * 20, type=pa.float32()),
        "source": pa.array(["adsb_icao"] * 20, type=pa.dictionary(pa.int8(), pa.string())),
        "cls": pa.array(["prop"] * 20, type=pa.dictionary(pa.int8(), pa.string())),
    })
    path = tmp_path / "points.parquet"
    pq.write_table(table, path)
    segs, stats = load_segments_with_stats(
        path, frame, TRUTH_LAT - 0.05, TRUTH_LAT + 0.05,
        TRUTH_LON - 0.1, TRUTH_LON + 0.1)
    assert segs == []
    assert stats["segments_dropped_no_alt"] == 1
    assert stats["alt_baro_fallback_points"] == 0
    # Explicit opt-in: baro is used, and counted.
    segs2, stats2 = load_segments_with_stats(
        path, frame, TRUTH_LAT - 0.05, TRUTH_LAT + 0.05,
        TRUTH_LON - 0.1, TRUTH_LON + 0.1, allow_baro_fallback=True)
    assert len(segs2) == 1
    assert stats2["alt_baro_fallback_points"] == 20
    assert abs(float(np.median(segs2[0].enu[:, 2])) - 950.0) < 50.0


def test_make_grid_nan_height_filled_not_nan():
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)

    def bad_height(lats, lons):
        h = np.zeros_like(lats, dtype=float)
        h[::2] = np.nan
        return h

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = make_grid(TRUTH_LAT - 0.02, TRUTH_LAT + 0.02,
                         TRUTH_LON - 0.02, TRUTH_LON + 0.02,
                         1000.0, frame, height_fn=bad_height)
    assert np.isfinite(grid.enu).all(), "NaN ground height leaked into ENU"
    assert grid.height_nan_count > 0


# Fix 6c: mode clustering in metres, and GeoTIFF CRS / distortion.
def test_find_modes_clusters_in_metres_not_degrees():
    from geoloc.search import _find_modes
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    grid = make_grid(TRUTH_LAT - 0.02, TRUTH_LAT + 0.02,
                     TRUTH_LON - 0.02, TRUTH_LON + 0.02,
                     500.0, frame, height_fn=0.0)
    ny, nx = grid.shape
    prob = np.zeros(len(grid.enu))
    # Two peaks one cell (500 m) apart must cluster into one mode;
    # two peaks three cells (1500 m) apart must stay separate.
    c0 = (ny // 2) * nx + nx // 2
    prob[c0] = 0.5
    prob[c0 + 1] = 0.4  # 500 m east, local max in its own neighbourhood? not a
    # local max vs c0, so use a separate grid for the 1500 m case below.
    modes = _find_modes(grid, prob, np.log(np.maximum(prob, 1e-300)),
                        SearchConfig(), [], [])
    assert len(modes) == 1, modes
    prob2 = np.zeros(len(grid.enu))
    prob2[c0] = 0.5
    prob2[c0 + 3] = 0.4  # 1500 m east
    modes2 = _find_modes(grid, prob2, np.log(np.maximum(prob2, 1e-300)),
                         SearchConfig(), [], [])
    assert len(modes2) == 2, modes2


def test_grid_distortion_grows_with_box():
    from geoloc.search import _grid_distortion
    frame = LocalFrame(41.65, -73.85, 0.0)
    small = make_grid(41.6, 41.7, -73.9, -73.8, 10000.0, frame, height_fn=0.0)
    large = make_grid(40.0, 45.0, -80.0, -75.0, 10000.0, frame, height_fn=0.0)
    d_small = _grid_distortion(small)
    d_large = _grid_distortion(large)
    assert d_large["dlon_rel_variation"] > d_small["dlon_rel_variation"]
    assert d_large["dlon_rel_variation"] > 0.005, d_large


# Fix 6d: scale guard refuses a CONUS-sized box instead of OOMing.
def test_scale_guard_refuses_large_box():
    _frame, segs, perc, _grid = _pair_scene()
    cfg = SearchConfig()  # default limits
    with pytest.raises(ValueError, match="Tile|G4|tile"):
        run_search(perc, segs, (24.0, 50.0, -125.0, -66.0),
                   height_fn=0.0, config=cfg)
    # A small box with a deliberately tiny limit also refuses.
    cfg2 = SearchConfig(max_grid_cells=100)
    with pytest.raises(ValueError, match="Tile|G4|tile"):
        run_search(perc, segs, (TRUTH_LAT - 0.03, TRUTH_LAT + 0.03,
                                TRUTH_LON - 0.03, TRUTH_LON + 0.03),
                   height_fn=0.0, config=cfg2)


# Fix 2: refinement with known T0, single-event refinement, failure recorded.
def test_refine_with_known_t0_and_single_event(tmp_path):
    import math
    # (a) Pair case, known T0: refined peak must exist and its T0 must
    # stay at the known T0 (within tolerance).
    centre_lat = TRUTH_LAT + 6000.0 / 111320.0
    centre_lon = TRUTH_LON + 8000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT)))
    frame = LocalFrame(centre_lat, centre_lon, 0.0)
    scene = build_scene(frame, dt_between=600.0, duration=1100.0, seed=1)
    y, truth = render_scene(scene["sources"], np.array([0.0, 0.0, 0.0]),
                            scene["t0"], scene["duration"], fs=4000, seed=1)
    perc = audio_mod.detect(y, fs=4000)
    assert len(perc.events) >= 2
    box = (centre_lat - 10000.0 / 111320.0, centre_lat + 10000.0 / 111320.0,
           centre_lon - 10000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT))),
           centre_lon + 10000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT))))
    cfg = SearchConfig(grid_spacing_m=1000.0,
                       refine_spacings_m=(100.0, 25.0),
                       refine_window_half_m=(500.0, 150.0))
    result = run_search(perc, scene["segments"], box, height_fn=0.0,
                        config=cfg, out_dir=tmp_path / "pair_t0",
                        frame=frame, t0_known=float(T0))
    assert result.funnel_counts["refine_errors"] == 0, result.report["refine_errors"]
    assert result.refined, "pair + known T0 was not refined"
    assert abs(float(result.refined[0].best_t0) - float(T0)) < 5.0, \
        result.refined[0].best_t0
    # (b) Single event + known T0: refined result must exist (mirror
    # ambiguity allowed -- truth or its 3000 m mirror).  Use a 500 m
    # coarse grid here: on a 1000 m grid the nearest cell is ~680 m
    # from truth and its CPA distance (1427 m) falls just outside
    # the event's +/-20% gate (1443-2165 m), so the gate finds
    # nothing -- a grid-discretisation limit, not a refine bug.
    perc1 = audio_mod.Perception(duration=perc.duration, events=[perc.events[0]],
                                 silences=perc.silences, spec=perc.spec)
    cfg1 = SearchConfig(grid_spacing_m=500.0,
                        refine_spacings_m=(100.0, 25.0),
                        refine_window_half_m=(500.0, 150.0))
    result1 = run_search(perc1, scene["segments"], box, height_fn=0.0,
                         config=cfg1, out_dir=tmp_path / "single_t0",
                         frame=frame, t0_known=float(T0))
    assert result1.refined, "single-event + known T0 was never refined"
    r = result1.refined[0]
    frame_truth = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    d_truth = float(np.hypot(*frame_truth.to_enu(
        np.array([r.best_lat]), np.array([r.best_lon]), np.array([0.0]))[0, :2]))
    d_mirror = float(np.hypot(*frame_truth.to_enu(
        np.array([r.best_lat]), np.array([r.best_lon]), np.array([0.0]))[0, :2]
        - np.array([0.0, 3000.0])))
    assert min(d_truth, d_mirror) < 1500.0, (r.best_lat, r.best_lon, d_truth)


def test_refine_failure_is_recorded_not_swallowed(tmp_path):
    # Height function that works on the coarse grid but raises in the
    # small refine windows (lat span < 0.02 deg).
    def height_fn(lats, lons):
        lats = np.asarray(lats)
        if float(np.ptp(lats)) < 0.02:
            raise RuntimeError("refine height boom")
        return np.zeros_like(lats, dtype=float)

    _frame, segs, perc, _grid = _pair_scene()
    # _pair_scene's fake spectrogram ends at 700 s, but event 2 spans
    # 757-957 s, so scoring would return None for it.  Use a longer spec.
    perc = audio_mod.Perception(duration=1100.0, events=perc.events,
                                silences=[], spec=_fake_spec(1100.0))
    box = (TRUTH_LAT - 0.03, TRUTH_LAT + 0.03, TRUTH_LON - 0.03, TRUTH_LON + 0.03)
    cfg = SearchConfig(grid_spacing_m=1000.0,
                       refine_spacings_m=(100.0,),
                       refine_window_half_m=(500.0,),
                       # Fake spec scores ~0 dB, so lower the quick-score
                       # floor or no blob is scored and refine never runs.
                       quick_score_min_db=-10.0)
    result = run_search(perc, segs, box, height_fn=height_fn,
                        config=cfg, out_dir=tmp_path / "ref_fail")
    assert result.funnel_counts["refine_errors"] >= 1
    assert result.report["refine_errors"], "refinement failure was swallowed"
    assert "boom" in result.report["refine_errors"][0]["error"]


# Fix 4 + Fix 5: three-flight answer in the report, and correct mode fields.
def test_g3_triple_in_report_and_modes_correct(tmp_path):
    import math
    centre_lat = TRUTH_LAT + 6000.0 / 111320.0
    centre_lon = TRUTH_LON + 8000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT)))
    frame = LocalFrame(centre_lat, centre_lon, 0.0)
    scene = build_scene(frame, dt_between=600.0, duration=1100.0, seed=1)
    y, truth = render_scene(scene["sources"], np.array([0.0, 0.0, 0.0]),
                            scene["t0"], scene["duration"], fs=4000, seed=1)
    perc = audio_mod.detect(y, fs=4000)
    assert len(perc.events) >= 3, [e.f0 for e in perc.events]
    box = (centre_lat - 10000.0 / 111320.0, centre_lat + 10000.0 / 111320.0,
           centre_lon - 10000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT))),
           centre_lon + 10000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT))))
    cfg = SearchConfig(grid_spacing_m=500.0, refine_spacings_m=(100.0, 25.0, 5.0),
                       refine_window_half_m=(500.0, 150.0, 50.0))
    out = tmp_path / "g3triple"
    result = run_search(perc, scene["segments"], box, height_fn=0.0,
                        config=cfg, out_dir=out, frame=frame)
    # Fix 4: a 3-event hypothesis with the true flights must be scored
    # and present in report.json (not just pairs / the full 4-set).
    triples = [s for s in result.report["scored_hypotheses"]
               if len(s["event_idx"]) == 3]
    assert triples, "no 3-event hypothesis in the report"
    assert any({f["icao"] for f in s["flights"]} == {"TESTA", "TESTB", "TESTC"}
               for s in triples), triples
    # Fix 5: modes have real flights, whole-peak mass, and log_prob
    # (log posterior) distinct from loglik (scored loglik).
    assert result.modes, "no modes"
    m0 = result.modes[0]
    assert m0["flights"], "mode flights is empty"
    assert {f["icao"] for f in m0["flights"]} <= {"TESTA", "TESTB", "TESTC", "TESTD", "TESTE"}
    single_prob = float(result.prob_marg[m0["cell"]])
    assert m0["mass"] > single_prob, (m0["mass"], single_prob)
    assert 0.0 < m0["mass"] <= 1.0
    assert sum(m["mass"] for m in result.modes) <= 1.0 + 1e-6
    assert m0["log_prob_marg"] <= 0.0
    assert abs(m0["log_prob_marg"] - math.log(max(single_prob, 1e-300))) < 1e-6
    assert m0["loglik_marg"] is not None and m0["loglik_marg"] > 0.0
    assert m0["loglik_marg"] != pytest.approx(m0["log_prob_marg"])
    # Fix 6c: GeoTIFF is in the projected (aeqd) CRS, and its transform
    # maps the peak cell to its ENU coordinates.
    import rasterio
    with rasterio.open(out / "heatmap.tif") as src:
        assert "aeqd" in src.crs.to_proj4(), src.crs
        assert src.crs.to_proj4().count("aeqd") >= 1
    # Fix 6d: truncations are recorded (even if zero in this small run).
    assert "blobs_truncated_by_max_blobs_to_score" in result.funnel_counts
    assert "refine_errors" in result.funnel_counts
    assert result.report["refine_errors"] == []
    # Truth still recovered.
    lat_m, lon_m, _p = result.peak(which="marg")
    frame_truth = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    d_m = float(np.hypot(*frame_truth.to_enu(np.array([lat_m]), np.array([lon_m]),
                                             np.array([0.0]))[0, :2]))
    assert d_m < 1000.0, f"peak error {d_m:.0f} m"
    assert result.truth_in_region(TRUTH_LAT, TRUTH_LON, level=0.90, which="marg")


# --- G3b real-flight fixes (2026-11-08) ------------------------------------
# Corridor estimate, time filter in the loader, spatial trim, per-piece
# identities, trim-edge safety, timing breakdown, and the synthetic
# G3b-like cases (50 km box, 200 m grid, 311 / 3,274 segments).
# Real 2026-11-06 data is NOT in this workspace; the real-day timing
# is run by the user with scripts/g3b_real.py (see README_G3.md).
from geoloc.search import corridor_cpa_estimate, load_segments_with_stats, trim_segments

_G3B_BOX = (TRUTH_LAT - 0.225, TRUTH_LAT + 0.225, TRUTH_LON - 0.30, TRUTH_LON + 0.30)


def _realistic_track(rng, kind, t0):
    """One realistic synthetic track in ENU metres (frame origin = truth).

    ``kind``: "prop" (turning, heading drifts), "jet" (fast, turning),
    "pattern" (airport pattern work: 1-2 circles), "outback"
    (out and back: fly out, reverse, fly back).  Nominal 3 s
    reporting with ~10% dropped points and occasional 12 s gaps,
    like real low-altitude ADS-B.
    """
    x0, y0 = rng.uniform(-45000, 45000, 2)
    if kind in ("prop", "pattern", "outback"):
        alt = rng.uniform(600, 2500)
        speed = rng.uniform(50, 80)
    else:
        alt = rng.uniform(3000, 9000)
        speed = rng.uniform(120, 250)
    dur = rng.uniform(300, 900)
    t = np.arange(t0, t0 + dur, 3.0)
    keep = rng.random(len(t)) > 0.10
    keep[0] = keep[-1] = True
    t = t[keep]
    if kind == "pattern":
        r = rng.uniform(2000, 4000)
        w = speed / r
        ph = rng.uniform(0, 2 * np.pi)
        x = x0 + r * np.cos(ph + w * (t - t0))
        y = y0 + r * np.sin(ph + w * (t - t0))
    elif kind == "outback":
        h = rng.uniform(0, 2 * np.pi)
        d = np.array([np.sin(h), np.cos(h)])
        s = speed * (t - t0)
        half = dur / 2
        along = np.where(t - t0 <= half, s, 2 * speed * half - s)
        x = x0 + along * d[0]
        y = y0 + along * d[1]
    else:
        h0 = rng.uniform(0, 2 * np.pi)
        dh = rng.uniform(-1.0, 1.0)
        h = h0 + dh * (t - t0) / dur
        dt = np.diff(t, prepend=t[0])
        x = x0 + np.cumsum(speed * np.sin(h) * dt)
        y = y0 + np.cumsum(speed * np.cos(h) * dt)
    pos = np.column_stack([x, y, np.full(len(t), alt)])
    return Track(t, pos)


def _g3b_segments(n_prop, n_jet, seed=7):
    """Realistic day-spread segments: 311 / 3,274 like the user's
    Hudson Valley / New York boxes on 2026-11-06.  Segment start
    times are spread over the whole day, so the time filter (not
    the corridor estimate) is what cuts them down to the window."""
    rng = np.random.default_rng(seed)
    items = []
    kinds = ["prop"] * n_prop + ["jet"] * n_jet
    for i, kind in enumerate(kinds):
        k = kind
        if kind == "prop" and i % 20 == 0:
            k = "pattern"
        elif kind == "prop" and i % 33 == 0:
            k = "outback"
        t0 = T0 + rng.uniform(-43200, 43200)
        items.append((f"AC{i:04d}", i % 97,
                      kind if kind != "pattern" else "prop",
                      _realistic_track(rng, k, t0)))
    return segments_from_tracks(items)


def _g3b_window_filtered(segs):
    """The real pipeline's time filter + trim, in memory: window
    [T0-600, T0+2520] (T0 known +/-600 s, 1600 s recording, events
    up to 920 s apart), trimmed to the 50 km box + rmax + margin."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    w0, w1 = T0 - 600, T0 + 2520
    segs = [s for s in segs if s.t[0] <= w1 and s.t[-1] >= w0]
    segs, _st = trim_segments(segs, frame, *_G3B_BOX, rmax_m=10_000.0,
                              min_points=5)
    return segs


def test_corridor_estimate_user_cases_and_jets():
    """Fix 1 + check 4: the user's two cases, 50 km box at 200 m
    (62,500 cells).  The old product estimates are 19.4M (311) and
    204.6M (3,274) vs a 5M cap.  The corridor estimate must be far
    below the product, must count jets (2,884 of 3,274 in NY --
    most of the work), and must pass the guard after time filter
    + trim, where the product would STILL refuse."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    grid = make_grid(*_G3B_BOX, 200.0, frame, height_fn=0.0)
    assert len(grid.enu) == 62_500
    cfg = SearchConfig(grid_spacing_m=200.0)
    # Hudson Valley: 311 props, whole day, trimmed only.
    segs311 = _g3b_segments(311, 0)
    segs311, _ = trim_segments(segs311, frame, *_G3B_BOX, rmax_m=10_000.0,
                               min_points=5)
    est311, _, bd311 = corridor_cpa_estimate(segs311, grid, cfg.rmax_m)
    assert est311 < 5_000_000, est311
    assert est311 < 19_437_500 // 4, est311  # far below the 19.4M product
    # New York: 390 props + 2,884 jets, whole day, trimmed only.
    segs3274 = _g3b_segments(390, 2884)
    # Check 4 (on the untrimmed day load): jets are counted, at full
    # cost, and reported separately -- 2,884 of 3,274 are jets.
    est_raw, _, bd_raw = corridor_cpa_estimate(segs3274, grid, cfg.rmax_m)
    assert bd_raw["n_silence_only_segments"] == 2884
    assert bd_raw["n_candidate_segments"] == 390
    assert bd_raw["silence_only"] > bd_raw["candidate"], bd_raw
    assert (bd_raw["candidate"] + bd_raw["silence_only"]) == est_raw
    segs3274, _ = trim_segments(segs3274, frame, *_G3B_BOX, rmax_m=10_000.0,
                                min_points=5)
    est3274, _, bd3274 = corridor_cpa_estimate(segs3274, grid, cfg.rmax_m)
    assert est3274 < 204_625_000 // 3, est3274  # far below the 204.6M product
    # After the time filter the NY case passes the guard -- where the
    # product estimate (62,500 x survivors) would still refuse.
    filt = _g3b_window_filtered(segs3274)
    assert 50 < len(filt) < 300, len(filt)  # ~95-126, like the real day
    est_f, _, bd_f = corridor_cpa_estimate(filt, grid, cfg.rmax_m)
    assert est_f < 5_000_000, est_f
    assert len(grid.enu) * len(filt) > est_f  # product is the pessimistic one
    assert bd_f["silence_only"] > 0  # jets still in the estimate after filter


def _write_points_parquet(path, rows):
    """Rows: (icao, seg, cls, t_array, lat_array, lon_array, alt)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    icao, seg, t, lat, lon, alt, cls = [], [], [], [], [], [], []
    for _icao, _seg, _cls, _t, _lat, _lon, _alt in rows:
        n = len(_t)
        icao += [_icao] * n
        seg += [_seg] * n
        cls += [_cls] * n
        t += [float(v) for v in _t]
        lat += [float(v) for v in _lat]
        lon += [float(v) for v in _lon]
        alt += [float(_alt)] * n
    table = pa.table({
        "icao": pa.array(icao, type=pa.string()),
        "seg": pa.array(seg, type=pa.int16()),
        "t": pa.array(t, type=pa.float64()),
        "lat": pa.array(lat, type=pa.float64()),
        "lon": pa.array(lon, type=pa.float64()),
        "alt_geom_m": pa.array(alt, type=pa.float32()),
        "alt_baro_m": pa.array(alt, type=pa.float32()),
        "gs_ms": pa.array([60.0] * len(t), type=pa.float32()),
        "track_deg": pa.array([90.0] * len(t), type=pa.float32()),
        "source": pa.array(["adsb_icao"] * len(t),
                           type=pa.dictionary(pa.int8(), pa.string())),
        "cls": pa.array(cls, type=pa.dictionary(pa.int8(), pa.string())),
    })
    pq.write_table(table, path)


def _time_filter_rows():
    """Five segments touching the box, window [T0, T0+600]:
    A inside, B 12 h away, C straddling the start, D straddling
    the end, E just before the window (inside a 300 s margin)."""
    def _ll(n):
        return (np.full(n, TRUTH_LAT + 1500.0 / 111320.0),
                TRUTH_LON + np.linspace(-0.05, 0.05, n))
    lat, lon = _ll(20)
    lat31, lon31 = _ll(31)
    return [
        ("AAAAAA", 0, "prop", T0 + 100 + 5 * np.arange(20), lat, lon, 1000.0),
        ("BBBBBB", 0, "prop", T0 + 43200 + 5 * np.arange(20), lat, lon, 1000.0),
        ("CCCCCC", 0, "prop", T0 - 50 + 5 * np.arange(20), lat, lon, 1000.0),
        ("DDDDDD", 0, "prop", T0 + 550 + 5 * np.arange(20), lat, lon, 1000.0),
        ("EEEEEE", 0, "prop", T0 - 400 + 5 * np.arange(31), lat31, lon31, 1000.0),
    ]


def test_load_segments_time_filter(tmp_path):
    """Fix 2: the loader's time filter.  A segment touching the box
    at a different time of day is not loaded at all; a straddling
    segment is kept, trimmed to the window, with one bracketing
    point either side; the margin works; None means no filtering."""
    from geoloc.search import load_segments
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    path = tmp_path / "points.parquet"
    _write_points_parquet(path, _time_filter_rows())
    box = (TRUTH_LAT - 0.05, TRUTH_LAT + 0.05, TRUTH_LON - 0.1, TRUTH_LON + 0.1)
    stats = {}
    segs = load_segments(path, frame, *box, t_start=T0, t_end=T0 + 600,
                         time_margin_s=0.0, stats=stats)
    got = {s.icao for s in segs}
    assert got == {"AAAAAA", "CCCCCC", "DDDDDD"}, got
    assert stats["segments_dropped_by_time"] == 2, stats
    # Outside the window: B 20 + E 31 + C 10 + D 9 = 70 points, less
    # the 2 bracketing points kept (C T0-5, D T0+605).
    assert stats["points_time_filtered_out"] == 68, stats
    by_icao = {s.icao: s for s in segs}
    # C straddles the start: one bracketing point before the window.
    assert by_icao["CCCCCC"].t[0] == pytest.approx(T0 - 5)
    assert by_icao["CCCCCC"].t[-1] == pytest.approx(T0 + 45)
    # D straddles the end: one bracketing point after the window.
    assert by_icao["DDDDDD"].t[0] == pytest.approx(T0 + 550)
    assert by_icao["DDDDDD"].t[-1] == pytest.approx(T0 + 605)
    # Margin: E (T0-400..T0-250) is outside [T0, T0+600] but inside
    # a 300 s margin -- kept, trimmed to the window + bracketing.
    stats2 = {}
    segs2 = load_segments(path, frame, *box, t_start=T0, t_end=T0 + 600,
                          time_margin_s=300.0, stats=stats2)
    got2 = {s.icao for s in segs2}
    assert got2 == {"AAAAAA", "CCCCCC", "DDDDDD", "EEEEEE"}, got2
    e = {s.icao: s for s in segs2}["EEEEEE"]
    assert e.t[0] == pytest.approx(T0 - 305) and e.t[-1] == pytest.approx(T0 - 250)
    # None = no filtering: the whole day is loaded.
    stats3 = {}
    segs3 = load_segments(path, frame, *box, stats=stats3)
    assert {s.icao for s in segs3} == {"AAAAAA", "BBBBBB", "CCCCCC", "DDDDDD", "EEEEEE"}
    assert stats3["segments_dropped_by_time"] == 0
    assert stats3["points_time_filtered_out"] == 0


def test_trim_long_flight_to_box():
    """Fix 3: a 300 km straight flight crossing a 50 km box is
    trimmed to ~70 km of points (box + 2x rmax + bracketing)."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    tr = _straight_track([0.0, 1500.0, 1000.0], 90, 60.0, 1000.0, T0,
                         t_span=2500, dt=5.0)
    segs = segments_from_tracks([("LONG", 0, "prop", tr)])
    assert len(segs[0].t) == 1001  # 300 km at 60 m/s, 5 s samples
    stats = {}
    trimmed, st = trim_segments(segs, frame, *_G3B_BOX, rmax_m=10_000.0,
                                min_points=5, stats=stats)
    assert len(trimmed) == 1
    n = len(trimmed[0].t)
    # 50 km box + 2 x (10 km rmax + 3 km safety margin) = ~76 km,
    # at 300 m per sample -> ~255 points.
    assert 200 < n < 320, n
    xs = trimmed[0].enu[:, 0]
    # The bracketing points are the nearest points outside the
    # expanded box on each side (~+/-38.1 km).
    assert xs.min() == pytest.approx(-38100.0, abs=300.0)
    assert xs.max() == pytest.approx(38100.0, abs=300.0)
    assert st["points_trimmed_spatially"] > 500


def test_trim_preserves_cpa_and_truth():
    """Fix 3: CPA / truth recovery is unchanged when the target
    tracks are 300 km long and trimmed before the CPA table."""
    frame, segs, perc, grid = _pair_scene()
    # Rebuild each target as a 300 km straight track, same CPA.
    tr_a = _straight_track([0.0, 1500.0, 1000.0], 90, 60.0, 1000.0, T0 + 250,
                           t_span=2500, dt=5.0)
    tr_b = _straight_track([2000.0, 0.0, 1200.0], 0, 55.0, 1200.0, T0 + 850,
                           t_span=2500, dt=5.0)
    long_segs = segments_from_tracks([("A", 0, "prop", tr_a),
                                      ("B", 1, "prop", tr_b)])
    box = (TRUTH_LAT - 0.03, TRUTH_LAT + 0.03, TRUTH_LON - 0.03, TRUTH_LON + 0.03)
    trimmed, _st = trim_segments(long_segs, frame, *box, rmax_m=10_000.0,
                                 min_points=5)
    assert all(len(s.t) < 200 for s in trimmed)  # was 1001 points each
    cpa = build_cpa_table(grid, trimmed, perc, SearchConfig())
    hyps = pair_gate(cpa, perc, SearchConfig())
    assert hyps, "pair gate found nothing after trim"
    h = min(hyps, key=lambda x: x.dt_err)
    assert h.dt_err < 2.0
    assert abs(h.t0 - T0) < 3.0
    assert set(h.seg_idx) == {0, 1}


def _out_and_back_track():
    """One aircraft, two passes (check 3): eastbound at y=1500 m
    from x=-40 km, U-turn (r=2 km) at x=+40 km, westbound at
    y=5500 m back to x=-40 km.  Speed 60 m/s, 5 s samples.
    Eastbound CPA (0, 1500) at T0+250; westbound CPA (0, 5500)
    at T0+1688."""
    speed = 60.0
    # Route waypoints (x, y) in metres, sampled by arc length.
    pts = [(-40000.0, 1500.0), (40000.0, 1500.0)]
    th = np.linspace(-np.pi / 2, np.pi / 2, 25)
    pts += [(40000.0 + 2000.0 * np.cos(a), 3500.0 + 2000.0 * np.sin(a))
            for a in th[1:]]
    pts += [(40000.0, 5500.0), (-40000.0, 5500.0)]
    pts = np.array(pts, dtype=float)
    seg_len = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s_cum = np.r_[0.0, np.cumsum(seg_len)]
    total = s_cum[-1]
    s = np.arange(0.0, total + 1.0, speed * 5.0)
    x = np.interp(s, s_cum, pts[:, 0])
    y = np.interp(s, s_cum, pts[:, 1])
    t0 = T0 + 250.0 - 40000.0 / speed  # eastbound CPA at T0+250
    return Track(t0 + s / speed, np.column_stack([x, y, np.full(len(s), 1000.0)]))


def test_trim_splits_reentering_flight():
    """Fix 3: a flight that leaves the box and re-enters is split
    into separate pieces, not joined across the gap."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    tr = _out_and_back_track()
    segs = segments_from_tracks([("RT", 0, "prop", tr)])
    box = (TRUTH_LAT - 0.05, TRUTH_LAT + 0.05, TRUTH_LON - 0.06, TRUTH_LON + 0.06)
    stats = {}
    trimmed, st = trim_segments(segs, frame, *box, rmax_m=10_000.0,
                                min_points=5, stats=stats)
    assert len(trimmed) == 2, f"expected 2 pieces, got {len(trimmed)}"
    assert stats["segments_split"] == 1
    assert {s.icao for s in trimmed} == {"RT"}
    assert {s.seg for s in trimmed} == {0}
    assert {s.subseg for s in trimmed} == {0, 1}
    # The pieces' time ranges are disjoint (no invented track
    # across the U-turn outside the box).
    assert trimmed[0].t[-1] < trimmed[1].t[0]
    assert trimmed[1].t[0] - trimmed[0].t[-1] > 100.0


def test_trim_drops_runs_shorter_than_min_points():
    """Fix 3: a run shorter than min_points (after bracketing) is
    dropped and counted, not kept as a stub segment."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    # A line that only clips the corner of the expanded box: one
    # point inside, two bracketing points -> 3 < min_points=5.
    xs = np.arange(-40000.0, 40001.0, 5000.0)
    ys = xs + 30000.0
    t = T0 + np.arange(len(xs)) * 5.0
    pos = np.column_stack([xs, ys, np.full(len(xs), 1000.0)])
    segs = segments_from_tracks([("CLIP", 0, "prop", Track(t, pos))])
    box = (TRUTH_LAT - 0.05, TRUTH_LAT + 0.05, TRUTH_LON - 0.06, TRUTH_LON + 0.06)
    stats = {}
    trimmed, st = trim_segments(segs, frame, *box, rmax_m=10_000.0,
                                min_points=5, stats=stats)
    assert trimmed == []
    assert st["runs_dropped_too_short"] == 1, st


def test_trim_keeps_cpa_at_corridor_edge():
    """Check 5: a flight whose CPA lies just inside rmax (9.5 km)
    from a cell at the box edge is still found after trimming --
    including a reporting gap straddling the trim boundary, where
    the bracketing point is the nearest point OUTSIDE the expanded
    box, across the gap."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    box = (TRUTH_LAT - 0.045, TRUTH_LAT + 0.045, TRUTH_LON - 0.055, TRUTH_LON + 0.055)
    # North-south flight at x=+13.566 km, y from -40 to +40 km, 60 m/s.
    # The easternmost cell column is at x ~ +4.066 km, so the CPA to
    # it is 9.5 km -- just inside rmax.
    y = np.arange(-40000.0, 40001.0, 300.0)
    t = T0 + 250.0 + (y - 4500.0) / 60.0  # y=+4.5 km at T0+250
    # Reporting gap straddling the trim boundary y=-18 km.
    keep = ~((y >= -19500.0) & (y <= -16500.0))
    y, t = y[keep], t[keep]
    pos = np.column_stack([np.full(len(y), 13566.0), y, np.full(len(y), 1000.0)])
    segs = segments_from_tracks([("EDGE", 0, "prop", Track(t, pos))])
    trimmed, _st = trim_segments(segs, frame, *box, rmax_m=10_000.0,
                                 min_points=5)
    assert len(trimmed) == 1
    # Bracketing point is the nearest point outside the expanded
    # box on each side -- across the gap on the south side.
    assert trimmed[0].enu[:, 1].min() == pytest.approx(-19600.0)
    assert trimmed[0].enu[:, 1].max() == pytest.approx(18200.0)
    # CPA to the cell nearest (4.5, 4.5) km: d ~9.5 km, t_cpa T0+250.
    grid = make_grid(*box, 1000.0, frame, height_fn=0.0)
    d_cpa = 9500.0
    ev = _event(250.0 + d_cpa / 343.0, d_cpa, 60.0, 90.0)
    perc = _perc([ev])
    cpa = build_cpa_table(grid, trimmed, perc, SearchConfig())
    cell = int(np.argmin(np.hypot(grid.enu[:, 0] - 4500.0, grid.enu[:, 1] - 4500.0)))
    a, b = cpa.entries_for_cell(cell)
    assert b - a >= 1, "CPA at the corridor edge was lost by trimming"
    i = a + int(np.argmin(np.abs(cpa.d[a:b] - d_cpa)))
    assert abs(cpa.d[i] - d_cpa) < 300.0, cpa.d[i]
    assert abs(cpa.t_cpa[i] - (T0 + 250.0)) < 5.0


def test_cpa_beyond_rmax_excluded_by_gate_not_trim():
    """Check 5: a flight 11 km from every cell is KEPT by the trim
    (inside box + rmax + margin) but correctly EXCLUDED by the CPA
    gate -- the exclusion is the gate's job, not the trim's."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    box = (TRUTH_LAT - 0.045, TRUTH_LAT + 0.045, TRUTH_LON - 0.055, TRUTH_LON + 0.055)
    y = np.arange(-40000.0, 40001.0, 300.0)
    t = T0 + 250.0 + (y - 4500.0) / 60.0
    pos = np.column_stack([np.full(len(y), 15500.0), y, np.full(len(y), 1000.0)])
    segs = segments_from_tracks([("FAR", 0, "prop", Track(t, pos))])
    trimmed, _st = trim_segments(segs, frame, *box, rmax_m=10_000.0,
                                 min_points=5)
    assert len(trimmed) == 1  # trim keeps it: 15.5 km < 5 + 13 km
    grid = make_grid(*box, 1000.0, frame, height_fn=0.0)
    ev = _event(250.0 + 11000.0 / 343.0, 11000.0, 60.0, 90.0)
    perc = _perc([ev])
    cpa = build_cpa_table(grid, trimmed, perc, SearchConfig())
    assert int(cpa.cell_ptr[-1]) == 0, "11 km CPA should have no table entries"


def test_out_and_back_two_pieces_two_events():
    """Check 3: one aircraft, two separate passes, both heard as
    two events -- both pieces must be usable, one per event, and
    the pair must be found.  The report must tell the pieces
    apart: (icao, seg, piece)."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    box = (TRUTH_LAT - 0.05, TRUTH_LAT + 0.05, TRUTH_LON - 0.06, TRUTH_LON + 0.06)
    tr = _out_and_back_track()
    segs = segments_from_tracks([("RT", 0, "prop", tr)])
    trimmed, _st = trim_segments(segs, frame, *box, rmax_m=10_000.0,
                                 min_points=5)
    assert len(trimmed) == 2
    grid = make_grid(*box, 500.0, frame, height_fn=0.0)
    # Events from the CPA table at the truth cell, one per piece.
    cpa_probe = build_cpa_table(grid, trimmed, _perc([]), SearchConfig())
    cell = int(np.argmin(np.hypot(grid.enu[:, 0], grid.enu[:, 1])))
    a, b = cpa_probe.entries_for_cell(cell)
    assert b - a >= 2
    by_seg = {}
    for i in range(a, b):
        by_seg.setdefault(int(cpa_probe.seg[i]), i)
    assert set(by_seg) == {0, 1}, by_seg
    events = []
    for piece in (0, 1):
        i = by_seg[piece]
        events.append(_event(float(cpa_probe.t_arr[i] - T0),
                             float(cpa_probe.d[i]), float(cpa_probe.v[i]),
                             90.0 if piece == 0 else 110.0))
    perc = _perc(events)
    cpa = build_cpa_table(grid, trimmed, perc, SearchConfig())
    hyps = pair_gate(cpa, perc, SearchConfig(), t0_known=T0)
    assert hyps, "pair gate found nothing for the two-piece aircraft"
    h = min(hyps, key=lambda x: x.dt_err)
    assert set(h.seg_idx) == {0, 1}, h.seg_idx
    assert h.dt_err < 2.0 and abs(h.t0 - T0) < 3.0
    # The two pieces are the same aircraft, different pieces.
    assert trimmed[0].icao == trimmed[1].icao == "RT"
    assert trimmed[0].seg == trimmed[1].seg == 0
    assert {trimmed[0].subseg, trimmed[1].subseg} == {0, 1}
    # And the report tells them apart.
    result = run_search(perc, trimmed, box, height_fn=0.0, t0_known=T0,
                        config=SearchConfig(grid_spacing_m=500.0), frame=frame)
    rep_segs = result.report["segments"]
    assert len(rep_segs) == 2
    assert {(s["icao"], s["seg"], s["piece"]) for s in rep_segs} == \
        {("RT", 0, 0), ("RT", 0, 1)}


def test_time_filter_recorded_in_report():
    """Check 2: the time window / margin is recorded in the report,
    with the assumption stated.  If only the date is known, the
    report must say so."""
    _frame, segs, perc, _grid = _pair_scene()
    box = (TRUTH_LAT - 0.03, TRUTH_LAT + 0.03, TRUTH_LON - 0.03, TRUTH_LON + 0.03)
    result = run_search(perc, segs, box, height_fn=0.0, t0_known=T0,
                        config=SearchConfig(grid_spacing_m=1000.0),
                        t_start=T0 - 600, t_end=T0 + 2520, time_margin_s=60.0)
    tf = result.report["time_filter"]
    assert tf["t_start"] == pytest.approx(T0 - 600)
    assert tf["t_end"] == pytest.approx(T0 + 2520)
    assert tf["margin_s"] == pytest.approx(60.0)
    assert "60" in tf["assumption"] and "whole day" in tf["assumption"]
    result2 = run_search(perc, segs, box, height_fn=0.0, t0_known=T0,
                         config=SearchConfig(grid_spacing_m=1000.0))
    tf2 = result2.report["time_filter"]
    assert tf2["t_start"] is None and tf2["t_end"] is None
    assert "whole day" in tf2["assumption"]


def test_timing_breakdown_present_and_consistent():
    """Check 6: run_search reports per-stage times, and they sum to
    roughly the total.  Silence time is a SUBSET of the scoring /
    posterior / refine stages, so it is not part of the sum."""
    _frame, segs, perc, _grid = _pair_scene()
    box = (TRUTH_LAT - 0.03, TRUTH_LAT + 0.03, TRUTH_LON - 0.03, TRUTH_LON + 0.03)
    result = run_search(perc, segs, box, height_fn=0.0, t0_known=T0,
                        config=SearchConfig(grid_spacing_m=1000.0))
    fc = result.funnel_counts
    for key in ("time_cpa_table_s", "time_pair_gate_s", "time_scoring_s",
                "time_refine_s", "time_marginalised_sweep_s",
                "time_silence_s", "time_total_s"):
        assert key in fc, key
        assert fc[key] >= 0.0, (key, fc[key])
    stages = sum(fc[k] for k in ("time_cpa_table_s", "time_pair_gate_s",
                                 "time_scoring_s", "time_refine_s",
                                 "time_marginalised_sweep_s"))
    assert stages <= fc["time_total_s"] + 0.05, (stages, fc["time_total_s"])
    assert fc["time_total_s"] - stages < max(2.0, 0.5 * fc["time_total_s"])


def test_g3b_like_311_segments_50km_box(tmp_path):
    """Check 1: the synthetic G3b case at Hudson-Valley scale -- a
    real rendered scene (targets A/B + distractors C/D/E) plus ~300
    realistic day-spread segments, time filter + trim applied, 50 km
    box at 200 m.  The guard must pass and the truth must still be
    recovered.  Slow (~1 min: render + search); the 3,274-segment
    case below covers the New York scale without rendering."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    scene = build_scene(frame, dt_between=920.0, duration=1600.0, seed=1)
    y, _truth = render_scene(scene["sources"], np.array([0.0, 0.0, 0.0]),
                             scene["t0"], scene["duration"], fs=4000, seed=1)
    perc = audio_mod.detect(y, fs=4000)
    assert len(perc.events) >= 2
    distractors = _g3b_window_filtered(_g3b_segments(300, 0))
    segs = scene["segments"] + distractors
    cfg = SearchConfig(grid_spacing_m=200.0,
                       refine_spacings_m=(100.0, 25.0, 5.0),
                       refine_window_half_m=(500.0, 150.0, 50.0))
    result = run_search(perc, segs, _G3B_BOX, height_fn=0.0, t0_known=T0,
                        config=cfg, t_start=T0 - 600, t_end=T0 + 2520,
                        time_margin_s=60.0, frame=frame)
    fc = result.funnel_counts
    assert fc["cpa_entries_estimated"] < 5_000_000, fc["cpa_entries_estimated"]
    assert fc["cpa_entries_estimated"] < fc["cpa_entries_estimated_product"]
    assert result.report["time_filter"]["margin_s"] == pytest.approx(60.0)
    assert len(result.scored) >= 1
    lat_m, lon_m, _p = result.peak(which="marg")
    d_m = float(np.hypot(*frame.to_enu(np.array([lat_m]), np.array([lon_m]),
                                       np.array([0.0]))[0, :2]))
    assert d_m < 1000.0, f"peak error {d_m:.0f} m"
    assert result.truth_in_region(TRUTH_LAT, TRUTH_LON, level=0.90, which="marg")


def test_g3b_like_3274_segments_guard_passes():
    """Check 1: the synthetic G3b case at New York scale -- 390
    props + 2,884 jets over the whole day, time filter + trim
    applied, 50 km box at 200 m, real scene targets A/B as events.
    The guard must pass (the corridor estimate, not the product)
    and run_search must complete in reasonable time.  Events are
    constructed from the scene geometry (not a rendered spectrogram),
    so this asserts the guard + table + pair gate, not truth
    recovery -- that is test_g3b_like_311_segments_50km_box."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    scene = build_scene(frame, dt_between=920.0, duration=1600.0, seed=1)
    distractors = _g3b_window_filtered(_g3b_segments(390, 2884))
    segs = scene["segments"] + distractors
    # Events matching the scene targets A (f0 90, CPA 1500 m N) and
    # B (f0 110, CPA 2000 m E), as audio.detect() would report them.
    ev_a = _event(255.26, 1803.0, 60.0, 90.0)
    ev_b = _event(1176.80, 2332.0, 55.0, 110.0)
    perc = _perc([ev_a, ev_b], spec=_fake_spec(t_max=1600.0))
    cfg = SearchConfig(grid_spacing_m=200.0)
    result = run_search(perc, segs, _G3B_BOX, height_fn=0.0, t0_known=T0,
                        config=cfg, t_start=T0 - 600, t_end=T0 + 2520,
                        time_margin_s=60.0, frame=frame)
    fc = result.funnel_counts
    assert fc["cpa_entries_estimated"] < 5_000_000, fc["cpa_entries_estimated"]
    # Jet deferral (on by default): the early table has (almost) no
    # jets -- _g3b_segments jets fly at 3000-9000 m, so they are
    # deferred, not silence-only in the early estimate.
    assert fc["segments_deferred_jets"] > 0, fc
    assert fc["cpa_entries_estimated_deferred_jets"] > 0, fc
    assert fc["cpa_entries_deferred"] >= 0
    assert fc["time_total_s"] > 0.0
    assert fc["time_cpa_table_s"] < 30.0, fc["time_cpa_table_s"]


# --- v3 notes patch (2026-11-08, user notes 1-6) --------------------------

def test_corridor_estimate_unknown_bucket():
    """User note 3: unknown-class segments are their own bucket in
    the corridor estimate -- NOT silence-only -- because the search
    treats them as possible matches (weight 0.5).  The three
    buckets and the three segment counts must sum to the totals."""
    from geoloc.search import corridor_cpa_estimate

    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    grid = make_grid(TRUTH_LAT - 0.03, TRUTH_LAT + 0.03,
                     TRUTH_LON - 0.03, TRUTH_LON + 0.03,
                     1000.0, frame, height_fn=0.0)
    tr_p = _straight_track([0.0, 1500.0, 1000.0], 90, 60.0, 1000.0, T0 + 250)
    tr_j = _straight_track([0.0, -1500.0, 5000.0], 90, 200.0, 5000.0, T0 + 250)
    tr_u = _straight_track([2000.0, 0.0, 1200.0], 0, 55.0, 1200.0, T0 + 850)
    segs = segments_from_tracks([("P", 0, "prop", tr_p),
                                 ("J", 1, "jet", tr_j),
                                 ("U", 2, "unknown", tr_u)])
    est, _, bd = corridor_cpa_estimate(segs, grid, 10_000.0)
    assert bd["n_candidate_segments"] == 1, bd
    assert bd["n_silence_only_segments"] == 1, bd  # the jet only
    assert bd["n_unknown_segments"] == 1, bd
    assert bd["n_candidate_segments"] + bd["n_silence_only_segments"] \
        + bd["n_unknown_segments"] == bd["n_segments"] == 3
    assert bd["candidate"] + bd["silence_only"] + bd["unknown"] == est
    assert bd["unknown"] > 0 and bd["silence_only"] > 0
    # And the unknown segment is still matchable: the pair gate
    # must recover the truth using it as the second event's flight.
    ev_a = _event(255.25, 1803.0, 60.0, 90.0)
    ev_b = _event(856.8, 2332.0, 55.0, 110.0)
    perc = _perc([ev_a, ev_b])
    cpa = build_cpa_table(grid, segs, perc, SearchConfig())
    hyps = pair_gate(cpa, perc, SearchConfig())
    assert hyps, "pair gate found nothing"
    h = min(hyps, key=lambda x: x.dt_err)
    assert set(h.seg_idx) == {0, 2}, h.seg_idx  # prop + unknown


def test_time_filter_no_filter_wording_and_margin_basis():
    """User note 5: with no time filter the report must NOT say
    "±0 s" -- it says "no time filter (whole day)" and margin_s
    is None.  And the margin must be described as the window
    margin, not automatically as start-time uncertainty: when
    the caller already folded the uncertainty into t_start/t_end,
    the report must say so."""
    _frame, segs, perc, _grid = _pair_scene()
    box = (TRUTH_LAT - 0.03, TRUTH_LAT + 0.03, TRUTH_LON - 0.03, TRUTH_LON + 0.03)
    cfg = SearchConfig(grid_spacing_m=1000.0)
    # No filter.
    result = run_search(perc, segs, box, height_fn=0.0, t0_known=T0, config=cfg)
    tf = result.report["time_filter"]
    assert tf["t_start"] is None and tf["t_end"] is None
    assert tf["margin_s"] is None
    assert tf["filtered"] is False
    assert "±0" not in tf["assumption"] and "+/-0" not in tf["assumption"]
    assert "No time filter" in tf["assumption"]
    assert "whole day" in tf["assumption"]
    # Filter, margin additional (default): margin IS the uncertainty.
    result2 = run_search(perc, segs, box, height_fn=0.0, t0_known=T0,
                         config=cfg, t_start=T0 - 600, t_end=T0 + 2520,
                         time_margin_s=60.0)
    tf2 = result2.report["time_filter"]
    assert tf2["margin_s"] == pytest.approx(60.0)
    assert tf2["filtered"] is True
    assert tf2["margin_already_in_window"] is False
    assert "±60" in tf2["assumption"]
    # Filter, uncertainty already folded into the window ends.
    result3 = run_search(perc, segs, box, height_fn=0.0, t0_known=T0,
                         config=cfg, t_start=T0 - 600, t_end=T0 + 2520,
                         time_margin_s=60.0, margin_already_in_window=True)
    tf3 = result3.report["time_filter"]
    assert tf3["margin_already_in_window"] is True
    assert "already includes" in tf3["assumption"]
    assert "NOT additional" in tf3["assumption"]


def test_spectrogram_value_nan_no_warning():
    """User note 6: audio.py:29 "invalid value in cast" -- a NaN
    modelled frequency (from an interpolated ratio outside the
    emission-time range, in search._score_event_matrix) reached
    Spectrogram.value, and np.rint(NaN).astype(int) warned.  NaN
    inputs must stay NaN in the output, with no warning."""
    spec = _fake_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        out = spec.value(np.array([10.0, 20.0, 30.0]),
                         np.array([100.0, np.nan, 200.0]))
        assert np.isnan(out[1])
        assert np.isfinite(out[0]) and np.isfinite(out[2])
        out2 = spec.value(np.array([np.nan, 10.0]), np.array([100.0, 100.0]))
        assert np.isnan(out2[0]) and np.isfinite(out2[1])
        # Scalar NaN, as well as out-of-range values, stay NaN.
        assert np.isnan(float(spec.value(10.0, float("nan"))))
        # Inf is non-finite too.
        out3 = spec.value(np.array([10.0, np.inf]), np.array([100.0, 100.0]))
        assert np.isnan(out3[1]) and np.isfinite(out3[0])
        # Finite inputs are unchanged (nearest bin, same as before).
        spec2 = _fake_spec()
        spec2.S[100, 40] = 7.5
        assert float(spec2.value(10.0, 130.0)) == 7.5
        # (c) The NaN guard must not depend on the grid settings:
        # a spectrogram with a different time step (0.5 s, not
        # 0.25 s) and frequency step (2 Hz, not 1 Hz) must still
        # return NaN for NaN inputs, with no warning, and finite
        # values from the right bin.
        t_g = np.arange(0.0, 200.0, 0.5)
        f_g = np.arange(30.0, 501.0, 2.0)
        S_g = np.zeros((len(f_g), len(t_g)))
        S_g[50, 20] = 3.25  # f = 130 Hz, t = 10 s
        spec_g = audio_mod.Spectrogram(t=t_g, f=f_g, S=S_g,
                                       loud=np.zeros(len(t_g)))
        out_g = spec_g.value(np.array([10.0, 20.0, 30.0]),
                             np.array([130.0, np.nan, 200.0]))
        assert float(out_g[0]) == 3.25
        assert np.isnan(out_g[1]) and np.isfinite(out_g[2])
        assert np.isnan(float(spec_g.value(float("nan"), 130.0)))


def _near_edge_rows():
    """Three segments for the expanded-hit test: A inside the box,
    B passing 5 km north of the box edge (never entering it), C
    passing 20 km north of the edge.  All eastbound, CPA T0+250."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)

    def _rows(icao, y_m):
        tr = _straight_track([0.0, y_m, 1000.0], 90, 60.0, 1000.0, T0 + 250)
        latlon = frame.to_llh(tr.position(tr.t))
        return (icao, 0, "jet", np.asarray(tr.t),
                np.asarray(latlon[0]), np.asarray(latlon[1]), 1000.0)

    box_half_lat_m = 0.05 * 111320.0
    return [
        _rows("AAAAAA", 0.0),
        _rows("BBBBBB", box_half_lat_m + 5000.0),
        _rows("CCCCCC", box_half_lat_m + 20000.0),
    ]


def test_loader_expanded_hit_near_edge_flight(tmp_path):
    """User note 4: the loader must pick up a flight passing 5 km
    outside the box edge (never entering it) -- it can be some
    cell's CPA and matters for the silence check -- while a flight
    20 km away must not be loaded."""
    from geoloc.search import load_segments

    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    path = tmp_path / "points.parquet"
    _write_points_parquet(path, _near_edge_rows())
    box = (TRUTH_LAT - 0.05, TRUTH_LAT + 0.05, TRUTH_LON - 0.1, TRUTH_LON + 0.1)
    stats = {}
    segs = load_segments(path, frame, *box, stats=stats)
    got = {s.icao for s in segs}
    assert got == {"AAAAAA", "BBBBBB"}, got
    assert stats["segments_added_by_expanded_hit"] == 1, stats
    # B must appear in the CPA table: a cell at the north edge of
    # the box is ~5 km from its track.
    grid = make_grid(*box, 1000.0, frame, height_fn=0.0)
    ev = _event(255.25, 5000.0, 60.0, 90.0)
    perc = _perc([ev])
    cpa = build_cpa_table(grid, segs, perc, SearchConfig())
    seg_ids = {int(s) for s in cpa.seg}
    b_idx = [i for i, s in enumerate(segs) if s.icao == "BBBBBB"][0]
    assert b_idx in seg_ids, seg_ids


def test_g3b_pick_window_script(tmp_path):
    """The pick-window helper: given a day's Parquet and a box, it
    prints small-plane segments per UTC hour.  Exercised here on
    a synthetic Parquet (2 props in hour 11, 1 prop + 1 jet in
    hour 12, nothing else)."""
    rows = []

    def _add(icao, cls, t0):
        lat = np.full(10, TRUTH_LAT)
        lon = TRUTH_LON + np.linspace(-0.05, 0.05, 10)
        rows.append((icao, 0, cls, t0 + 5 * np.arange(10), lat, lon, 1000.0))

    day_start = T0 - (T0 % 86400)
    _add("P1", "prop", day_start + 11 * 3600)
    _add("P2", "prop", day_start + 11 * 3600 + 1800)
    _add("P3", "prop", day_start + 12 * 3600)
    _add("J1", "jet", day_start + 12 * 3600)
    path = tmp_path / "points.parquet"
    _write_points_parquet(path, rows)
    script = Path(__file__).resolve().parents[1] / "scripts" / "g3b_pick_window.py"
    box = (TRUTH_LAT - 0.05, TRUTH_LAT + 0.05, TRUTH_LON - 0.1, TRUTH_LON + 0.1)
    day = datetime_from_unix(day_start)
    r = subprocess.run(
        [sys.executable, str(script), "--day", day, "--points", str(path),
         "--box", *[str(v) for v in box]],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    row11 = [ln for ln in lines if ln.split() and ln.split()[0] == "11"]
    row12 = [ln for ln in lines if ln.split() and ln.split()[0] == "12"]
    assert row11 and row12, lines
    # Columns: hour prop turbo rotor jet unk small total.
    assert row11[0].split() == ["11", "2", "0", "0", "0", "0", "2", "2"], row11
    assert row12[0].split() == ["12", "1", "0", "0", "1", "0", "1", "2"], row12
    assert "11:00" in r.stdout  # most small-plane traffic


def datetime_from_unix(ts):
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y-%m-%d")


# --- jet deferral (2026-11-08, user proposal) ---------------------------

def _jet_deferral_scene():
    """_pair_scene + one HIGH jet distractor: alt 3500 m, CPA 1 km
    north of truth at T0+450, so its sound arrives at ~461 s --
    inside the (400, 500) silence window, and loud enough to
    penalise (P(audible) ~0.3 at 3.6 km slant distance).  It must
    penalise the truth cell in the full run, and the deferred
    pass must reproduce exactly that penalty."""
    frame, segs, perc, grid = _pair_scene()
    tr_j = _straight_track([0.0, 1000.0, 3500.0], 90, 200.0, 3500.0, T0 + 450)
    segs = segs + segments_from_tracks([("J", 2, "jet", tr_j)])
    # Fake spec must cover the whole event span (event B spans to
    # ~956 s); the default 700 s fake spec scores nothing.
    perc = _perc(perc.events, silences=[(400.0, 500.0)],
                 spec=_fake_spec(t_max=1600.0))
    return frame, segs, perc, grid


def _modes_key(result):
    return [(round(m["lat"], 9), round(m["lon"], 9),
             round(m["mass"], 12),
             None if m["loglik_max"] is None else round(m["loglik_max"], 6),
             None if m["loglik_marg"] is None else round(m["loglik_marg"], 6),
             round(m["silence_loglik"], 6)
             if m["silence_loglik"] is not None else None)
            for m in result.modes]


def test_jet_deferral_identity_g3_scene():
    """Identity proof 1: on the G3 scene, deferral on vs off must
    give identical modes, pair hypotheses and refined peaks --
    no high jet is matched as an event there (TESTE is never in
    a pair hypothesis and its sound arrives outside every silence
    window, so its silence contribution is 0 either way)."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    scene = build_scene(frame, dt_between=920.0, duration=1600.0, seed=1)
    segs = scene["segments"]
    ev_a = _event(255.26, 1803.0, 60.0, 90.0)
    ev_b = _event(1176.80, 2332.0, 55.0, 110.0)
    perc = _perc([ev_a, ev_b], silences=[(400.0, 500.0)],
                 spec=_fake_spec(t_max=1600.0))
    box = _G3B_BOX
    cfg_on = SearchConfig(grid_spacing_m=200.0, defer_jets=True)
    cfg_off = SearchConfig(grid_spacing_m=200.0, defer_jets=False)
    on = run_search(perc, segs, box, height_fn=0.0, t0_known=T0,
                    config=cfg_on, t_start=T0 - 600, t_end=T0 + 2520,
                    time_margin_s=60.0, frame=frame)
    off = run_search(perc, segs, box, height_fn=0.0, t0_known=T0,
                     config=cfg_off, t_start=T0 - 600, t_end=T0 + 2520,
                     time_margin_s=60.0, frame=frame)
    assert _modes_key(on) == _modes_key(off), (_modes_key(on), _modes_key(off))
    assert on.funnel_counts["pair_hypotheses"] == off.funnel_counts["pair_hypotheses"]
    assert [(round(r.best_lat, 9), round(r.best_lon, 9))
            for r in on.refined] == [(round(r.best_lat, 9), round(r.best_lon, 9))
                                     for r in off.refined]
    assert on.funnel_counts["segments_deferred_jets"] == 1, on.funnel_counts
    assert on.funnel_counts["cpa_entries_estimated_deferred_jets"] > 0
    assert off.funnel_counts["segments_deferred_jets"] == 0


def test_jet_deferral_high_jet_silence_identical():
    """Identity proof 2: a high jet that IS in a silence window
    must penalise the truth cell by exactly the same amount with
    deferral on as off -- and the penalty must be real (the same
    search with the jet removed entirely gives silence 0)."""
    frame, segs, perc, grid = _jet_deferral_scene()
    box = (TRUTH_LAT - 0.03, TRUTH_LAT + 0.03, TRUTH_LON - 0.03, TRUTH_LON + 0.03)
    # Fake (zero) spectrogram: scores are 0 dB, so the quick-score
    # threshold must be lowered or nothing is scored at all.
    cfg_on = SearchConfig(grid_spacing_m=1000.0, defer_jets=True,
                          quick_score_min_db=-10.0)
    cfg_off = SearchConfig(grid_spacing_m=1000.0, defer_jets=False,
                           quick_score_min_db=-10.0)
    on = run_search(perc, segs, box, height_fn=0.0, t0_known=T0,
                    config=cfg_on, frame=frame)
    off = run_search(perc, segs, box, height_fn=0.0, t0_known=T0,
                     config=cfg_off, frame=frame)
    assert _modes_key(on) == _modes_key(off)
    # The penalty must be real, on the scored truth hypothesis
    # (prop A + prop B): the top mode may sit on an unpenalised
    # cell, so check the scored hypothesis, not modes[0].
    truth_scored = [s for s in on.scored
                    if set(s.hypothesis.seg_idx) == {0, 1}]
    assert truth_scored, "truth hypothesis was not scored"
    assert truth_scored[0].silence_loglik < -0.1, truth_scored[0]
    assert on.funnel_counts["cpa_entries_deferred"] > 0, on.funnel_counts
    assert on.funnel_counts["time_deferred_jet_silence_s"] >= 0.0
    # Jet removed entirely: no penalty at all.
    nojet = run_search(perc, segs[:2], box, height_fn=0.0, t0_known=T0,
                       config=cfg_on, frame=frame)
    truth_nojet = [s for s in nojet.scored
                   if set(s.hypothesis.seg_idx) == {0, 1}]
    assert truth_nojet and truth_nojet[0].silence_loglik == pytest.approx(0.0)
    assert nojet.funnel_counts["segments_deferred_jets"] == 0


def test_jet_deferral_low_jet_still_matched():
    """A LOW jet (alt 2000 m) is a genuine candidate and must stay
    early: it must still be matched as an event with deferral on."""
    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    tr_j = _straight_track([0.0, 1500.0, 2000.0], 90, 80.0, 2000.0, T0 + 250)
    tr_b = _straight_track([2000.0, 0.0, 1200.0], 0, 55.0, 1200.0, T0 + 850)
    segs = segments_from_tracks([("J", 0, "jet", tr_j), ("B", 1, "prop", tr_b)])
    ev_a = _event(257.3, 2500.0, 80.0, 90.0)  # jet CPA 1500 m N, alt 2000 m
    ev_b = _event(856.8, 2332.0, 55.0, 110.0)
    perc = _perc([ev_a, ev_b], spec=_fake_spec(t_max=1600.0))
    box = (TRUTH_LAT - 0.03, TRUTH_LAT + 0.03, TRUTH_LON - 0.03, TRUTH_LON + 0.03)
    result = run_search(perc, segs, box, height_fn=0.0, t0_known=T0,
                        config=SearchConfig(grid_spacing_m=1000.0,
                                            defer_jets=True,
                                            quick_score_min_db=-10.0),
                        frame=frame)
    assert result.funnel_counts["segments_deferred_jets"] == 0, \
        result.funnel_counts
    # The low jet must survive the pair gate AND scoring as an
    # event flight (seg_idx 0, with the prop as seg_idx 1).
    assert any(set(s.hypothesis.seg_idx) == {0, 1} for s in result.scored), \
        [s.hypothesis.seg_idx for s in result.scored]


def test_jet_deferral_ny_like_guard_and_silence():
    """New York-like case: ~390 props + 2,884 jets, 1 h window.
    With deferral the guard must pass, the early CPA table must
    be far smaller than the all-segments table (~100x in the
    user's real numbers: ~30k prop entries vs ~3.1M jet), and
    the deferred-jet silence must still be applied on surviving
    cells (cpa_entries_deferred > 0)."""
    from geoloc.search import build_cpa_table, split_early_deferred

    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    scene = build_scene(frame, dt_between=920.0, duration=1600.0, seed=1)
    distractors = _g3b_window_filtered(_g3b_segments(390, 2884))
    segs = scene["segments"] + distractors
    ev_a = _event(255.26, 1803.0, 60.0, 90.0)
    ev_b = _event(1176.80, 2332.0, 55.0, 110.0)
    perc = _perc([ev_a, ev_b], silences=[(400.0, 500.0)],
                 spec=_fake_spec(t_max=1600.0))
    cfg = SearchConfig(grid_spacing_m=200.0, defer_jets=True)
    result = run_search(perc, segs, _G3B_BOX, height_fn=0.0, t0_known=T0,
                        config=cfg, t_start=T0 - 600, t_end=T0 + 2520,
                        time_margin_s=60.0, frame=frame)
    fc = result.funnel_counts
    assert fc["cpa_entries_estimated"] < 5_000_000, fc
    assert fc["segments_deferred_jets"] > 50, fc
    assert fc["cpa_entries_deferred"] > 0, fc
    # Early vs all-segments CPA tables on the same grid.  The
    # saving here is ~5x, not the ~100x of the real day: the
    # synthetic props are long tracks crossing the whole box, so
    # each prop contributes ~25k entries, while the real 1 h
    # window had very few, short props (~30k entries total) vs
    # ~150 jets (~3.1M).  The ratio depends on the prop/jet mix;
    # what deferral guarantees is that the jet part is gone.
    grid = make_grid(*_G3B_BOX, 200.0, frame, height_fn=0.0)
    early, deferred = split_early_deferred(segs, cfg)
    cpa_all = build_cpa_table(grid, segs, perc, cfg)
    cpa_early = build_cpa_table(grid, early, perc, cfg)
    n_all = int(cpa_all.cell_ptr[-1])
    n_early = int(cpa_early.cell_ptr[-1])
    assert n_early * 3 < n_all, (n_early, n_all)
    assert fc["cpa_entries"] == n_early, (fc["cpa_entries"], n_early)


# --- Density-scaled scoring budget (2026-11-09, G3b New York) -----------

def test_effective_max_blobs_to_score():
    """Budget = min(cap, max(base, ceil(frac * n_blobs))).

    Sparse scenes reproduce today's fixed 20; dense scenes score up
    to all blobs, bounded by the cap.
    """
    cfg = SearchConfig()
    assert cfg.max_blobs_to_score == 20
    assert cfg.blobs_to_score_frac == 1.0
    assert cfg.max_blobs_to_score_cap == 100
    assert effective_max_blobs_to_score(0, cfg) == 20
    assert effective_max_blobs_to_score(6, cfg) == 20   # G3a: unchanged
    assert effective_max_blobs_to_score(20, cfg) == 20
    assert effective_max_blobs_to_score(21, cfg) == 21
    assert effective_max_blobs_to_score(46, cfg) == 46  # NY run_01, gate 2.0
    assert effective_max_blobs_to_score(103, cfg) == 100  # NY run_04: capped
    # A smaller fraction scales more slowly, and the cap binds first.
    half = SearchConfig(blobs_to_score_frac=0.5)
    assert effective_max_blobs_to_score(46, half) == 23
    assert effective_max_blobs_to_score(103, half) == 52
    small_cap = SearchConfig(max_blobs_to_score_cap=30)
    assert effective_max_blobs_to_score(103, small_cap) == 30
    # A larger base is a floor, not an addition.
    big_base = SearchConfig(max_blobs_to_score=50)
    assert effective_max_blobs_to_score(6, big_base) == 50
    assert effective_max_blobs_to_score(103, big_base) == 100


def test_run_search_records_effective_blob_budget(tmp_path):
    """run_search honours the effective budget and records it.

    On the small G3 scene (6 blobs) the default effective budget is
    20, so nothing is truncated.  Forcing base=cap=1 truncates to
    one blob and the counts say so.
    """
    centre_lat = TRUTH_LAT + 6000.0 / 111320.0
    centre_lon = TRUTH_LON + 8000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT)))
    frame = LocalFrame(centre_lat, centre_lon, 0.0)
    scene = build_scene(frame, dt_between=600.0, duration=1100.0, seed=1)
    y, truth = render_scene(scene["sources"], np.array([0.0, 0.0, 0.0]),
                            scene["t0"], scene["duration"], fs=4000, seed=1)
    perc = audio_mod.detect(y, fs=4000)
    box = (centre_lat - 10000.0 / 111320.0, centre_lat + 10000.0 / 111320.0,
           centre_lon - 10000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT))),
           centre_lon + 10000.0 / (111320.0 * math.cos(math.radians(TRUTH_LAT))))
    cfg = SearchConfig(grid_spacing_m=500.0, refine_spacings_m=(100.0, 25.0, 5.0),
                       refine_window_half_m=(500.0, 150.0, 50.0))
    result = run_search(perc, scene["segments"], box, height_fn=0.0,
                        config=cfg, out_dir=tmp_path / "g3", frame=frame)
    fc = result.funnel_counts
    assert fc["max_blobs_to_score_effective"] == 20, fc
    assert fc["blobs_truncated_by_max_blobs_to_score"] == 0, fc
    assert fc["blobs"] <= 20, fc
    # Forced tiny budget: exactly one blob scored, the rest recorded.
    tiny = SearchConfig(grid_spacing_m=500.0,
                        refine_spacings_m=(100.0, 25.0, 5.0),
                        refine_window_half_m=(500.0, 150.0, 50.0),
                        max_blobs_to_score=1, max_blobs_to_score_cap=1)
    result2 = run_search(perc, scene["segments"], box, height_fn=0.0,
                         config=tiny, out_dir=tmp_path / "g3tiny", frame=frame)
    fc2 = result2.funnel_counts
    assert fc2["max_blobs_to_score_effective"] == 1, fc2
    assert fc2["blobs_truncated_by_max_blobs_to_score"] == fc2["blobs"] - 1, fc2
    assert len(result2.scored) <= 1, fc2
