"""Limit-calibration benchmark (G4 preparation, 2026-11-08).

Times the FULL run_search at large CPA-table sizes, to calibrate the
5M-entry scale guard -- NOT to raise it.  The user measured the table
directly: one CPA entry is 48 bytes, so the 5M cap is only ~240 MB
(Hudson whole day 4.8M entries / 233 MB / 2.5 s to build; New York
1 h 2.2M / 105 MB / 1.1 s; his machine has 64 GB).  Before raising
the cap, the full search must be timed at that size, because the
later stages (pair gate, scoring, marginalised sweep) may slow
down faster than the table grows.

Method: the real G3 perception (render + audio.detect, 1600 s,
dt 920 s) plus N synthetic distractor segments in the recording
window, in a 50 km box at 200 m (62,500 cells), with jet deferral
OFF by default so the table actually holds the jets -- the cap
question is about table size, and the later stages see the same
hypotheses either way.  Distractors are the realistic generator
from tests/test_search.py (turning tracks, 3 s reporting, ~10%
dropped points, pattern work, out-and-back), with start times
spread over the recording window [T0-600, T0+1600] so they behave
like real in-window traffic.  Mix is ~12% prop / ~88% jet, like
the user's New York box (390 prop + 2,884 jet of 3,274).

Traffic density at the top end is NOT realistic: ~3,500 segments
in one 50 km box in one hour is ~20x New York.  The point is the
table size and how each stage scales with it, not the traffic
level; the per-cell density is what a busy CONUS tile would see.
A 50M table in real life comes from a whole day over a bigger
area (or a bigger tile), not from 3,500 planes in one hour in
Hudson Valley.

Usage (from the project root):
    python scripts/bench_large.py --n-prop 40 --n-jet 310 --out results/bench_5m.json
    python scripts/bench_large.py --n-prop 170 --n-jet 1230 --out results/bench_20m.json
    python scripts/bench_large.py --n-prop 420 --n-jet 3080 --out results/bench_50m.json
    python scripts/bench_large.py --n-prop 40 --n-jet 310 --only-cpa   # table only
    python scripts/bench_large.py --n-prop 40 --n-jet 310 --defer-jets # deferral on

On Windows the same, with backslashes, in the geoloc conda env.
Peak memory is the process peak RSS (resource.getrusage, Linux /
Windows via resource if available); table bytes are entries x 48.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoloc import audio as audio_mod
from geoloc.geodesy import LocalFrame
from geoloc.physics import Track
from geoloc.search import (SearchConfig, build_cpa_table, corridor_cpa_estimate,
                           make_grid, run_search, segments_from_tracks)
from geoloc.synth import render_scene
from geoloc.synthetic_scene import TRUTH_LAT, TRUTH_LON, T0, build_scene

BYTES_PER_CPA_ENTRY = 48  # measured by the user, 2026-11-08
BOX_HALF_M = 25_000.0


def _realistic_track(rng, kind, t0):
    """One realistic synthetic track in ENU metres (frame origin = truth).
    Copied from tests/test_search.py so the benchmark is self-contained."""
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


def _distractors(n_prop, n_jet, seed):
    """In-window distractors: start times spread over [T0-600, T0+1600]."""
    rng = np.random.default_rng(seed)
    items = []
    kinds = ["prop"] * n_prop + ["jet"] * n_jet
    for i, kind in enumerate(kinds):
        k = kind
        if kind == "prop" and i % 20 == 0:
            k = "pattern"
        elif kind == "prop" and i % 33 == 0:
            k = "outback"
        t0 = T0 + rng.uniform(-600, 1600)
        items.append((f"AC{i:04d}", i % 97,
                      kind if kind != "pattern" else "prop",
                      _realistic_track(rng, k, t0)))
    return segments_from_tracks(items)


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-prop", type=int, default=40)
    ap.add_argument("--n-jet", type=int, default=310)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grid", type=float, default=200.0)
    ap.add_argument("--defer-jets", action="store_true",
                    help="run with jet deferral ON (default is OFF, so the "
                         "table actually holds the jets)")
    ap.add_argument("--only-cpa", action="store_true",
                    help="build the CPA table only, skip the full search")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    box = (TRUTH_LAT - BOX_HALF_M / 111320.0, TRUTH_LAT + BOX_HALF_M / 111320.0,
           TRUTH_LON - BOX_HALF_M / (111320.0 * math.cos(math.radians(TRUTH_LAT))),
           TRUTH_LON + BOX_HALF_M / (111320.0 * math.cos(math.radians(TRUTH_LAT))))
    cfg = SearchConfig(grid_spacing_m=args.grid, allow_large_box=True)
    cfg.defer_jets = args.defer_jets

    # --- Perception: real G3 render + detect --------------------------
    scene = build_scene(frame, dt_between=920.0, duration=1600.0, seed=1)
    t0 = time.time()
    y, _truth = render_scene(scene["sources"], np.array([0.0, 0.0, 0.0]),
                             scene["t0"], 1600.0, fs=4000, seed=1)
    render_s = time.time() - t0
    t0 = time.time()
    perc = audio_mod.detect(y, fs=4000)
    detect_s = time.time() - t0
    print(f"perception: {len(perc.events)} events, render {render_s:.1f} s, "
          f"detect {detect_s:.1f} s", flush=True)

    # --- Segments: G3 targets + distractors ---------------------------
    segs = list(scene["segments"]) + _distractors(args.n_prop, args.n_jet, args.seed)
    print(f"segments: {len(segs)} total ({len(scene['segments'])} G3 + "
          f"{args.n_prop} prop + {args.n_jet} jet distractors), "
          f"defer_jets={cfg.defer_jets}", flush=True)
    grid = make_grid(*box, args.grid, frame, height_fn=0.0)
    est, _per_seg, bd = corridor_cpa_estimate(segs, grid, cfg.rmax_m)
    print(f"corridor estimate: {est:,} entries "
          f"(candidate {bd['candidate']:,}, silence-only {bd['silence_only']:,}, "
          f"unknown {bd['unknown']:,})", flush=True)

    rss0 = _rss_mb()
    result = {"n_prop": args.n_prop, "n_jet": args.n_jet, "seed": args.seed,
              "grid": args.grid, "defer_jets": cfg.defer_jets,
              "n_segments": len(segs), "n_cells": len(grid.enu),
              "corridor_estimate": int(est), "rss_before_mb": round(rss0, 1),
              "render_s": round(render_s, 1), "detect_s": round(detect_s, 1)}

    if args.only_cpa:
        t0 = time.time()
        cpa = build_cpa_table(grid, segs, perc, cfg)
        result["cpa_entries"] = int(cpa.cell_ptr[-1])
        result["cpa_table_s"] = round(time.time() - t0, 1)
        result["cpa_table_mb"] = round(result["cpa_entries"] * BYTES_PER_CPA_ENTRY / 1e6, 1)
        result["peak_rss_mb"] = round(_rss_mb(), 1)
        print(f"CPA table only: {result['cpa_entries']:,} entries in "
              f"{result['cpa_table_s']} s, {result['cpa_table_mb']} MB, "
              f"peak RSS {result['peak_rss_mb']} MB", flush=True)
    else:
        t0 = time.time()
        res = run_search(perc, segs, box, height_fn=0.0, config=cfg, frame=frame)
        result["run_search_s"] = round(time.time() - t0, 1)
        result["funnel_counts"] = res.funnel_counts
        result["cpa_entries"] = int(res.funnel_counts["cpa_entries"])
        result["cpa_table_mb"] = round(result["cpa_entries"] * BYTES_PER_CPA_ENTRY / 1e6, 1)
        result["peak_rss_mb"] = round(_rss_mb(), 1)
        lat_m, lon_m, _p = res.peak(which="marg")
        err = float(np.hypot(*frame.to_enu(np.array([lat_m]), np.array([lon_m]),
                                           np.array([0.0]))[0, :2]))
        result["peak_error_m"] = round(err, 1)
        result["truth_in_90"] = bool(res.truth_in_region(TRUTH_LAT, TRUTH_LON,
                                                         level=0.90, which="marg"))
        # Diagnostics for the truth-recovery question: top modes, and
        # whether the true flight pair (G3 scene segments 0 and 1,
        # TESTA/TESTB) was among the scored blobs at all.
        result["top_modes"] = [
            {"lat": round(float(m["lat"]), 5), "lon": round(float(m["lon"]), 5),
             "mass": round(float(m["mass"]), 4),
             "flights": [f["icao"] for f in m.get("flights", [])]}
            for m in res.modes[:3]]
        scored_keys = [(tuple(s.hypothesis.event_idx), tuple(s.hypothesis.seg_idx))
                       for s in res.scored]
        result["n_scored_blobs"] = len(scored_keys)
        result["truth_pair_scored"] = bool(
            any(0 in k[1] and 1 in k[1] for k in scored_keys))
        fc = res.funnel_counts
        print(f"run_search: {result['run_search_s']} s; "
              f"CPA {fc['cpa_entries']:,} entries ({result['cpa_table_mb']} MB) "
              f"in {fc.get('time_cpa_table_s')} s; pair gate "
              f"{fc.get('time_pair_gate_s')} s ({fc.get('pair_hypotheses')} hyps, "
              f"{fc.get('blobs')} blobs); scoring {fc.get('time_scoring_s')} s; "
              f"refine {fc.get('time_refine_s')} s; marginalised sweep "
              f"{fc.get('time_marginalised_sweep_s')} s "
              f"({fc.get('posterior_cells_scored')} cells); "
              f"peak error {err:.0f} m, truth in 90%={result['truth_in_90']}; "
              f"peak RSS {result['peak_rss_mb']} MB", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
