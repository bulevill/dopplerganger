"""G3 driver (gate G3): synthetic end-to-end test.

Picks a real flight from the converted day if ``points.parquet`` exists,
otherwise generates synthetic ADS-B-like tracks in-memory (two target
props with non-parallel tracks, plus distractors, see
``geoloc.synthetic_scene``).  Renders fake audio with
``synth.render_scene()``, runs ``audio.detect()`` for real, runs
``search.run_search()``, and reports the error against the known truth.

Usage (from the project root, in the ``geoloc`` conda env):
    python scripts/g3_synthetic.py [--day 2026-11-06] [--box ...]
        [--truth LAT LON] [--out results/g3] [--duration 1600]
        [--dt-between 920] [--grid 200] [--seed 1]

On Windows the same, with backslashes.  The in-memory scene is the
default: no D: drive or real Parquet is needed to run G3 here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoloc import audio as audio_mod
from geoloc.geodesy import LocalFrame
from geoloc.paths import PROCESSED
from geoloc.search import SearchConfig, run_search
from geoloc.synth import render_scene
from geoloc.synthetic_scene import TRUTH_LAT, TRUTH_LON, build_scene


def _error_m(lat, lon, truth_lat, truth_lon):
    frame = LocalFrame(truth_lat, truth_lon, 0.0)
    enu = frame.to_enu(np.array([lat]), np.array([lon]), np.array([0.0]))[0]
    return float(np.hypot(enu[0], enu[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default=None,
                    help="ADS-B day (YYYY-MM-DD) to load points.parquet for; "
                         "if the file does not exist, the in-memory scene is used")
    ap.add_argument("--truth", type=float, nargs=2, default=[TRUTH_LAT, TRUTH_LON])
    ap.add_argument("--box-offset", type=float, nargs=2, default=[8000.0, 6000.0],
                    help="search-box centre offset from truth, east/north metres")
    ap.add_argument("--box-half", type=float, default=25000.0,
                    help="search box half-size in metres (default 25 km -> 50 km box)")
    ap.add_argument("--duration", type=float, default=1600.0)
    ap.add_argument("--dt-between", type=float, default=920.0)
    ap.add_argument("--grid", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/g3")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip F9 refinement (faster smoke run)")
    ap.add_argument("--no-defer-jets", action="store_true",
                    help="run with jet deferral OFF (defer_jets=False) "
                         "-- old behaviour, for the identity proof")
    args = ap.parse_args(argv)

    truth_lat, truth_lon = args.truth
    t_start = time.time()
    # --- Scene -----------------------------------------------------------
    # Search frame: box centre offset from truth, so truth is off-centre.
    centre_lat = truth_lat + args.box_offset[1] / 111320.0
    centre_lon = truth_lon + args.box_offset[0] / (111320.0 * math.cos(math.radians(truth_lat)))
    frame = LocalFrame(centre_lat, centre_lon, 0.0)
    points_path = None
    if args.day:
        cand = PROCESSED / "adsblol" / args.day / "points.parquet"
        if cand.exists():
            points_path = cand
            print(f"Real Parquet found: {cand} (G3-from-real-day is not yet "
                  "implemented in this driver -- using the in-memory scene "
                  "for the tracks; see NOTES in the report)")
    scene = build_scene(frame, dt_between=args.dt_between,
                        duration=args.duration, seed=args.seed)
    print(f"Scene: truth {truth_lat}, {truth_lon}; T0 {scene['t0']}; "
          f"duration {args.duration:.0f} s; dt_between {args.dt_between:.0f} s; "
          f"segments {[s.icao for s in scene['segments']]}")
    # --- Render + detect (M1, for real -- no bypass) ----------------------
    t0 = time.time()
    y, truth = render_scene(scene["sources"], np.array([0.0, 0.0, 0.0]),
                            scene["t0"], args.duration, fs=4000, seed=args.seed)
    print(f"render_scene: {time.time() - t0:.1f} s; truth={truth}")
    t0 = time.time()
    perc = audio_mod.detect(y, fs=4000)
    print(f"audio.detect: {time.time() - t0:.1f} s; {len(perc.events)} events, "
          f"silences={perc.silences}")
    for e in perc.events:
        print(f"  event t_c={e.t_c:.2f} d={e.d:.0f} v={e.v:.1f} f0={e.f0:.2f} "
              f"harm={e.harmonics} span=({e.span[0]:.0f},{e.span[1]:.0f})")
    # --- Search (M3-M6) ----------------------------------------------------
    half = args.box_half
    box = (centre_lat - half / 111320.0, centre_lat + half / 111320.0,
           centre_lon - half / (111320.0 * math.cos(math.radians(truth_lat))),
           centre_lon + half / (111320.0 * math.cos(math.radians(truth_lat))))
    cfg = SearchConfig(grid_spacing_m=args.grid)
    cfg.defer_jets = not args.no_defer_jets
    if args.no_refine:
        cfg.refine_spacings_m = ()
        cfg.refine_window_half_m = ()
    t0 = time.time()
    out_dir = Path(args.out)
    result = run_search(perc, scene["segments"], box, height_fn=0.0,
                        config=cfg, out_dir=out_dir, frame=frame)
    print(f"run_search: {time.time() - t0:.1f} s; funnel={result.funnel_counts}")
    # --- Report ------------------------------------------------------------
    lat_marg, lon_marg, _p = result.peak(which="marg")
    lat_max, lon_max, _p2 = result.peak(which="max")
    err_marg = _error_m(lat_marg, lon_marg, truth_lat, truth_lon)
    err_max = _error_m(lat_max, lon_max, truth_lat, truth_lon)
    in90 = result.truth_in_region(truth_lat, truth_lon, level=0.90, which="marg")
    in95 = result.truth_in_region(truth_lat, truth_lon, level=0.95, which="marg")
    in50 = result.truth_in_region(truth_lat, truth_lon, level=0.50, which="marg")
    print(f"Peak (marginalised): {lat_marg:.5f}, {lon_marg:.5f}  error {err_marg:.0f} m")
    print(f"Peak (max):          {lat_max:.5f}, {lon_max:.5f}  error {err_max:.0f} m")
    if result.refined:
        r = result.refined[0]
        err_ref = _error_m(r.best_lat, r.best_lon, truth_lat, truth_lon)
        print(f"Refined peak:        {r.best_lat:.5f}, {r.best_lon:.5f}  error {err_ref:.0f} m")
    print(f"Truth in 50/90/95% regions (marg): {in50} / {in90} / {in95}")
    print(f"Modes: {result.modes}")
    passed = in90 and err_marg < 1000.0
    print(f"G3 {'PASS' if passed else 'FAIL'}: truth in 90%={in90}, "
          f"top-mode error {err_marg:.0f} m (<1000 m required for this smoke test; "
          f"full S2 target is <=500 m regional)")
    print(f"Total time: {time.time() - t_start:.1f} s; outputs in {out_dir}")
    summary = {"g3_pass": bool(passed), "error_m_marg": err_marg,
               "error_m_max": err_max,
               "truth_in_50_90_95": [bool(in50), bool(in90), bool(in95)],
               "peak_marg": [lat_marg, lon_marg], "peak_max": [lat_max, lon_max],
               "funnel_counts": result.funnel_counts,
               "truth": [truth_lat, truth_lon],
               "synth_truth": truth}
    (out_dir / "g3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
