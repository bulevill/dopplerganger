"""G3b real-day timing harness -- RUN ON THE USER'S WINDOWS MACHINE.

The original development sandbox had NO real 2026-11-06 points.parquet, so the
real-day numbers were NOT measured in this environment.  Run this
script on the Windows machine (conda env "geoloc") against
D:\\geoloc-data\\processed\\adsblol\\2026-11-06\\points.parquet:

    python scripts/g3b_real.py --day 2026-11-06 --preset hudson --timing-only
    python scripts/g3b_real.py --day 2026-11-06 --preset newyork --timing-only
    python scripts/g3b_real.py --day 2026-11-06 --preset hudson --audio rec.wav

--timing-only loads the segments (time filter + trim), builds the
50 km / 200 m grid, prints the guard estimate (old product vs new
corridor, candidate vs silence-only), builds the CPA table, and
prints load / CPA times -- no audio needed.  With --audio it also
runs the full run_search and prints the per-stage timing breakdown
(load, CPA table, pair gate, silence, scoring, refinement,
marginalised sweep, total).

Presets are EXAMPLE boxes -- replace them with the boxes actually
used on 2026-11-06 (pass --box lat_min lat_max lon_min lon_max to
override).  hudson is centred on 41.65 N, -73.85 W (the synthetic
truth area); newyork is centred on Manhattan (40.75 N, -73.98 W).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from geoloc import audio as audio_mod
from geoloc.geodesy import LocalFrame
from geoloc.paths import PROCESSED
from geoloc.search import (SearchConfig, build_cpa_table,
                           corridor_cpa_estimate, load_segments_with_stats,
                           make_grid, run_search, split_early_deferred)

PRESETS = {
    # 50 km boxes: +/-0.225 deg lat, +/-0.30 deg lon at ~41 N.
    "hudson": (41.65 - 0.225, 41.65 + 0.225, -73.85 - 0.30, -73.85 + 0.30),
    "newyork": (40.75 - 0.225, 40.75 + 0.225, -73.98 - 0.30, -73.98 + 0.30),
}


def _parse_time(s):
    """Unix seconds, or ISO 8601 (e.g. 2026-11-06T12:00:00Z)."""
    try:
        return float(s)
    except ValueError:
        pass
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _empty_perception(duration=1600.0):
    t = np.arange(0.0, duration, 0.25)
    f = np.arange(30.0, 501.0, 1.0)
    spec = audio_mod.Spectrogram(t=t, f=f, S=np.zeros((len(f), len(t))),
                                 loud=np.zeros(len(t)))
    return audio_mod.Perception(duration=duration, events=[], silences=[],
                                spec=spec)


def _print_breakdown(fc):
    rows = [("load_segments", fc.get("load_seconds")),
            ("CPA table", fc.get("time_cpa_table_s")),
            ("pair gate", fc.get("time_pair_gate_s")),
            ("silence (subset of scoring/posterior/refine)",
             fc.get("time_silence_s")),
            ("scoring (F8)", fc.get("time_scoring_s")),
            ("refinement (F9)", fc.get("time_refine_s")),
            ("marginalised sweep", fc.get("time_marginalised_sweep_s")),
            ("TOTAL run_search", fc.get("time_total_s"))]
    print("  timing breakdown (s):")
    for name, val in rows:
        if val is not None:
            print(f"    {name:>55}: {val:>8.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="2026-11-06",
                    help="date folder under processed/adsblol (default 2026-11-06)")
    ap.add_argument("--points", default=None,
                    help="points.parquet path (default: "
                         "PROCESSED/adsblol/<day>/points.parquet)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="hudson")
    ap.add_argument("--box", nargs=4, type=float, metavar=("LAT_MIN", "LAT_MAX",
                                                           "LON_MIN", "LON_MAX"),
                    help="override the preset box")
    ap.add_argument("--t-start", default=None,
                    help="recording window start (Unix or ISO 8601)")
    ap.add_argument("--t-end", default=None,
                    help="recording window end (Unix or ISO 8601)")
    ap.add_argument("--time-margin", type=float, default=60.0,
                    help="time margin in seconds (default 60)")
    ap.add_argument("--grid", type=float, default=200.0,
                    help="grid spacing in metres (default 200)")
    ap.add_argument("--audio", default=None,
                    help="recording file (WAV/MP3) for a full run_search")
    ap.add_argument("--timing-only", action="store_true",
                    help="load + CPA table only, no run_search")
    ap.add_argument("--max-cpa-entries", type=int, default=5_000_000,
                    help="scale-guard cap on the corridor CPA estimate "
                         "(default 5,000,000, the SearchConfig default). "
                         "Raise it only after the limit-calibration "
                         "benchmark (see README_G3.md) -- e.g. 50,000,000 "
                         "for a whole-day Hudson box.  The cap is on the "
                         "corridor ESTIMATE, which over-counts the actual "
                         "table by ~1.4-2.9x, so a 50M cap admits roughly "
                         "17-35M actual entries.")
    ap.add_argument("--out", default=None, help="output dir for a full run")
    args = ap.parse_args(argv)

    points = Path(args.points) if args.points else \
        PROCESSED / "adsblol" / args.day / "points.parquet"
    if not points.exists():
        ap.error(f"points.parquet not found: {points} "
                 "(set --points, or GEOLOC_DATA, or plug in the drive)")
    box = tuple(args.box) if args.box else PRESETS[args.preset]
    lat_min, lat_max, lon_min, lon_max = box
    frame = LocalFrame(0.5 * (lat_min + lat_max), 0.5 * (lon_min + lon_max), 0.0)
    t_start = _parse_time(args.t_start) if args.t_start else None
    t_end = _parse_time(args.t_end) if args.t_end else None
    margin = float(args.time_margin)
    cfg = SearchConfig(grid_spacing_m=float(args.grid))
    cfg.max_cpa_entries = int(args.max_cpa_entries)
    print(f"scale-guard cap: {cfg.max_cpa_entries:,} CPA entries (estimate)")

    print(f"points: {points}")
    print(f"box: lat [{lat_min:.4f}, {lat_max:.4f}] lon [{lon_min:.4f}, {lon_max:.4f}] "
          f"(preset {args.preset})")
    print(f"time window: t_start={t_start} t_end={t_end} margin={margin} s")
    if t_start is None and t_end is None:
        print("  NOTE: no time filter -- the whole day is loaded, and the "
              "size question comes back (see README_G3.md).")
    print(f"grid spacing: {args.grid} m")

    t0 = time.time()
    segs, stats = load_segments_with_stats(
        points, frame, lat_min, lat_max, lon_min, lon_max,
        t_start=t_start, t_end=t_end, time_margin_s=margin,
        trim_spatial=True, config=cfg)
    load_s = time.time() - t0
    print(f"loaded {len(segs)} segments in {load_s:.1f} s "
          f"(loader reports load_seconds={stats.get('load_seconds')})")
    print(f"  loader stats: {stats}")

    grid = make_grid(lat_min, lat_max, lon_min, lon_max, float(args.grid),
                     frame, height_fn=0.0)
    n_cells = len(grid.enu)
    product = n_cells * max(1, len(segs))
    est, _per_seg, bd = corridor_cpa_estimate(segs, grid, cfg.rmax_m)
    print(f"grid cells: {n_cells:,}")
    print(f"  OLD product estimate: {product:,}  (cap {cfg.max_cpa_entries:,})")
    print(f"  NEW corridor estimate: {est:,}  "
          f"(candidate {bd['candidate']:,}, silence-only {bd['silence_only']:,}, "
          f"unknown {bd['unknown']:,})")
    print(f"  ratio corridor/product: {est / max(1, product):.4f}")
    print(f"  guard: {'PASS' if est <= cfg.max_cpa_entries else 'REFUSE'} "
          f"(cap {cfg.max_cpa_entries:,})")
    # Jet deferral: the early CPA table below is built from early
    # segments only (high jets deferred to a per-cell silence pass
    # inside run_search).  Report both, so the ~100x saving is
    # visible and the deferred-jet silence is not forgotten.
    early_segs, deferred_segs = split_early_deferred(segs, cfg)
    est_e, _pe, bd_e = corridor_cpa_estimate(early_segs, grid, cfg.rmax_m)
    est_d, _pd, _bd_d = corridor_cpa_estimate(deferred_segs, grid, cfg.rmax_m)
    print(f"  jet deferral: {len(early_segs)} early / {len(deferred_segs)} "
          f"deferred high jets; early corridor {est_e:,} vs deferred "
          f"{est_d:,} (deferred jets are NOT in the guard or the "
          f"early table)")

    perc = _empty_perception()
    t0 = time.time()
    cpa = build_cpa_table(grid, early_segs, perc, cfg)
    cpa_s = time.time() - t0
    print(f"CPA table (early segments only): {int(cpa.cell_ptr[-1]):,} "
          f"entries in {cpa_s:.1f} s")

    if args.timing_only or args.audio is None:
        print("timing-only mode: no run_search (pass --audio <file> for a full run).")
        print(f"SUMMARY load={load_s:.1f}s cpa={cpa_s:.1f}s "
              f"corridor={est:,} product={product:,}")
        return 0

    print(f"detecting events in {args.audio} ...")
    y = audio_mod.load(args.audio)
    perc = audio_mod.detect(y)
    print(f"  detected {len(perc.events)} events, {len(perc.silences)} silences")
    t0 = time.time()
    result = run_search(perc, segs, box, height_fn=0.0, config=cfg,
                        frame=frame, t_start=t_start, t_end=t_end,
                        time_margin_s=margin, out_dir=args.out)
    total_s = time.time() - t0
    fc = dict(result.funnel_counts)
    fc["load_seconds"] = round(load_s, 3)
    print(f"run_search finished in {total_s:.1f} s "
          f"(funnel time_total_s={fc.get('time_total_s')})")
    _print_breakdown(fc)
    print(f"  hypotheses={fc.get('pair_hypotheses')} blobs={fc.get('blobs')} "
          f"scored={fc.get('blobs_scored')} modes={len(result.modes)}")
    for m in result.modes[:3]:
        flights = [(f["icao"], f["seg"], f["piece"]) for f in m.get("flights", [])]
        print(f"  mode lat={m['lat']:.5f} lon={m['lon']:.5f} "
              f"mass={m['mass']:.3f} flights={flights}")
    print(f"SUMMARY load={load_s:.1f}s cpa={cpa_s:.1f}s "
          f"run_search={total_s:.1f}s corridor={est:,} product={product:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
