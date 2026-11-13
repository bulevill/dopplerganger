"""Pick a G3b time window: count small-plane segments per hour.

The user's 1 h noon window on 2026-11-06 had only one prop/turboprop
in each 50 km box -- too few events for a real-track test.  This
script counts, for one day's points.parquet and one box, how many
segments of each class fly in the box in each UTC hour, so a time
and box with more small-plane (prop / turboprop / rotor) traffic
can be picked before running G3b.

Run on the Windows machine (conda env "geoloc"):

    python scripts\\g3b_pick_window.py --day 2026-11-06 --preset hudson
    python scripts\\g3b_pick_window.py --day 2026-11-06 --preset newyork
    python scripts\\g3b_pick_window.py --day 2026-11-06 --box 41.425 41.875 -74.15 -73.55

A segment is counted in every hour in which it has a point in the
box (that is the hour in which it could be heard).  "Small" =
prop + turboprop + rotor.  Jets and unknown-class aircraft are
shown separately: jets matter for the silence check, unknown-class
aircraft are treated as possible matches (weight 0.5, see
corridor_cpa_estimate), so both are worth knowing about.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoloc.paths import PROCESSED

PRESETS = {
    # 50 km boxes: +/-0.225 deg lat, +/-0.30 deg lon.
    "hudson": (41.65 - 0.225, 41.65 + 0.225, -73.85 - 0.30, -73.85 + 0.30),
    "newyork": (40.75 - 0.225, 40.75 + 0.225, -73.98 - 0.30, -73.98 + 0.30),
}

CLASSES = ("prop", "turboprop", "rotor", "jet", "unknown")
SMALL = ("prop", "turboprop", "rotor")


def _day_start(day):
    dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def count_by_hour(points_path, box, day_start):
    """Return {hour_utc: {cls: n_segments}} for segments in the box.

    ``hour_utc`` is the hour of day (0-23) of a point's timestamp;
    a segment spanning hours is counted in each hour it is in the
    box.  Only points inside the box itself are counted here (the
    loader's expanded hit may add a few near-edge flights on top).
    """
    import duckdb

    lat_min, lat_max, lon_min, lon_max = [float(v) for v in box]
    lit = str(points_path).replace("'", "''")
    rows = duckdb.sql(f"""
        SELECT CAST(FLOOR((t - {day_start!r}) / 3600) AS INTEGER) AS hour,
               CAST(cls AS VARCHAR) AS cls,
               COUNT(DISTINCT icao || ':' || seg) AS n
        FROM '{lit}'
        WHERE lat BETWEEN {lat_min} AND {lat_max}
          AND lon BETWEEN {lon_min} AND {lon_max}
          AND t >= {day_start!r} AND t < {day_start + 86400!r}
        GROUP BY hour, cls
        ORDER BY hour, cls
    """).df()
    out = {}
    for _, r in rows.iterrows():
        h = int(r["hour"])
        if not 0 <= h < 24:
            continue
        out.setdefault(h, {})[str(r["cls"])] = int(r["n"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", required=True, help="day, YYYY-MM-DD, e.g. 2026-11-06")
    ap.add_argument("--points", default=None,
                    help="points.parquet path (default: "
                         "D:\\geoloc-data\\processed\\adsblol\\<day>\\points.parquet)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="hudson")
    ap.add_argument("--box", nargs=4, type=float, default=None,
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                    help="override the preset box")
    args = ap.parse_args(argv)

    points = Path(args.points) if args.points else \
        PROCESSED / "adsblol" / args.day / "points.parquet"
    if not points.exists():
        ap.error(f"points.parquet not found: {points} "
                 "(set --points, or GEOLOC_DATA, or plug in the drive)")
    box = tuple(args.box) if args.box else PRESETS[args.preset]
    day_start = _day_start(args.day)
    counts = count_by_hour(points, box, day_start)

    print(f"points: {points}")
    print(f"day: {args.day}  box: lat [{box[0]:.4f}, {box[1]:.4f}] "
          f"lon [{box[2]:.4f}, {box[3]:.4f}] (preset {args.preset})")
    print(f"segments in the box per UTC hour "
          f"(a segment spanning hours is counted in each):")
    print(f"{'hour':>4} {'prop':>5} {'turbo':>6} {'rotor':>6} "
          f"{'jet':>5} {'unk':>4} {'small':>6} {'total':>6}")
    best = None
    for h in range(24):
        c = counts.get(h, {})
        small = sum(c.get(k, 0) for k in SMALL)
        total = sum(c.get(k, 0) for k in CLASSES) + sum(
            v for k, v in c.items() if k not in CLASSES)
        print(f"{h:>4} {c.get('prop', 0):>5} {c.get('turboprop', 0):>6} "
              f"{c.get('rotor', 0):>6} {c.get('jet', 0):>5} "
              f"{c.get('unknown', 0):>4} {small:>6} {total:>6}")
        if best is None or small > best[1]:
            best = (h, small)
    if best is not None:
        print(f"most small-plane traffic: {best[0]:02d}:00 UTC "
              f"({best[1]} prop/turboprop/rotor segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
