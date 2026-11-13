"""Gate G1: profile one adsb.lol day. Run: python scripts/g1_probe_adsblol.py 2026-11-06

Answers: how many aircraft and points, how many are props/rotorcraft, how often
low-altitude props report a position, how often geometric altitude is present,
and how much of it is over the continental US.
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from geoloc.adsblol import acoustic_class, iter_traces, load_type_table
from geoloc.paths import RAW, REPORTS, require_data_root

LOW_ALT_FT = 10_000  # below this, a prop or rotorcraft may be audible
CONUS = (24.0, 50.0, -125.0, -66.0)  # lat_min, lat_max, lon_min, lon_max


def main(day):
    require_data_root()
    types = load_type_table()
    aircraft = Counter()
    points = Counter()
    low_points = Counter()
    conus_aircraft = Counter()
    sources = Counter()
    geom_present = Counter()
    gaps = defaultdict(list)  # seconds between consecutive low-altitude airborne points
    no_type = 0

    for n, tr in enumerate(iter_traces(RAW / "adsblol" / day), 1):
        typecode = tr.get("t")
        cls = acoustic_class(types.get(typecode, [None, None])[1]) if typecode else "unknown"
        no_type += typecode is None
        rows = tr.get("trace") or []
        aircraft[cls] += 1
        points[cls] += len(rows)

        low_t, in_conus = [], False
        for row in rows:
            alt = row[3]
            if alt == "ground" or alt is None or alt >= LOW_ALT_FT:
                continue
            low_t.append(row[0])
            sources[(cls, row[9])] += 1
            geom_present[(cls, row[10] is not None)] += 1
            lat, lon = row[1], row[2]
            in_conus |= CONUS[0] <= lat <= CONUS[1] and CONUS[2] <= lon <= CONUS[3]
        low_points[cls] += len(low_t)
        conus_aircraft[cls] += in_conus
        if len(low_t) > 1:
            d = np.diff(low_t)
            gaps[cls].append(d[d < 300])  # longer gaps are coverage holes, counted separately

        if n % 10_000 == 0:
            print(f"  {n:,} aircraft read", flush=True)

    report = {"day": day, "aircraft_total": sum(aircraft.values()), "aircraft_without_typecode": no_type,
              "by_class": {}}
    for cls in sorted(aircraft):
        g = np.concatenate(gaps[cls]) if gaps[cls] else np.array([])
        low = low_points[cls]
        report["by_class"][cls] = {
            "aircraft": aircraft[cls],
            "aircraft_low_alt_over_conus": conus_aircraft[cls],
            "points": points[cls],
            "points_below_10k_ft": low,
            "gap_s_p10_p50_p90": [round(float(x), 2) for x in np.percentile(g, [10, 50, 90])] if g.size else None,
            "share_with_alt_geom": round(geom_present[(cls, True)] / low, 3) if low else None,
            "sources": {s: c for (k, s), c in sources.most_common() if k == cls},
        }

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"g1_adsblol_{day}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-11-06")
