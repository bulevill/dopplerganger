"""Opportunity map: where and when could a recording catch enough small aircraft to be located?

For one converted ADS-B day, for every ~2 km cell of CONUS, this estimates the chance that a
recording of W minutes, started at a random daytime moment (08-20 local solar time), contains
at least 1 / at least 2 audible small-aircraft passes. The search needs >= 2.

A "pass" is one aircraft (prop / turboprop / rotor; unknown-class reported separately as an upper
bound) whose track comes within about R_KM horizontally of the cell centre while at most
MAX_AGL_M above the ground. Its time is the time of closest approach. Thresholds are
deliberately simple proxies for audibility, not the M4 audibility model.

    python scripts/opportunity_map.py --day 2026-11-06

Outputs go to results/opportunity/<day>/ (maps, per-cell GeoTIFFs, summary.json). The coarse
CONUS ground model is cached at PROCESSED/dem/conus_ground_0p01.npz.

Limitations (also written to summary.json):
- One day only; weekday/weekend, season and weather change traffic a lot.
- ADS-B coverage: aircraft without ADS-B or out of receiver range are missing (remote areas).
- The converter keeps points below 10,000 ft MSL, so in high terrain (ground > ~1,500 m)
  low-AGL flights near the ceiling are partly missing.
- "Land" is everything inside the CONUS box, so border strips of Canada and Mexico are included.
- Audibility is a fixed distance/height cut-off; real audibility depends on background noise,
  wind, aircraft type and whether the recorder is indoors.
"""
import argparse
import calendar
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoloc.paths import PROCESSED  # noqa: E402
from geoloc.telemetry import CONUS  # noqa: E402
from geoloc.terrain import COP90_URL, _tile_name, geoid_height  # noqa: E402

SMALL = ("prop", "turboprop", "rotor")
CELL_M = 2000.0            # observer cell size (Albers metres)
R_KM = 3.0                 # pass = point within the 3x3 cell block, i.e. ~3 km horizontally
MAX_AGL_M = 1500.0         # ~5,000 ft above ground
DAY_START_H, DAY_END_H = 8.0, 20.0   # local solar time
WINDOWS_MIN = (10, 20, 30)
DEM_RES = 0.01             # degrees
DEM_CACHE = PROCESSED / "dem" / "conus_ground_0p01.npz"
ALBERS = "EPSG:5070"

# Places to report in the summary table (public reference points, not recording sites).
PLACES = {
    "Richmond VA (downtown)": (37.540, -77.436),
    "Hudson Valley NY (G3 truth)": (41.650, -73.850),
    "Manhattan NY": (40.750, -73.980),
    "Teterboro NJ (KTEB)": (40.850, -74.061),
    "Chicago IL (downtown)": (41.880, -87.630),
    "Rural Kansas (Russell)": (38.895, -98.860),
    "Denver CO (downtown)": (39.740, -104.990),
    "Phoenix AZ (downtown)": (33.450, -112.070),
    "Rural Nebraska Sandhills": (42.000, -101.000),
    "Orlando FL (downtown)": (28.540, -81.380),
    "Seattle WA (downtown)": (47.610, -122.330),
}


# --------------------------------------------------------------------------- ground model

def _tile_overview(lat_floor, lon_floor):
    """One GLO-90 tile read from its 4x overview (~360 m) over HTTP, averaged to DEM_RES. None = ocean."""
    url = "/vsicurl/" + COP90_URL.format(name=_tile_name(lat_floor, lon_floor))
    n = round(1.0 / DEM_RES)
    for attempt in range(3):
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                with rasterio.open(url) as src:
                    a = src.read(1, out_shape=(n, n), resampling=Resampling.average).astype(np.float32)
            return a
        except rasterio.errors.RasterioIOError as exc:
            if "404" in str(exc) or "does not exist" in str(exc) or "No such file" in str(exc):
                return None
            if attempt == 2:
                raise
            time.sleep(2 + 3 * attempt)


def build_ground(tiles):
    """Ground height above the WGS84 ellipsoid, and a land mask, on a DEM_RES lat/lon grid covering `tiles`.

    Land = the averaged Copernicus height is non-zero (sea pixels are exactly 0; missing ocean
    tiles count as sea). Large lakes above sea level count as land.
    """
    lats = [la for la, _ in tiles]
    lons = [lo for _, lo in tiles]
    la0, la1, lo0, lo1 = min(lats), max(lats) + 1, min(lons), max(lons) + 1
    n = round(1.0 / DEM_RES)
    H = np.full(((la1 - la0) * n, (lo1 - lo0) * n), np.nan, np.float32)  # row 0 = north
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        res = dict(zip(tiles, ex.map(lambda tl: _tile_overview(*tl), tiles)))
    ocean = 0
    for (la, lo), a in res.items():
        r = (la1 - 1 - la) * n
        c = (lo - lo0) * n
        if a is None:
            ocean += 1
            H[r:r + n, c:c + n] = 0.0
        else:
            H[r:r + n, c:c + n] = a
    print(f"  DEM: {len(tiles)} tiles ({ocean} ocean) in {time.time() - t0:.0f} s")
    land = np.isfinite(H) & (H != 0.0)
    # Geoid (EGM2008) on a 0.25 deg grid, interpolated: Copernicus heights are above the geoid.
    glat = np.arange(la0, la1 + 0.25, 0.25)
    glon = np.arange(lo0, lo1 + 0.25, 0.25)
    GLON, GLAT = np.meshgrid(glon, glat)
    N = geoid_height(GLAT.ravel(), GLON.ravel()).reshape(GLAT.shape)
    from scipy.interpolate import RegularGridInterpolator
    lat_c = la1 - (np.arange(H.shape[0]) + 0.5) * DEM_RES
    lon_c = lo0 + (np.arange(H.shape[1]) + 0.5) * DEM_RES
    f = RegularGridInterpolator((glat, glon), N)
    LONc, LATc = np.meshgrid(lon_c, lat_c)
    H = H + f(np.column_stack([LATc.ravel(), LONc.ravel()])).reshape(H.shape).astype(np.float32)
    return H, land, (la0, la1, lo0, lo1)


def ground_lookup(H, ext, lat, lon):
    la0, la1, lo0, lo1 = ext
    r = np.clip(((la1 - lat) / DEM_RES).astype(np.int64), 0, H.shape[0] - 1)
    c = np.clip(((lon - lo0) / DEM_RES).astype(np.int64), 0, H.shape[1] - 1)
    return H[r, c]


# --------------------------------------------------------------------------- passes

def union_fraction(times, W, lo, hi, need):
    """Fraction of start times s in [lo, hi] whose window [s, s+W] holds >= need sorted pass times.

    need=1: s in [t_i - W, t_i].  need=2: s in [t_{i+1} - W, t_i] (consecutive pairs; any window
    with two passes contains a consecutive pair).  Both interval ends are non-decreasing, so the
    union is a running-max sweep.
    """
    if len(times) < need:
        return 0.0
    if need == 1:
        a, b = times - W, times
    else:
        a, b = times[1:] - W, times[:-1]
        ok = a <= b
        a, b = a[ok], b[ok]
        if len(a) == 0:
            return 0.0
    a = np.clip(a, lo, hi)
    b = np.clip(b, lo, hi)
    prev = np.maximum.accumulate(np.concatenate([[lo], b[:-1]]))
    return float(np.sum(np.clip(b - np.maximum(a, prev), 0, None)) / (hi - lo))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", default="2026-11-06")
    ap.add_argument("--max-agl", type=float, default=MAX_AGL_M)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    pts = (PROCESSED / "adsblol" / args.day / "points.parquet").as_posix()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "opportunity" / args.day
    out.mkdir(parents=True, exist_ok=True)
    day0 = calendar.timegm(time.strptime(args.day, "%Y-%m-%d"))  # UTC midnight
    t_all = time.time()

    # 1. Small-aircraft points, thinned to one per 10 s per track (~0.5 km at piston speeds).
    cls_list = ", ".join(f"'{c}'" for c in SMALL + ("unknown",))
    df = duckdb.sql(f"""
        SELECT icao, seg, cls, t, lat, lon, coalesce(alt_geom_m, alt_baro_m) AS alt
        FROM (SELECT *, row_number() OVER (PARTITION BY icao, seg, floor(t / 10) ORDER BY t) AS k
              FROM '{pts}' WHERE cls IN ({cls_list}))
        WHERE k = 1 AND alt IS NOT NULL AND NOT isnan(alt)
    """).df()
    print(f"points (thinned): {len(df):,}  in {time.time() - t_all:.0f} s")

    # 2. Height above ground. The ground model covers every 1-degree tile of the converter's CONUS box.
    tiles = [(la, lo) for la in range(int(CONUS[0]), int(CONUS[1])) for lo in range(int(CONUS[2]), int(CONUS[3]))]
    if DEM_CACHE.exists():
        z = np.load(DEM_CACHE)
        H, LAND, ext = z["H"], z["land"], tuple(int(v) for v in z["ext"])
    else:
        print(f"building ground model from {len(tiles)} GLO-90 tiles (overviews over HTTP) ...")
        H, LAND, ext = build_ground(tiles)
        DEM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DEM_CACHE, H=H, land=LAND, ext=np.array(ext))
    agl = df.alt.values - ground_lookup(H, ext, df.lat.values, df.lon.values)
    df["agl"] = agl
    df = df[(agl <= args.max_agl) & (agl > -150)]
    print(f"points <= {args.max_agl:.0f} m AGL: {len(df):,}")

    # 3. Albers cells; expand to the 3x3 block; closest approach per (observer cell, track).
    tr = pyproj.Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
    x, y = tr.transform(df.lon.values, df.lat.values)
    ix0, iy0 = math.floor(np.min(x) / CELL_M) - 1, math.floor(np.min(y) / CELL_M) - 1
    ix = (np.floor(x / CELL_M) - ix0).astype(np.int32)
    iy = (np.floor(y / CELL_M) - iy0).astype(np.int32)
    nx, ny = int(ix.max()) + 2, int(iy.max()) + 2
    # Track id, small-vs-unknown flag.
    trk = (df.icao.astype(str) + ":" + df.seg.astype(str)).astype("category").cat.codes.values.astype(np.int64)
    unk = (df.cls.values == "unknown")
    rows = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            d2 = (x - (ix0 + cx + 0.5) * CELL_M) ** 2 + (y - (iy0 + cy + 0.5) * CELL_M) ** 2
            rows.append((cy.astype(np.int64) * nx + cx, trk, d2, df.t.values, unk))
    cell = np.concatenate([r[0] for r in rows])
    trk9 = np.concatenate([r[1] for r in rows])
    d2 = np.concatenate([r[2] for r in rows])
    t9 = np.concatenate([r[3] for r in rows])
    unk9 = np.concatenate([r[4] for r in rows])
    keep = d2 <= (R_KM * 1000.0) ** 2
    cell, trk9, d2, t9, unk9 = cell[keep], trk9[keep], d2[keep], t9[keep], unk9[keep]
    o = np.lexsort((d2, trk9, cell))           # by cell, track, distance
    cell, trk9, t9, unk9 = cell[o], trk9[o], t9[o], unk9[o]
    first = np.ones(len(cell), bool)
    first[1:] = (cell[1:] != cell[:-1]) | (trk9[1:] != trk9[:-1])
    pc, pt, pu = cell[first], t9[first], unk9[first]  # one pass per (cell, track): time of CPA
    print(f"passes (cell x track): {len(pc):,} ({int(pu.sum()):,} unknown-class)")

    # 4. Local solar time of day; daytime start range per cell.
    ccx = (pc % nx + ix0 + 0.5) * CELL_M
    ccy = (pc // nx + iy0 + 0.5) * CELL_M
    clon, clat = pyproj.Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True).transform(ccx, ccy)
    tod = np.mod(pt - day0 + clon / 15.0 * 3600.0, 86400.0)

    results = {}
    for label, sel in (("small", ~pu), ("small_plus_unknown", np.ones(len(pc), bool))):
        c_, t_ = pc[sel], tod[sel]
        o = np.lexsort((t_, c_))
        c_, t_ = c_[o], t_[o]
        starts = np.flatnonzero(np.r_[True, c_[1:] != c_[:-1]])
        ends = np.r_[starts[1:], len(c_)]
        day_lo = DAY_START_H * 3600.0
        per_hour = np.zeros(ny * nx, np.float32)
        frac = {(W, k): np.zeros(ny * nx, np.float32) for W in WINDOWS_MIN for k in (1, 2)}
        for s, e in zip(starts, ends):
            tt = t_[s:e]
            tt = tt[(tt >= day_lo) & (tt <= DAY_END_H * 3600.0)]
            cid = c_[s]
            per_hour[cid] = len(tt) / (DAY_END_H - DAY_START_H)
            for W in WINDOWS_MIN:
                w = W * 60.0
                for k in (1, 2):
                    frac[(W, k)][cid] = union_fraction(tt, w, day_lo, DAY_END_H * 3600.0 - w, k)
        results[label] = (per_hour.reshape(ny, nx), {k: v.reshape(ny, nx) for k, v in frac.items()})
    print(f"window statistics done ({time.time() - t_all:.0f} s total)")

    # 5. Land mask on the Albers grid (from the ground model: NaN outside tiles, 0 in ocean tiles).
    gx = (ix0 + np.arange(nx) + 0.5) * CELL_M
    gy = (iy0 + np.arange(ny) + 0.5) * CELL_M
    GX, GY = np.meshgrid(gx, gy)
    glon, glat = pyproj.Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True).transform(GX, GY)
    la0, la1, lo0, lo1 = ext
    inside = (glat >= la0) & (glat < la1) & (glon >= lo0) & (glon < lo1)
    land = np.zeros(GX.shape, bool)
    land[inside] = ground_lookup(LAND, ext, glat[inside], glon[inside])

    # 6. Outputs: GeoTIFFs, maps, summary.
    transform = rasterio.transform.from_origin(ix0 * CELL_M, (iy0 + ny) * CELL_M, CELL_M, CELL_M)
    summary = {"day": args.day, "cell_m": CELL_M, "radius_km": R_KM, "max_agl_m": args.max_agl,
               "daytime_local_solar_h": [DAY_START_H, DAY_END_H], "windows_min": list(WINDOWS_MIN),
               "classes": list(SMALL), "upper_bound_adds": "unknown",
               "limitations": __doc__.split("Limitations (also written to summary.json):")[1].strip()}
    land_n = int(land.sum())
    for label, (per_hour, frac) in results.items():
        prof = dict(
            driver="GTiff", height=ny, width=nx, count=1 + 2 * len(WINDOWS_MIN), dtype="float32",
            crs=ALBERS, transform=transform, compress="deflate", nodata=None, tiled=True,
            blockxsize=256, blockysize=256)
        with rasterio.open(out / f"opportunity_{label}.tif", "w", **prof) as dst:
            dst.write(per_hour[::-1], 1)
            dst.set_band_description(1, "passes per daytime hour")
            b = 2
            for W in WINDOWS_MIN:
                for k in (1, 2):
                    dst.write(frac[(W, k)][::-1], b)
                    dst.set_band_description(b, f"P(>= {k} passes in {W} min)")
                    b += 1
        s = {"land_cells": land_n}
        for W in WINDOWS_MIN:
            p2 = frac[(W, 2)][land]
            s[f"{W}min"] = {f"share_of_land_P2_ge_{q}": round(float(np.mean(p2 >= q)), 4) for q in (0.1, 0.25, 0.5, 0.9)}
            s[f"{W}min"]["share_of_land_P1_ge_0.5"] = round(float(np.mean(frac[(W, 1)][land] >= 0.5)), 4)
        places = {}
        tr_f = pyproj.Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
        for name, (la, lo) in PLACES.items():
            px, py = tr_f.transform(lo, la)
            cx, cy = int(math.floor(px / CELL_M) - ix0), int(math.floor(py / CELL_M) - iy0)
            if 0 <= cx < nx and 0 <= cy < ny:
                # Median over the surrounding 5x5 cells (10 km), so one cell's luck doesn't dominate.
                sl = (slice(max(cy - 2, 0), cy + 3), slice(max(cx - 2, 0), cx + 3))
                places[name] = {"passes_per_hour": round(float(np.median(per_hour[sl])), 2),
                                **{f"P2_{W}min": round(float(np.median(frac[(W, 2)][sl])), 3) for W in WINDOWS_MIN},
                                "P1_30min": round(float(np.median(frac[(30, 1)][sl])), 3)}
        s["places"] = places
        summary[label] = s
        _plot(out / f"map_{label}_30min.png", frac[(30, 2)], land, gx, gy, PLACES, tr_f, ix0, iy0,
              f"Chance a 30-min daytime recording catches >= 2 small-aircraft passes ({args.day})"
              + ("" if label == "small" else "\n(upper bound: unknown-class aircraft counted as small)"))
        _plot(out / f"map_{label}_10min_1pass.png", frac[(10, 1)], land, gx, gy, PLACES, tr_f, ix0, iy0,
              f"Chance a 10-min daytime recording catches >= 1 small-aircraft pass ({args.day})"
              + ("" if label == "small" else "\n(upper bound: unknown-class aircraft counted as small)"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "limitations"}, indent=2))
    print(f"outputs in {out}  ({time.time() - t_all:.0f} s)")


def _plot(path, field, land, gx, gy, places, tr_f, ix0, iy0, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    show = np.where(land, field, np.nan)
    bounds = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0001]
    cmap = ListedColormap(["#f2f0ea", "#fde7a9", "#fbbf5d", "#f08a3c", "#d9502a", "#a3201f", "#5c0a17"])
    cmap.set_bad("#9fb7c9")
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor("#9fb7c9")
    m = ax.pcolormesh(gx / 1000, gy / 1000, show, cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), shading="auto",
                      rasterized=True)
    for name, (la, lo) in places.items():
        px, py = tr_f.transform(lo, la)
        ax.plot(px / 1000, py / 1000, "o", ms=4, mfc="none", mec="#222")
        ax.annotate(name.split(" (")[0], (px / 1000, py / 1000), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color="#222")
    cb = fig.colorbar(m, ax=ax, shrink=0.7, ticks=bounds[:-1] + [1.0])
    cb.ax.set_yticklabels(["0", "1%", "5%", "10%", "25%", "50%", "75%", "100%"])
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    landy = np.flatnonzero(land.any(axis=1))
    landx = np.flatnonzero(land.any(axis=0))
    ax.set_xlim(gx[landx[0]] / 1000, gx[landx[-1]] / 1000)
    ax.set_ylim(gy[landy[0]] / 1000, gy[landy[-1]] / 1000)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
