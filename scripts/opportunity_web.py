"""Export the opportunity-map GeoTIFFs as compact web layers for the interactive page.

    python scripts/opportunity_web.py --day 2026-11-06 --out <dir>

Writes 8-bit grayscale PNG layers (0 = no data / zero), a relief layer (0 = sea), and
meta.json with layer descriptions, value scales, lat/lon <-> pixel lookup grids, place
markers and the summary numbers. Values are quantised: fractions as round(p * 250);
passes per hour on a log scale, round(log1p(v) / log1p(PPH_MAX) * 250).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoloc.paths import PROCESSED  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opportunity_map import ALBERS, DEM_CACHE, PLACES, WINDOWS_MIN, ground_lookup  # noqa: E402

PPH_MAX = 20.0
CITIES = {
    "Seattle": (47.61, -122.33), "Portland": (45.52, -122.68), "San Francisco": (37.77, -122.42),
    "Sacramento": (38.58, -121.49), "Los Angeles": (34.05, -118.24), "San Diego": (32.72, -117.16),
    "Las Vegas": (36.17, -115.14), "Phoenix": (33.45, -112.07), "Tucson": (32.22, -110.97),
    "Salt Lake City": (40.76, -111.89), "Boise": (43.62, -116.20), "Reno": (39.53, -119.81),
    "Spokane": (47.66, -117.43), "Billings": (45.78, -108.50), "Denver": (39.74, -104.99),
    "Albuquerque": (35.08, -106.65), "El Paso": (31.76, -106.49), "Dallas": (32.78, -96.80),
    "Houston": (29.76, -95.37), "San Antonio": (29.42, -98.49), "Austin": (30.27, -97.74),
    "Oklahoma City": (35.47, -97.52), "Kansas City": (39.10, -94.58), "Omaha": (41.26, -95.94),
    "Fargo": (46.88, -96.79), "Minneapolis": (44.98, -93.27), "Milwaukee": (43.04, -87.91),
    "Chicago": (41.88, -87.63), "St. Louis": (38.63, -90.20), "Memphis": (35.15, -90.05),
    "New Orleans": (29.95, -90.07), "Nashville": (36.16, -86.78), "Louisville": (38.25, -85.76),
    "Indianapolis": (39.77, -86.16), "Cincinnati": (39.10, -84.51), "Columbus": (39.96, -83.00),
    "Detroit": (42.33, -83.05), "Cleveland": (41.50, -81.69), "Pittsburgh": (40.44, -79.99),
    "Atlanta": (33.75, -84.39), "Charlotte": (35.23, -80.84), "Raleigh": (35.78, -78.64),
    "Jacksonville": (30.33, -81.66), "Orlando": (28.54, -81.38), "Tampa": (27.95, -82.46),
    "Miami": (25.76, -80.19), "Richmond": (37.54, -77.44), "Washington": (38.91, -77.04),
    "Baltimore": (39.29, -76.61), "Philadelphia": (39.95, -75.17), "New York": (40.71, -74.01),
    "Boston": (42.36, -71.06),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-11-06")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    src_dir = Path(__file__).resolve().parents[1] / "results" / "opportunity" / args.day
    out = Path(args.out)
    (out / "layers").mkdir(parents=True, exist_ok=True)
    summary = json.loads((src_dir / "summary.json").read_text())

    # Grid geometry from the first GeoTIFF (north-up rows).
    with rasterio.open(src_dir / "opportunity_small.tif") as ds:
        T, ny, nx = ds.transform, ds.height, ds.width
        cell = T.a
    gx = T.c + (np.arange(nx) + 0.5) * cell
    gy = T.f - (np.arange(ny) + 0.5) * cell
    GX, GY = np.meshgrid(gx, gy)
    inv = pyproj.Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
    glon, glat = inv.transform(GX, GY)

    # Land / relief from the cached ground model.
    z = np.load(DEM_CACHE)
    H, LAND, ext = z["H"], z["land"], tuple(int(v) for v in z["ext"])
    la0, la1, lo0, lo1 = ext
    inside = (glat >= la0) & (glat < la1) & (glon >= lo0) & (glon < lo1)
    land = np.zeros((ny, nx), bool)
    land[inside] = ground_lookup(LAND, ext, glat[inside], glon[inside])
    elev = np.zeros((ny, nx), np.float32)
    elev[inside] = ground_lookup(H, ext, glat[inside], glon[inside])
    # Crop to the land bounding box (plus a small margin).
    rows = np.flatnonzero(land.any(axis=1))
    cols = np.flatnonzero(land.any(axis=0))
    r0, r1 = max(rows[0] - 10, 0), min(rows[-1] + 11, ny)
    c0, c1 = max(cols[0] - 10, 0), min(cols[-1] + 11, nx)
    crop = (slice(r0, r1), slice(c0, c1))
    land, elev = land[crop], elev[crop]
    h, w = land.shape

    # Relief: hillshade (sun from the north-west), 1..255 on land, 0 at sea.
    gy_, gx_ = np.gradient(np.where(land, elev, 0.0), cell)
    slope = np.arctan(np.hypot(gx_, gy_) * 8.0)  # vertical exaggeration for a 2 km grid
    aspect = np.arctan2(-gx_, gy_)
    az, alt = math.radians(315.0), math.radians(45.0)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    rel = np.clip(1 + np.round((shade - shade[land].min()) / (np.ptp(shade[land]) + 1e-9) * 254), 1, 255)
    rel = np.where(land, rel, 0).astype(np.uint8)
    Image.fromarray(rel, "L").save(out / "layers" / "relief.png", optimize=True)

    layers = []
    for label in ("small", "small_plus_unknown"):
        with rasterio.open(src_dir / f"opportunity_{label}.tif") as ds:
            data = ds.read()[:, r0:r1, c0:c1]
        bands = [("pph", None, None)] + [("frac", W, k) for W in WINDOWS_MIN for k in (1, 2)]
        for b, (kind, W, k) in enumerate(bands):
            v = np.nan_to_num(data[b])
            if kind == "pph":
                q = np.round(np.clip(np.log1p(v) / math.log1p(PPH_MAX), 0, 1) * 250)
                lid = f"{label}_pph"
            else:
                q = np.round(np.clip(v, 0, 1) * 250)
                lid = f"{label}_w{W}_k{k}"
            q = np.where(land, q, 0).astype(np.uint8)
            Image.fromarray(q, "L").save(out / "layers" / f"{lid}.png", optimize=True)
            layers.append({"id": lid, "set": label, "kind": kind, "window_min": W, "min_passes": k,
                           "file": f"layers/{lid}.png"})

    # Lookup grids: lat/lon -> pixel (0.25 deg), pixel -> lat/lon (every 16 px).
    flat = np.arange(24.0, 50.0001, 0.25)
    flon = np.arange(-125.0, -66.0 + 0.0001, 0.25)
    FLON, FLAT = np.meshgrid(flon, flat)
    fx, fy = fwd.transform(FLON, FLAT)
    px = (fx - T.c) / cell - 0.5 - c0
    py = (T.f - fy) / cell - 0.5 - r0
    step = 16
    ix = np.arange(0, w + step, step)
    iy = np.arange(0, h + step, step)
    IX, IY = np.meshgrid(ix, iy)
    ilon, ilat = inv.transform(T.c + (IX + c0 + 0.5) * cell, T.f - (IY + r0 + 0.5) * cell)

    def to_px(la, lo):
        x, y = fwd.transform(lo, la)
        return round(float((x - T.c) / cell - 0.5 - c0), 1), round(float((T.f - y) / cell - 0.5 - r0), 1)

    meta = {
        "day": args.day, "width": w, "height": h, "cell_m": cell, "pph_max": PPH_MAX,
        "layers": layers, "relief": "layers/relief.png",
        "fwd": {"lat0": 24.0, "lon0": -125.0, "step": 0.25, "nlat": len(flat), "nlon": len(flon),
                "x": np.round(px, 2).ravel().tolist(), "y": np.round(py, 2).ravel().tolist()},
        "inv": {"step": step, "nx": len(ix), "ny": len(iy),
                "lat": np.round(ilat, 4).ravel().tolist(), "lon": np.round(ilon, 4).ravel().tolist()},
        "places": [{"name": n, "lat": la, "lon": lo, "px": to_px(la, lo)} for n, (la, lo) in PLACES.items()],
        "cities": [{"name": n, "px": to_px(la, lo)} for n, (la, lo) in CITIES.items()],
        "summary": summary,
    }
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{len(layers)} layers, {w}x{h} px, {total / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    raise SystemExit(main())
