"""Gate G1: fetch and read one ground-height tile and one hour of HRRR weather.

Run: python scripts/g1_fetch_terrain_weather.py
Only the needed weather fields are downloaded, using the .idx byte ranges.
"""
import re

import cfgrib
import numpy as np
import rasterio
import requests

from geoloc.paths import RAW, require_data_root
from geoloc.physics import speed_of_sound

DEM_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current/n42w074/USGS_1_n42w074.tif"
HRRR_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20261106/conus/hrrr.t18z.wrfsfcf00.grib2"
HRRR_FIELDS = r":(TMP:2 m above ground|UGRD:10 m above ground|VGRD:10 m above ground|PRES:surface):"
TEST_POINT = (41.65, -73.85)  # lat, lon inside the tile


def download(url, dest, headers=None):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def fetch_grib_fields(url, pattern, dest):
    """Download only the GRIB messages whose .idx line matches the pattern."""
    if dest.exists():
        return dest
    lines = requests.get(url + ".idx", timeout=60).text.strip().splitlines()
    offsets = [int(l.split(":")[1]) for l in lines]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for i, line in enumerate(lines):
            if not re.search(pattern, line):
                continue
            end = offsets[i + 1] - 1 if i + 1 < len(lines) else ""
            r = requests.get(url, headers={"Range": f"bytes={offsets[i]}-{end}"}, timeout=60)
            r.raise_for_status()
            f.write(r.content)
    return dest


def main():
    require_data_root()

    dem = download(DEM_URL, RAW / "dem" / "usgs_1arcsec" / "USGS_1_n42w074.tif")
    with rasterio.open(dem) as src:
        elev = next(src.sample([(TEST_POINT[1], TEST_POINT[0])]))[0]
        print(f"DEM  {dem.name}: {src.width}x{src.height} px, {src.res[0] * 3600:.2f} arc-sec, "
              f"CRS {src.crs}, vertical datum NAVD88 (orthometric)")
        print(f"     elevation at {TEST_POINT}: {elev:.1f} m")

    grib = fetch_grib_fields(HRRR_URL, HRRR_FIELDS, RAW / "hrrr" / "hrrr.20261106.t18z.sfc_subset.grib2")
    print(f"HRRR {grib.name}: {grib.stat().st_size / 1e6:.1f} MB")
    # cfgrib keeps fields at different heights (2 m, 10 m, surface) in separate datasets.
    for ds in cfgrib.open_datasets(grib, backend_kwargs={"indexpath": ""}):
        lat = ds.latitude.values
        lon = ds.longitude.values
        lon = np.where(lon > 180, lon - 360, lon)
        j, i = np.unravel_index(np.argmin((lat - TEST_POINT[0]) ** 2 + (lon - TEST_POINT[1]) ** 2), lat.shape)
        for var in ds.data_vars:
            print(f"     {var:6} at test point: {float(ds[var].values[j, i]):.2f} {ds[var].attrs.get('units', '')}")
        if "t2m" in ds:
            t_c = float(ds.t2m.values[j, i]) - 273.15
            print(f"     speed of sound at 2 m: {float(speed_of_sound(t_c)):.1f} m/s (vs. fixed 343)")


if __name__ == "__main__":
    main()
