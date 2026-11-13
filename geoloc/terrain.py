"""Module M2b: ground height above the WGS84 ellipsoid for any point.

Uses Copernicus GLO-90 tiles (free, no account) for the search stages. Copernicus heights
are above the EGM2008 geoid, so the geoid height is added to match ADS-B altitudes.
"""
import math

import numpy as np
import pyproj
import rasterio
import requests
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from .paths import RAW

COP90_URL = "https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"
COP90_DIR = RAW / "dem" / "copernicus_glo90"


def _tile_name(lat_floor, lon_floor):
    ns = f"N{lat_floor:02d}" if lat_floor >= 0 else f"S{-lat_floor:02d}"
    ew = f"E{lon_floor:03d}" if lon_floor >= 0 else f"W{-lon_floor:03d}"
    return f"Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM"


def _tile(lat_floor, lon_floor):
    name = _tile_name(lat_floor, lon_floor)
    path = COP90_DIR / f"{name}.tif"
    if not path.exists():
        COP90_DIR.mkdir(parents=True, exist_ok=True)
        r = requests.get(COP90_URL.format(name=name), timeout=120)
        if r.status_code == 404:
            return None  # open ocean: no tile is published
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def geoid_height(lat, lon):
    """EGM2008 geoid height (m) above the WGS84 ellipsoid. Downloads the PROJ grid once."""
    pyproj.network.set_network_enabled(True)
    t = Transformer.from_crs("EPSG:4326+3855", "EPSG:4979", always_xy=True)
    _, _, h = t.transform(np.asarray(lon, float), np.asarray(lat, float), np.zeros(np.shape(lat)))
    return np.asarray(h)


class Terrain:
    """Ground height above the ellipsoid inside a lat/lon box."""

    def __init__(self, lat_min, lat_max, lon_min, lon_max):
        rows = []
        for la in range(math.floor(lat_max), math.floor(lat_min) - 1, -1):  # north to south
            cols = []
            for lo in range(math.floor(lon_min), math.floor(lon_max) + 1):
                path = _tile(la, lo)
                if path is None:
                    cols.append(np.zeros((1200, 1200), np.float32))
                    continue
                with rasterio.open(path) as src:
                    data = src.read(1).astype(np.float32)
                # Tiles above 50 degrees are narrower in longitude; resample to a common width.
                if data.shape[1] != 1200:
                    idx = np.linspace(0, data.shape[1] - 1, 1200).round().astype(int)
                    data = data[:, idx]
                cols.append(data)
            rows.append(np.hstack(cols))
        grid = np.vstack(rows)
        lat_top = math.floor(lat_max) + 1
        lon_left = math.floor(lon_min)
        n_lat, n_lon = grid.shape
        # Each tile is 1200 x 1200 cells covering one degree (3 arc-seconds per cell).
        lats = lat_top - (np.arange(n_lat) + 0.5) / 1200
        lons = lon_left + (np.arange(n_lon) + 0.5) / 1200
        self._ortho = RegularGridInterpolator((lats[::-1], lons), grid[::-1], bounds_error=False, fill_value=None)
        # The geoid varies slowly: sample it on a coarse grid and interpolate.
        glat = np.linspace(lat_min - 0.1, lat_max + 0.1, 12)
        glon = np.linspace(lon_min - 0.1, lon_max + 0.1, 12)
        mlat, mlon = np.meshgrid(glat, glon, indexing="ij")
        self._geoid = RegularGridInterpolator((glat, glon), geoid_height(mlat, mlon), bounds_error=False, fill_value=None)

    def height(self, lat, lon):
        """Ground height above the WGS84 ellipsoid (m)."""
        pts = np.column_stack([np.ravel(lat), np.ravel(lon)])
        return (self._ortho(pts) + self._geoid(pts)).reshape(np.shape(lat))
