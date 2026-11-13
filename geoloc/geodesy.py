"""Convert between latitude/longitude/height and a local east-north-up (ENU) frame in metres.

Heights are above the WGS84 ellipsoid (what ADS-B geometric altitude reports).
"""
import numpy as np
from pyproj import Transformer

_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
_TO_LLH = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)


class LocalFrame:
    def __init__(self, lat0, lon0, h0=0.0):
        self.lat0, self.lon0, self.h0 = lat0, lon0, h0
        self.origin = np.array(_TO_ECEF.transform(lon0, lat0, h0))
        phi, lam = np.radians(lat0), np.radians(lon0)
        self._rot = np.array([
            [-np.sin(lam), np.cos(lam), 0.0],
            [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
            [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
        ])

    def to_enu(self, lat, lon, h):
        x, y, z = _TO_ECEF.transform(np.asarray(lon, float), np.asarray(lat, float), np.asarray(h, float))
        return (np.column_stack([np.ravel(x), np.ravel(y), np.ravel(z)]) - self.origin) @ self._rot.T

    def to_llh(self, enu):
        ecef = np.atleast_2d(enu) @ self._rot + self.origin
        lon, lat, h = _TO_LLH.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
        return lat, lon, h
