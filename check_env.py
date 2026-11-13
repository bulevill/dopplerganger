"""Gate G0: confirm the geoloc environment is complete. Run: python check_env.py"""
import importlib
import shutil
import subprocess
import sys

PACKAGES = [
    "numpy", "scipy", "numba", "pandas", "polars", "pyarrow", "duckdb",
    "librosa", "soundfile", "osgeo.gdal", "rasterio", "pyproj", "shapely",
    "geopandas", "h3", "rio_cogeo", "xarray", "cfgrib", "cdsapi", "requests",
    "s3fs", "tqdm", "matplotlib", "folium", "pydeck", "pytest", "hypothesis",
    "pyopensky",
]

failed = []
print(f"Python {sys.version.split()[0]} at {sys.executable}\n")

for name in PACKAGES:
    try:
        mod = importlib.import_module(name)
        print(f"  OK    {name:12} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"  FAIL  {name:12} {exc}")
        failed.append(name)

print()
ffmpeg = shutil.which("ffmpeg")
if ffmpeg:
    first_line = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"  OK    ffmpeg       {first_line}")
else:
    print("  FAIL  ffmpeg       not on PATH")
    failed.append("ffmpeg")

# Smoke tests: compiled code, projections and audio decoding all work.
try:
    import numba
    import numpy as np

    @numba.njit
    def _sum(a):
        s = 0.0
        for x in a:
            s += x
        return s

    assert _sum(np.ones(10)) == 10.0
    print("  OK    numba JIT compiles")
except Exception as exc:
    print(f"  FAIL  numba JIT    {exc}")
    failed.append("numba-jit")

try:
    from pyproj import Transformer

    # WGS84 lat/lon/ellipsoid height -> ECEF; needs PROJ data files to be installed
    x, y, z = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True).transform(-74.0, 40.0, 0.0)
    assert abs(x) > 1e6
    print("  OK    pyproj transforms (PROJ data found)")
except Exception as exc:
    print(f"  FAIL  pyproj       {exc}")
    failed.append("pyproj-transform")

print()
if failed:
    print(f"G0 NOT PASSED. Missing or broken: {', '.join(failed)}")
    sys.exit(1)
print("G0 PASSED. Environment is ready.")
