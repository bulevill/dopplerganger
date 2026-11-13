"""List aircraft passing near a place (city-level centre) in a UTC window, with CPA time/distance.
usage: near_place.py DAY LAT LON RADIUS_KM T0_UTC T1_UTC [MAX_ALT_M]"""
import sys, math
import duckdb, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone
from geoloc.paths import PROCESSED
day, lat0, lon0, rkm, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6]
maxalt = float(sys.argv[7]) if len(sys.argv) > 7 else 4000.0
ts = lambda hm: datetime.strptime(f"{day} {hm}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()
t0, t1 = ts(a), ts(b)
P = (PROCESSED / "adsblol" / day / "points.parquet").as_posix(); A = (PROCESSED / "adsblol" / day / "aircraft.parquet").as_posix()
con = duckdb.connect()
dlat = rkm / 111.32; dlon = rkm / (111.32 * math.cos(math.radians(lat0)))
df = con.sql(f"""SELECT p.icao, p.t, p.lat, p.lon, coalesce(p.alt_geom_m, p.alt_baro_m) AS alt, p.gs_ms, a.cls, a.typecode
 FROM '{P}' p LEFT JOIN '{A}' a USING (icao)
 WHERE p.t BETWEEN {t0} AND {t1} AND p.lat BETWEEN {lat0-dlat} AND {lat0+dlat} AND p.lon BETWEEN {lon0-dlon} AND {lon0+dlon}""").df()
rows = []
for icao, g in df.groupby("icao"):
    g = g.sort_values("t")
    x = (g.lon - lon0) * 111320 * math.cos(math.radians(lat0)); y = (g.lat - lat0) * 111320
    h = np.hypot(x, y); k = int(np.argmin(h))
    if g.alt.iloc[k] > maxalt: continue
    v = float(g.gs_ms.iloc[k]) if np.isfinite(g.gs_ms.iloc[k]) else float("nan")
    rows.append((datetime.fromtimestamp(g.t.iloc[k], timezone.utc).strftime("%H:%M:%S"), icao, g.cls.iloc[k], g.typecode.iloc[k], round(h.iloc[k]), round(g.alt.iloc[k]), round(v)))
for r in sorted(rows): print(*r)
print(len(rows), "aircraft; columns: cpa_utc icao cls type horiz_m alt_m speed_mps")
