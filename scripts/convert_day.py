"""Convert one adsb.lol day to Parquet. Run: python scripts/convert_day.py 2026-11-06"""
import sys
import time

import duckdb

from geoloc.telemetry import convert_day

day = sys.argv[1] if len(sys.argv) > 1 else "2026-11-06"
start = time.time()
out = convert_day(day)
print(f"Converted {day} in {(time.time() - start) / 60:.1f} min -> {out}")

con = duckdb.connect()
# Print as a plain DataFrame: DuckDB's box-drawing table fails on the Windows cp1252 console.
print(con.sql(f"""
    SELECT cls, cls_source, count(*) AS aircraft, sum(points) AS points
    FROM '{(out / "aircraft.parquet").as_posix()}' GROUP BY ALL ORDER BY aircraft DESC
""").df().to_string(index=False))
for f in ("points.parquet", "aircraft.parquet"):
    print(f, f"{(out / f).stat().st_size / 1e6:.0f} MB")
