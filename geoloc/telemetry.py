"""Module M2a: turn a raw adsb.lol day into compact Parquet tables for processing.

Output (under PROCESSED/adsblol/<day>/):
  points.parquet    airborne points below max_alt_ft inside the bounding box, all aircraft
                    classes (jets are kept for the silence check)
  aircraft.parquet  one row per aircraft: type code, acoustic class and how it was decided
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .adsblol import acoustic_class, iter_traces, load_type_table
from .paths import PROCESSED, RAW, require_data_root

CONUS = (24.0, 50.0, -125.0, -66.0)  # lat_min, lat_max, lon_min, lon_max
FT = 0.3048
KT = 0.514444
SEGMENT_GAP_S = 60.0  # a longer silence in the track starts a new segment
POSITIVE_CLASSES = ("prop", "turboprop", "rotor")

POINT_SCHEMA = pa.schema([
    ("icao", pa.string()),
    ("seg", pa.int16()),
    ("t", pa.float64()),           # unix seconds
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("alt_geom_m", pa.float32()),  # GNSS height above the WGS84 ellipsoid; NaN if missing
    ("alt_baro_m", pa.float32()),  # pressure altitude, standard atmosphere
    ("gs_ms", pa.float32()),
    ("track_deg", pa.float32()),
    ("source", pa.dictionary(pa.int8(), pa.string())),
    ("cls", pa.dictionary(pa.int8(), pa.string())),
])


def classify(trace, types):
    """Acoustic class plus how it was decided: 'typecode', 'category' or 'none'.

    Aircraft without a type code fall back to the emitter category they broadcast:
    A7 is rotorcraft, A1 is light aircraft (< 15,500 lb, nearly always piston).
    """
    typecode = trace.get("t")
    if typecode and typecode in types:
        cls = acoustic_class(types[typecode][1])
        if cls != "unknown":
            return cls, "typecode", None
    category = next((row[8].get("category") for row in trace.get("trace") or []
                     if row[8] and row[8].get("category")), None)
    if category == "A7":
        return "rotor", "category", category
    if category == "A1":
        return "prop", "category", category
    return "unknown", "none", category


def _num(x):
    return np.nan if x is None else x


def convert_day(day, bbox=CONUS, max_alt_ft=10_000, batch_rows=2_000_000):
    require_data_root()
    types = load_type_table()
    out_dir = PROCESSED / "adsblol" / day
    out_dir.mkdir(parents=True, exist_ok=True)
    lat_min, lat_max, lon_min, lon_max = bbox

    cols = {name: [] for name in POINT_SCHEMA.names}
    aircraft = []
    writer = pq.ParquetWriter(out_dir / "points.parquet", POINT_SCHEMA, compression="zstd")

    def flush():
        if cols["t"]:
            writer.write_table(pa.table(cols, schema=POINT_SCHEMA))
            for v in cols.values():
                v.clear()

    for trace in iter_traces(RAW / "adsblol" / day):
        icao = trace["icao"]
        cls, cls_source, category = classify(trace, types)
        t0 = trace["timestamp"]
        seg, last_t, kept = 0, None, 0
        for row in trace.get("trace") or []:
            alt = row[3]
            if alt == "ground" or alt is None or alt >= max_alt_ft:
                continue
            lat, lon = row[1], row[2]
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue
            t = t0 + row[0]
            if last_t is not None and t - last_t > SEGMENT_GAP_S:
                seg += 1
            last_t = t
            cols["icao"].append(icao)
            cols["seg"].append(seg)
            cols["t"].append(t)
            cols["lat"].append(lat)
            cols["lon"].append(lon)
            cols["alt_geom_m"].append(_num(row[10]) * FT)
            cols["alt_baro_m"].append(alt * FT)
            cols["gs_ms"].append(_num(row[4]) * KT)
            cols["track_deg"].append(_num(row[5]))
            cols["source"].append(row[9] or "unknown")
            cols["cls"].append(cls)
            kept += 1
        if kept:
            aircraft.append({"icao": icao, "typecode": trace.get("t"), "registration": trace.get("r"),
                             "desc": trace.get("desc"), "category": category, "cls": cls,
                             "cls_source": cls_source, "points": kept, "segments": seg + 1})
        if len(cols["t"]) >= batch_rows:
            flush()
    flush()
    writer.close()
    pq.write_table(pa.Table.from_pylist(aircraft), out_dir / "aircraft.parquet", compression="zstd")
    return out_dir
