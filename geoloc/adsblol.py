"""Read adsb.lol globe_history daily archives (readsb trace_full format) without extracting them.

Each archive member traces/xx/trace_full_<icao>.json is gzipped JSON with aircraft
metadata and a "trace" list. Trace row fields (readsb):
  0 seconds after "timestamp"   1 lat   2 lon   3 alt_baro ft or "ground"
  4 ground speed kt   5 track deg   6 flags   7 baro rate ft/min
  8 details dict or None   9 source type   10 alt_geom ft   11 geom rate ft/min
  12 IAS kt   13 roll deg
"""
import gzip
import io
import json
import tarfile
from pathlib import Path

from .paths import REFERENCE

TRACE_FIELDS = ["dt", "lat", "lon", "alt_baro", "gs", "track", "flags", "baro_rate",
                "details", "source", "alt_geom", "geom_rate", "ias", "roll"]


class _ConcatReader(io.RawIOBase):
    """Read split archive parts (.tar.aa, .tar.ab, ...) as one stream."""

    def __init__(self, paths):
        self._files = [open(p, "rb") for p in paths]
        self._i = 0

    def readable(self):
        return True

    def readinto(self, buf):
        while self._i < len(self._files):
            n = self._files[self._i].readinto(buf)
            if n:
                return n
            self._files[self._i].close()
            self._i += 1
        return 0


def day_parts(day_dir):
    parts = sorted(Path(day_dir).glob("*.tar*"))
    parts = [p for p in parts if p.suffix != ".txt"]
    if not parts:
        raise FileNotFoundError(f"No archive parts in {day_dir}")
    return parts


def iter_traces(day_dir):
    """Yield one decoded trace dict per aircraft from a day's archive."""
    stream = io.BufferedReader(_ConcatReader(day_parts(day_dir)), buffer_size=8 << 20)
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            if not member.isfile() or "/traces/" not in "/" + member.name.lstrip("./"):
                continue
            raw = tar.extractfile(member).read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            yield json.loads(raw)


def load_type_table(path=None):
    """ICAO type designator -> (name, ICAO description code like 'L1P', wake category)."""
    path = Path(path or REFERENCE / "icao_aircraft_types2.js")
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def acoustic_class(desc_code):
    """Classify by ICAO description code: 'prop', 'turboprop', 'rotor', 'jet' or 'unknown'.

    The code is <kind><engine count><engine type>, e.g. L1P, L2T, H1T, L2J.
    """
    if not desc_code or len(desc_code) < 3:
        return "unknown"
    kind, engine = desc_code[0], desc_code[-1]
    if kind in "HG":  # helicopter, gyrocopter
        return "rotor"
    if engine == "P":
        return "prop"
    if engine == "T":
        return "turboprop"
    if engine == "J":
        return "jet"
    return "unknown"  # electric, rocket, or unlisted
