"""Where project data lives. Bulky data goes on the external drive, code stays on C:.

Override with the GEOLOC_DATA environment variable (e.g. if the drive letter changes).
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("GEOLOC_DATA", "D:/geoloc-data"))
RAW = DATA_ROOT / "raw"
REFERENCE = DATA_ROOT / "reference"
PROCESSED = DATA_ROOT / "processed"
REPORTS = DATA_ROOT / "reports"


def require_data_root():
    """Fail early with a clear message when the external drive is not plugged in."""
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Data folder {DATA_ROOT} not found. Plug in the external drive, "
            "or set GEOLOC_DATA to where the data is."
        )
    return DATA_ROOT
