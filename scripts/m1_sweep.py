"""Choose matched-detector acceptance thresholds on the TUNING sets only (synthetic, g3b).

Re-grades the cached matched-detector output (events + listed candidates, each with its contrast
and fit rms) under a grid of (min_contrast, max_rel_rms) and reports recall by band and false
positives per hour, so the thresholds are picked without touching the held-out Livermore set.
Approximate: during detection, accepted families are masked out with the built-in thresholds.

    python scripts/m1_sweep.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from m1_eval import BAND_NAMES, BANDS, EVAL, band_of, cache_key, expected, match  # noqa: E402


def regrade(rows, min_contrast, max_rel, min_side=5.0):
    out = []
    for e in rows:
        if e["quality"] == "ground":
            continue
        side = min(e["t_c"] - e["span"][0], e["span"][1] - e["t_c"])
        if e["contrast"] >= min_contrast and e["fit_rel_rms"] <= max_rel and side >= min_side and \
                e["sigma_d"] / max(e["d"], 1.0) <= 0.5 and 10.5 < e["v"] < 259.5:
            out.append(e)
    return out


def main():
    sets = {n: json.loads((EVAL / "sets" / f"{n}.json").read_text()) for n in ("synthetic", "g3b")}
    det = {}
    for n, s in sets.items():
        for it in s["items"]:
            f = EVAL / "matched" / "detections" / cache_key("matched") / (it["id"].replace("/", "__") + ".json")
            r = json.loads(f.read_text())
            det[(n, it["id"])] = r["events"] + r["candidates"]
    print("min_con max_rel | " + " | ".join(f"{n}: " + " ".join(BAND_NAMES) + " FP/h" for n in sets))
    for mc in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for mr in (0.012,):
            cells = []
            for n, s in sets.items():
                hit = np.zeros(len(BANDS), int)
                exp = np.zeros(len(BANDS), int)
                fp, hours = 0, 0.0
                for it in s["items"]:
                    ev = regrade(det[(n, it["id"])], mc, mr)
                    m = match(ev, it["passes"], 15.0)
                    hp = {j for _, j in m}
                    for j, p in enumerate(it["passes"]):
                        if expected(p):
                            exp[band_of(p["d"])] += 1
                            hit[band_of(p["d"])] += j in hp
                    fp += len(ev) - len(m)
                    hours += it["duration"] / 3600
                cells.append(" ".join(f"{h}/{e}" for h, e in zip(hit, exp)) + f" {fp / hours:.1f}")
            print(f"{mc:7.1f} {mr:7.3f} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
