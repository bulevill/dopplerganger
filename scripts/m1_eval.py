"""Phase 0 of doppler-curve-detection-claude-code-brief.md: score an M1 detector against the labelled sets.

    python scripts/m1_eval.py --detector two_tier            # all sets in results/m1_eval/sets/
    python scripts/m1_eval.py --detector tight --sets livermore

A detection matches a labelled pass when its closest-approach time is within --tol s and its
straight-pass fit is physically consistent with the aircraft (distance within a factor 3, speed
within -40 %/+60 %: straight fits of real passes are rough). Admissible pairs are assigned one-to-one
by minimum combined cost (time, distance and speed errors together; review item A1: closest-in-time-first
let a marginal match steal an event from a good one). The admissibility rule follows doppler-locate's
scripts/eval_detect.py (MIT); the code here is our own.

A match is also "search-usable" when it passes the search's own gate (search.py F4):
|d_det - d_true| <= max(0.20 d_det, 3 sigma_d) and the same for v. Recall is reported both ways.

Reported per set and overall:
  - recall by slant-distance band (<2, 2-4, 4-7, 7-12 km); passes beyond 12 km or with their closest
    point at a track end are not counted as expected, and a detection matching one is not a false positive;
  - false positives per hour, only for fully labelled sets (G3b, synthetic); for Livermore, where only
    the target flyover is labelled, unmatched detections are reported but not called false;
  - median errors on matched detections: |t_c|, d and v (relative), f0 (relative to the nearest harmonic
    of the labelled tone), and 3-sigma coverage of d and v (honest sigmas cover ~99 %);
  - chance baseline: when the median recording is >= 900 s, the same matching with detection times
    shifted by +-200..600 s; otherwise (Livermore, synthetic) each clip's detections against other
    clips' labels (cyclic shifts 1..8). Matches minus chance is the real signal.
Passes flagged low_trust (Livermore: poor hand fit or odd alignment) or outside_window (G3b: closest
approach just outside the recording, tone partly inside) are labelled but not expected: a detection of
one is neither a hit nor a false positive.
Detections are cached in results/m1_eval/<detector>/detections/<key>/, where key hashes the detector
kwargs and the audio.py source, so a code or setting change never reuses stale detections (review A7).
"""
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoloc import audio as A  # noqa: E402

EVAL = ROOT / "results" / "m1_eval"
BANDS = [(0, 2000), (2000, 4000), (4000, 7000), (7000, 12000)]
BAND_NAMES = ["<2 km", "2-4 km", "4-7 km", "7-12 km"]

DETECTORS = {  # name -> kwargs for audio.detect
    "tight": {"method": "ridge", "loose_rms_hz": None},
    "two_tier": {"method": "ridge"},
    "matched": {"method": "matched"},
}


def band_of(d):
    for k, (a, b) in enumerate(BANDS):
        if a <= d < b:
            return k
    return None


def expected(p):
    return (band_of(p["d"]) is not None and not p.get("track_edge", False) and not p.get("low_trust", False)
            and not p.get("outside_window", False))


def cache_key(name):
    h = hashlib.sha1(json.dumps(DETECTORS[name], sort_keys=True).encode())
    h.update((ROOT / "geoloc" / "audio.py").read_bytes())
    return h.hexdigest()[:10]


def run_detector(name, item, cache_dir):
    f = cache_dir / (item["id"].replace("/", "__") + ".json")
    if f.exists():
        return json.loads(f.read_text())
    y = A.load(item["audio"])
    t0 = time.time()
    P = A.detect(y, **DETECTORS[name])
    def row(e):
        return {"t_c": e.t_c, "d": e.d, "v": e.v, "f0": e.f0, "sigma_t": e.sigma_t, "sigma_d": e.sigma_d,
                "sigma_v": e.sigma_v, "harmonics": e.harmonics, "span": list(e.span), "quality": e.quality,
                "score": e.score, "contrast": e.contrast, "fit_rel_rms": e.fit_rel_rms}

    out = {"seconds": time.time() - t0, "silences": P.silences,
           "events": [row(e) for e in P.events], "candidates": [row(e) for e in P.candidates]}
    f.write_text(json.dumps(out))
    return out


def match(events, passes, tol, shift=0.0):
    """One-to-one matching of admissible pairs by minimum total cost; returns [(event index, pass index)]."""
    from scipy.optimize import linear_sum_assignment
    if not events or not passes:
        return []
    BIG = 1e6
    C = np.full((len(events), len(passes)), BIG)
    for i, e in enumerate(events):
        for j, p in enumerate(passes):
            dt = abs(e["t_c"] + shift - p["t_c"])
            rd = e["d"] / max(p["d"], 1.0)
            rv = e["v"] / max(p["v"], 1.0)
            if dt <= tol and 1 / 3 <= rd <= 3 and 0.6 <= rv <= 1.6:
                C[i, j] = (dt / tol) ** 2 + (np.log(rd) / np.log(3)) ** 2 + (np.log(rv) / np.log(1.6)) ** 2
    rows, cols = linear_sum_assignment(C)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if C[i, j] < BIG]


def usable(e, p, rel=0.20):
    """Would the search's F4 gate accept this event for this aircraft?"""
    return (abs(e["d"] - p["d"]) <= max(rel * e["d"], 3 * e.get("sigma_d", 0.0)) and
            abs(e["v"] - p["v"]) <= max(rel * e["v"], 3 * e.get("sigma_v", 0.0)))


def harmonic_rel_err(f_det, f_lab):
    """Relative f0 error allowing the detector's lowest tone to be a different harmonic of the labelled one."""
    best = np.inf
    for a in range(1, 6):
        for b in range(1, 6):
            best = min(best, abs(f_det * a / (f_lab * b) - 1.0))
    return best


def score_set(s, det_name, tol):
    cache = EVAL / det_name / "detections" / cache_key(det_name)
    cache.mkdir(parents=True, exist_ok=True)
    n_exp = np.zeros(len(BANDS), int)
    n_hit = np.zeros(len(BANDS), int)
    n_use = np.zeros(len(BANDS), int)
    errs = {"t": [], "d": [], "v": [], "f0": [], "d_eff": [], "v_eff": []}
    n_use_eff = n_has_eff = 0
    cov_d, cov_v = [], []
    n_det = n_fp = n_unmatched = 0
    hours = 0.0
    per_item = []
    dets_all = []
    for item in s["items"]:
        r = run_detector(det_name, item, cache)
        ev = r["events"]
        dets_all.append(ev)
        hours += item["duration"] / 3600.0
        n_det += len(ev)
        m = match(ev, item["passes"], tol)
        hit_p = {j for _, j in m}
        hit_e = {i for i, _ in m}
        use_p = {j for i, j in m if usable(ev[i], item["passes"][j])}
        for i, j in m:
            p = item["passes"][j]
            if expected(p) and p.get("d_eff"):
                # Against what the sound implies (straight-pass fit to the real track's curve): detector
                # accuracy, as opposed to search-usability against the geometric closest approach.
                n_has_eff += 1
                n_use_eff += usable(ev[i], {"d": p["d_eff"], "v": p["v_eff"]})
                errs["d_eff"].append(abs(ev[i]["d"] / p["d_eff"] - 1))
                errs["v_eff"].append(abs(ev[i]["v"] / p["v_eff"] - 1))
        for j, p in enumerate(item["passes"]):
            if expected(p):
                b = band_of(p["d"])
                n_exp[b] += 1
                n_hit[b] += j in hit_p
                n_use[b] += j in use_p
        for i, j in m:
            e, p = ev[i], item["passes"][j]
            cov_d.append(abs(e["d"] - p["d"]) <= 3 * e.get("sigma_d", 0.0))
            cov_v.append(abs(e["v"] - p["v"]) <= 3 * e.get("sigma_v", 0.0))
            errs["t"].append(abs(e["t_c"] - p["t_c"]))
            errs["d"].append(abs(e["d"] / p["d"] - 1))
            errs["v"].append(abs(e["v"] / p["v"] - 1))
            f_lab = p.get("f0", p.get("f_traced"))
            if f_lab:
                errs["f0"].append(harmonic_rel_err(e["f0"], f_lab))
        unmatched = len(ev) - len(hit_e)
        n_unmatched += unmatched
        if s["fully_labelled"]:
            n_fp += unmatched
        per_item.append({"id": item["id"], "events": len(ev), "matched": len(m),
                         "expected": sum(expected(p) for p in item["passes"]),
                         "expected_hit": sum(expected(item["passes"][j]) for j in hit_p),
                         "missed": [{"t_c": round(p["t_c"], 1), "d": round(p["d"]), "v": round(p["v"], 1), "cls": p["cls"]}
                                    for j, p in enumerate(item["passes"]) if expected(p) and j not in hit_p],
                         "seconds": round(r["seconds"], 1)})
    # Chance: same events against labels they cannot belong to.
    chance = []
    if np.median([it["duration"] for it in s["items"]]) >= 900:
        for shift in (-600, -450, -300, -200, 200, 300, 450, 600):
            chance.append(sum(len(match(ev, it["passes"], tol, shift)) for ev, it in zip(dets_all, s["items"])))
    else:
        n = len(s["items"])
        for k in range(1, min(n, 9)):
            chance.append(sum(len(match(dets_all[i], s["items"][(i + k) % n]["passes"], tol)) for i in range(n)))
    n_match_total = sum(p["matched"] for p in per_item)
    return {
        "set": s["name"], "role": s["role"], "fully_labelled": s["fully_labelled"], "items": len(s["items"]),
        "audio_hours": round(hours, 2), "detections": n_det,
        "recall_by_band": {BAND_NAMES[k]: [int(n_hit[k]), int(n_exp[k])] for k in range(len(BANDS))},
        "recall_all": [int(n_hit.sum()), int(n_exp.sum())],
        "usable_by_band": {BAND_NAMES[k]: [int(n_use[k]), int(n_exp[k])] for k in range(len(BANDS))},
        "usable_all": [int(n_use.sum()), int(n_exp.sum())],
        "coverage_3sigma": {"d": _frac(cov_d), "v": _frac(cov_v)},
        "usable_vs_effective": [n_use_eff, n_has_eff] if n_has_eff else None,
        "matched_total": n_match_total,
        "chance_matches_mean": float(np.mean(chance)) if chance else None,
        "false_positives_per_hour": round(n_fp / hours, 1) if s["fully_labelled"] else None,
        "unmatched_detections": n_unmatched,
        "median_errors": {"t_c_s": _med(errs["t"]), "d_rel": _med(errs["d"]), "v_rel": _med(errs["v"]),
                          "f0_rel": _med(errs["f0"]),
                          "d_rel_vs_effective": _med(errs["d_eff"]), "v_rel_vs_effective": _med(errs["v_eff"])},
        "per_item": per_item,
    }


def _med(x):
    return round(float(np.median(x)), 3) if x else None


def _frac(x):
    return round(float(np.mean(x)), 2) if x else None


def table(results, det_name):
    lines = [f"### Detector `{det_name}` (cache {cache_key(det_name)})", "",
             "Recall cells: harmonic-matched / search-usable / expected.", "",
             "| Set | Role | " + " | ".join(BAND_NAMES) + " | All | Matches (chance) | FP/h | Unmatched | "
             "Median abs t_c err (s) | d err | v err | f0 err | 3-sigma cover d / v |",
             "|" + "---|" * (len(BANDS) + 11)]
    for r in results:
        rb = " | ".join(f"{h}/{u}/{n}" if n else "–" for (h, n), (u, _) in
                        zip(r["recall_by_band"].values(), r["usable_by_band"].values()))
        me = r["median_errors"]
        cv = r["coverage_3sigma"]
        ch = f"{r['chance_matches_mean']:.1f}" if r["chance_matches_mean"] is not None else "–"
        lines.append(f"| {r['set']} | {r['role']} | {rb} | {r['recall_all'][0]}/{r['usable_all'][0]}/{r['recall_all'][1]} | "
                     f"{r['matched_total']} ({ch}) | {r['false_positives_per_hour'] if r['false_positives_per_hour'] is not None else '–'} | "
                     f"{r['unmatched_detections']} | {me['t_c_s']} | {me['d_rel']} | {me['v_rel']} | {me['f0_rel']} | "
                     f"{cv['d']} / {cv['v']} |")
    eff = [r for r in results if r.get("usable_vs_effective")]
    if eff:
        lines += ["", "Against the sound-implied geometry (straight-pass fit to each real track's own Doppler curve; "
                  "turning tracks sound closer than their geometric closest approach):", ""]
        for r in eff:
            u, n = r["usable_vs_effective"]
            me = r["median_errors"]
            lines.append(f"- {r['set']}: usable {u}/{n} matched expected passes; median d err {me['d_rel_vs_effective']}, "
                         f"v err {me['v_rel_vs_effective']}")
    return "\n".join(lines)


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=sorted(DETECTORS), default="two_tier")
    ap.add_argument("--sets", default="livermore,synthetic,g3b")
    ap.add_argument("--tol", type=float, default=15.0, help="closest-approach time tolerance, s")
    args = ap.parse_args(argv)
    results = []
    for name in args.sets.split(","):
        s = json.loads((EVAL / "sets" / f"{name}.json").read_text())
        t0 = time.time()
        r = score_set(s, args.detector, args.tol)
        r["tol_s"] = args.tol
        results.append(r)
        print(f"{name}: recall {r['recall_all'][0]}/{r['recall_all'][1]}, matches {r['matched_total']} "
              f"(chance {r['chance_matches_mean']}), FP/h {r['false_positives_per_hour']} ({time.time() - t0:.0f} s)")
    out = EVAL / args.detector
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(results, indent=1))
    md = table(results, args.detector)
    (out / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
