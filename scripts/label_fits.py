"""Label every loose ridge fit in saved G3b recordings as true/false against ADS-B geometry."""
import sys, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import g3b_render as G
from geoloc import audio as A
from geoloc.geodesy import LocalFrame
from geoloc.search import SearchConfig, load_segments_with_stats

rows = []
for batch in sys.argv[1:]:
    batch = Path(batch)
    preset = "newyork" if "newyork" in batch.name else "hudson"
    box = G.PRESETS[preset]
    frame = LocalFrame(0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3]), 0.0)
    pts = (G.PROCESSED / "adsblol" / "2026-11-06" / "points.parquet").as_posix()
    for rd in sorted(batch.glob("run_*")):
        j = json.loads((rd / "g3b_run.json").read_text())
        T0 = j["t0"]; lat, lon, h = j["truth"]
        segs, _ = load_segments_with_stats(pts, frame, *box, t_start=T0, t_end=T0 + 1800, time_margin_s=120, config=SearchConfig())
        obs = frame.to_enu(np.array([lat]), np.array([lon]), np.array([h]))[0]
        truth = []
        for s in segs:
            if s.cls == "jet":
                continue
            ts = np.arange(s.t[0], s.t[-1], 1.0)
            if len(ts) < 2: continue
            d = np.linalg.norm(s.track.position(ts) - obs, axis=1); k = int(np.argmin(d))
            truth.append((ts[k] + d[k] / 343.0 - T0, d[k], s.cls))
        y = A.load(next(rd.glob("recording.*")))
        spec = A.spectrogram(y)
        for r in A._ridges(spec, 8.0, 8, 20.0):
            out = A._fit(r)
            if out is None: continue
            (t_c, d, v, f0), sig, rms = out
            if (t_c - r["t"][0]) < 5 or (r["t"][-1] - t_c) < 5 or rms > 4.0: continue
            ok = any(abs(tt - t_c) < 15 and 0.5 < d / max(dd, 1) < 2.0 for tt, dd, _ in truth)
            rows.append(dict(batch=batch.name, run=rd.name, t_c=t_c, d=d, v=v, f0=f0, rms=rms, n=len(r["t"]),
                             dur=float(r["t"][-1] - r["t"][0]), lvl=float(np.median(r["lvl"])),
                             sd=float(sig[1]) / max(d, 1), sv=float(sig[2]) / max(v, 1), st=float(sig[0]),
                             side=float(min(t_c - r["t"][0], r["t"][-1] - t_c)), true=bool(ok)))
json.dump(rows, open(Path(__file__).with_name("fits.json"), "w"), indent=1)
print(len(rows), "fits;", sum(r["true"] for r in rows), "true")
