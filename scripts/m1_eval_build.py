"""Phase 0 of doppler-curve-detection-claude-code-brief.md: build the labelled sets the M1 harness scores against.

    python scripts/m1_eval_build.py livermore    # held-out: 40 real single flyovers (Pathak, Zenodo 23424278, CC BY 4.0)
    python scripts/m1_eval_build.py g3b          # tuning: 24 G3b renders of real 2026-11-06 tracks
    python scripts/m1_eval_build.py synthetic    # tuning: single passes injected into our own real background audio

Each set is written to results/m1_eval/sets/<name>.json as
    {"name", "fully_labelled", "role", "items": [{"id", "audio", "duration", "passes": [...], "exclude": [...]}]}
with passes {"t_c", "d", "v", "cls", "f0"?, "d_eff"?, "v_eff"?} in audio seconds / metres / m/s. "fully_labelled" says whether
every aircraft in the audio is in "passes" (G3b, synthetic) or only the target flyover (Livermore), so the
harness knows whether unmatched detections are false positives or just unlabelled.
"""
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geoloc import audio as A  # noqa: E402
from geoloc.paths import DATA_ROOT, PROCESSED  # noqa: E402

OUT = ROOT / "results" / "m1_eval" / "sets"
LIVERMORE = DATA_ROOT / "raw" / "livermore"
OWN_AUDIO = DATA_ROOT / "raw" / "own_audio"


def _utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def effective_pass(track, obs, t_c_audio, d, v, T0, t_lo, t_hi, c=343.0):
    """(d, v) of the straight pass whose Doppler curve best fits the real track's noise-free curve.

    A turning or accelerating aircraft sounds like a straight pass at a different (mostly closer)
    distance than its geometric closest approach; this is what a straight-pass detector can measure.
    Fitted over arrival times within max(60 s, 4 d/v) of closest approach, inside the track's span."""
    from scipy.optimize import least_squares
    from geoloc.physics import straight_pass_curve
    te = np.arange(t_lo, t_hi, 0.5)
    if len(te) < 20:
        return None
    r = np.linalg.norm(track.position(te) - obs, axis=1)
    ta = te + r / c - T0
    ratio = 1.0 / (1.0 + np.gradient(r, te) / c)
    m = np.abs(ta - t_c_audio) <= max(60.0, 4.0 * d / max(v, 1.0))
    if m.sum() < 20:
        return None
    T, F = ta[m], ratio[m]
    lo, hi = np.array([T[0] - 60, 30.0, 5.0, 0.5]), np.array([T[-1] + 60, 3e4, 300.0, 2.0])
    x0 = np.clip([t_c_audio, d, v, 1.0], lo + 1e-6, hi - 1e-6)
    fit = least_squares(lambda x: straight_pass_curve(T, *x) - F, x0, bounds=(lo, hi))
    return float(fit.x[1]), float(fit.x[2])


def build_livermore():
    """Truth: ADS-B closest distance and speed from the dataset's table; t_c from a straight-pass fit
    to the hand-traced frequency curve, put on the audio clock by sliding that curve over the
    spectrogram (the dataset gives UTC for the curve, not for the audio start)."""
    import pandas as pd
    from scipy.signal import stft

    table = pd.read_csv(LIVERMORE / "final_results_40_cases.csv")
    items = []
    for row in table.itertuples():
        case_dir = next((LIVERMORE / "extracted").glob(f"V_*/{row.case}"))
        y = A.load(case_dir / "audio.wav")
        fs = A.ANALYSIS_FS
        f, t, Z = stft(y, fs, window="hann", nperseg=2048, noverlap=2048 - 200, boundary=None, padded=False)
        P = 10 * np.log10(np.abs(Z) ** 2 + 1e-20)
        P -= np.median(P, axis=1, keepdims=True)
        curve = pd.read_csv(case_dir / "frequency_vs_time_used.csv")
        cu = np.array([_utc(s) for s in curve.time_utc])
        cf = curve.frequency_hz.to_numpy()
        rel = cu - cu[0]
        fi = np.clip(np.rint(cf / (f[1] - f[0])).astype(int), 0, len(f) - 1)
        best = (-np.inf, 0.0)
        for off in np.arange(-rel[-1], t[-1], 0.05):          # audio time of the curve's first point
            ti = np.rint((rel + off - t[0]) / (t[1] - t[0])).astype(int)
            ok = (ti >= 0) & (ti < len(t))
            if ok.mean() < 0.8:
                continue
            score = float(np.mean(P[fi[ok], ti[ok]]))
            if score > best[0]:
                best = (score, off)
        r = {"t": rel + best[1], "f": cf, "lvl": np.zeros_like(cf)}
        fit = A._fit(r)
        if fit is None:
            print(f"{row.case}: straight fit of the hand curve failed; skipped")
            continue
        (t_c, _d, _v, f_c), _sig, rms = fit
        # Review A3: 38/40 clips align at 4.96-5.02 s (a fixed pre-roll); an outlier offset or a poor
        # hand fit makes this clip's t_c truth doubtful, so it is labelled but not expected.
        low_trust = bool(rms > 5.0 or not 4.9 <= best[1] <= 5.1)
        items.append({
            "id": row.case, "audio": (case_dir / "audio.wav").as_posix(), "duration": len(y) / fs,
            "category": row.category,
            "passes": [{"t_c": float(t_c), "d": float(row.R_ADSB_m), "v": float(row.V_ADSB_at_closest_kmh) / 3.6,
                        "cls": "prop", "f_traced": float(f_c), "f0_table": float(row.MII_f0_Hz),
                        "low_trust": low_trust}],
            "align": {"score_db": best[0], "curve_start_audio_s": best[1], "hand_fit_rms_hz": rms},
            "exclude": [],
        })
        print(f"{row.case}: aligned at {best[1]:.2f} s (mean {best[0]:.1f} dB along curve), t_c {t_c:.1f} s, "
              f"d {row.R_ADSB_m:.0f} m")
    return {"name": "livermore", "role": "held-out", "fully_labelled": False, "items": items}


def build_g3b():
    """Truth: closest approach of every non-jet segment in each run's window, from the converted day."""
    import g3b_render as G
    from geoloc.geodesy import LocalFrame
    from geoloc.search import SearchConfig, load_segments_with_stats

    items = []
    for batch in ("hudson_2026-11-06", "newyork_2026-11-06", "hudson_2026-11-06_v2"):
        preset = "newyork" if "newyork" in batch else "hudson"
        box = G.PRESETS[preset]
        frame = LocalFrame(0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3]), 0.0)
        pts = (PROCESSED / "adsblol" / "2026-11-06" / "points.parquet").as_posix()
        for rd in sorted((ROOT / "results" / "g3b" / batch).glob("run_*")):
            j = json.loads((rd / "g3b_run.json").read_text())
            T0 = j["t0"]
            lat, lon, h = j["truth"]
            segs, _ = load_segments_with_stats(pts, frame, *box, t_start=T0, t_end=T0 + 1800, time_margin_s=120,
                                               config=SearchConfig())
            obs = frame.to_enu(np.array([lat]), np.array([lon]), np.array([h]))[0]
            passes = []
            for s in segs:
                if s.cls == "jet":       # rendered as broadband rumble: not a tonal target
                    continue
                ts = np.arange(s.t[0], s.t[-1], 1.0)
                if len(ts) < 2:
                    continue
                P = s.track.position(ts)
                dist = np.linalg.norm(P - obs, axis=1)
                k = int(np.argmin(dist))
                t_c = ts[k] + dist[k] / 343.0 - T0
                # Review A5: a pass whose closest approach falls just outside the recording is still
                # partly audible inside it; label it (not expected) so detecting it is not a false positive.
                if not -300 <= t_c <= 2100:
                    continue
                outside = not 0 <= t_c <= 1800
                v = float(np.linalg.norm(s.track.velocity(ts[k:k + 1])[0][:2]))
                # A closest point at a track end has no full curve; keep it but mark it.
                edge = min(ts[k] - s.t[0], s.t[-1] - ts[k]) < 30
                eff = effective_pass(s.track, obs, t_c, float(dist[k]), v, T0, s.t[0], s.t[-1])
                passes.append({"t_c": float(t_c), "d": float(dist[k]), "v": v, "cls": s.cls, "track_edge": bool(edge),
                               "outside_window": bool(outside),
                               "d_eff": eff[0] if eff else None, "v_eff": eff[1] if eff else None,
                               "flight": f"{s.icao}:{s.seg}:{s.subseg}"})
            items.append({"id": f"{batch}/{rd.name}", "audio": next(rd.glob("recording.*")).as_posix(),
                          "duration": 1800.0, "passes": passes, "exclude": []})
            print(f"{batch}/{rd.name}: {len(passes)} non-jet passes")
    return {"name": "g3b", "role": "tuning", "fully_labelled": True, "items": items}


BACKGROUNDS = {  # real recordings with no aircraft tone (own recordings, not distributed; bg3 has its one real pass cut out)
    "bg1": ("bg1.wav", []),
    "bg2": ("bg2.wav", []),
    "bg3": ("bg3.wav", [(1100.0, 1350.0)]),
}


def build_synthetic(n_per_band=12, n_null=12, clip_s=300.0, seed=7):
    """Single passes at controlled slant distance, mixed into real background audio.

    Background level: the background is scaled to a fixed RMS (BG_RMS) on the renderer's absolute
    scale (amplitude 1000/d with 4 dB/km excess loss), chosen so a 4-7 km pass is near the per-frame
    threshold. Half the passes have a +-3 m/s speed wobble (real tracks are not perfectly straight).
    """
    import soundfile as sf

    from geoloc.physics import Track
    from geoloc.synth import Source, render_scene

    BG_RMS = 0.05
    rng = np.random.default_rng(seed)
    fs = A.ANALYSIS_FS
    bgs = {}
    for name, (fname, cut) in BACKGROUNDS.items():
        y = A.load(OWN_AUDIO / fname)
        keep = np.ones(len(y), bool)
        for a, b in cut:
            keep[int(a * fs):int(b * fs)] = False
        bgs[name] = y[keep]
    wav_dir = OUT.parent / "synthetic_audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    bands = [(0.5, 2.0), (2.0, 4.0), (4.0, 7.0), (7.0, 12.0)]
    plan = [b for b in bands for _ in range(n_per_band)] + [None] * n_null
    items = []
    for i, band in enumerate(plan):
        name = list(bgs)[i % len(bgs)]
        bg = bgs[name]
        a0 = int(rng.integers(0, len(bg) - int(clip_s * fs)))
        y_bg = bg[a0:a0 + int(clip_s * fs)].astype(float)
        y_bg *= BG_RMS / max(np.std(y_bg), 1e-9)
        passes = []
        y = y_bg
        if band is not None:
            d = float(rng.uniform(*band)) * 1000.0
            alt = float(min(rng.uniform(300.0, 1500.0), 0.9 * d))
            lateral = math.sqrt(max(d ** 2 - alt ** 2, 0.0))
            v = float(rng.uniform(40.0, 80.0))
            t_cpa = float(rng.uniform(90.0, clip_s - 90.0))
            heading = float(rng.uniform(0, 2 * math.pi))
            wobble = i % 2 == 1
            t = np.arange(t_cpa - 600.0, t_cpa + 600.0, 3.0)
            vv = v + (3.0 * np.sin(2 * np.pi * (t - t_cpa) / 45.0) if wobble else np.zeros_like(t))
            x = np.concatenate([[0], np.cumsum(0.5 * (vv[1:] + vv[:-1]) * np.diff(t))])
            x -= np.interp(t_cpa, t, x)
            u = np.array([math.cos(heading), math.sin(heading), 0.0])
            side = np.array([-u[1], u[0], 0.0])
            pos = x[:, None] * u + lateral * side + np.array([0.0, 0.0, alt])
            src = Source(Track(t, pos), "prop", f0=0.0)
            y_ac, truth = render_scene([src], np.zeros(3), 0.0, clip_s, fs=fs, noise_rms=0.0,
                                       seed=int(rng.integers(1 << 30)))
            y = y_bg + y_ac
            passes.append({"t_c": float(truth[0]["t_closest_heard"]), "d": d, "v": v, "cls": "prop",
                           "f0": float(truth[0]["f0"]), "wobble": wobble})
        y = y / max(1.0, np.max(np.abs(y)) * 1.05)
        path = wav_dir / f"syn_{i:03d}.wav"
        sf.write(path, y.astype(np.float32), fs)
        items.append({"id": f"syn_{i:03d}", "audio": path.as_posix(), "duration": clip_s, "background": name,
                      "passes": passes, "exclude": []})
    print(f"synthetic: {sum(1 for p in plan if p)} passes + {n_null} empty clips, background RMS {BG_RMS}")
    return {"name": "synthetic", "role": "tuning", "fully_labelled": True, "bg_rms": BG_RMS, "items": items}


def main(argv):
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    builders = {"livermore": build_livermore, "g3b": build_g3b, "synthetic": build_synthetic}
    for name in argv or builders:
        s = builders[name]()
        (OUT / f"{name}.json").write_text(json.dumps(s, indent=1))
        print(f"wrote {OUT / (name + '.json')}: {len(s['items'])} items")


if __name__ == "__main__":
    main(sys.argv[1:])
