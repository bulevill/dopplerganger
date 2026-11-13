"""G3b: end-to-end test on REAL ADS-B tracks (gate G3 as written, test T2).

For one converted day and one 50 km box, this
  1. picks truth points and recording start times where at least two small aircraft
     (prop / turboprop / rotor) really passed within --max-cpa metres (slant) of the point,
     at CPA-arrival times at least 150 s apart inside the recording;
  2. renders what a recorder on the ground there would have heard from EVERY aircraft
     in the data (jets as broadband rumble), with perturbations the search does not know:
       - sound speed from --temp-c (search assumes 343 m/s),
       - a uniform wind of --wind m/s from a random direction,
       - per-aircraft ADS-B clock error, N(0, --adsb-jitter s),
       - random source levels and f0, white + drift noise, optional real background (--bg-wav),
       - MP3 encoding at --mp3 kbps, decoded again for M1;
  3. runs M1 (audio.detect) and the search (run_search) as black boxes, with the real
     terrain, the loader's recording-window filter and a known start time (as for our own
     recordings; --unknown-t0 to drop it);
  4. scores each run like G3a: PASS = truth inside the 90% region and top peak < 1 km.

    python scripts/g3b_render.py --preset hudson --start-utc 20:15 --end-utc 22:45 --runs 8

Outputs: results/g3b/<preset>_<day>/run_XX/ (search outputs, recording.mp3, g3b_run.json)
and batch_summary.json.
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoloc import audio as audio_mod  # noqa: E402
from geoloc.geodesy import LocalFrame  # noqa: E402
from geoloc.paths import PROCESSED  # noqa: E402
from geoloc.physics import Track  # noqa: E402
from geoloc.search import SearchConfig, load_segments_with_stats, run_search  # noqa: E402
from geoloc.synth import Source, render_scene, write_audio  # noqa: E402
from geoloc.terrain import Terrain  # noqa: E402

PRESETS = {  # same 50 km boxes as scripts/g3b_real.py
    "hudson": (41.65 - 0.225, 41.65 + 0.225, -73.85 - 0.30, -73.85 + 0.30),
    "newyork": (40.75 - 0.225, 40.75 + 0.225, -73.98 - 0.30, -73.98 + 0.30),
}
SMALL = ("prop", "turboprop", "rotor")
C_SEARCH = 343.0


def error_m(lat, lon, lat0, lon0):
    dy = (lat - lat0) * 111_320.0
    dx = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    return float(math.hypot(dx, dy))


def utc(day, hhmm):
    h, m = (int(v) for v in hhmm.split(":"))
    d = datetime.strptime(day, "%Y-%m-%d").replace(hour=h, minute=m, tzinfo=timezone.utc)
    return d.timestamp()


def pick_truths(segs, frame, terrain, box, w0, w1, args, rng):
    """(lat, lon, h_ground, T0, passes) candidates where >= 2 small aircraft pass close by."""
    lat_c, lon_c = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
    half = args.truth_half_km * 1000.0
    xs = np.arange(-half, half + 1, 1000.0)
    X, Y = np.meshgrid(xs, xs)
    lat, lon, _h = frame.to_llh(np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]))
    hg = terrain.height(lat, lon) + 1.5
    obs = frame.to_enu(lat, lon, hg)                                   # (n, 3)
    passes = [[] for _ in range(len(obs))]                             # per point: (t_arr, d, label)
    for s in segs:
        if s.cls not in SMALL:
            continue
        ts = np.arange(s.t[0], s.t[-1], 2.0)
        if len(ts) < 2:
            continue
        P = s.track.position(ts)                                       # (m, 3)
        for a in range(0, len(obs), 256):
            o = obs[a:a + 256]
            d = np.linalg.norm(o[:, None, :] - P[None, :, :], axis=2)  # (k, m)
            j = np.argmin(d, axis=1)
            dm = d[np.arange(len(o)), j]
            # A closest point at a track end (take-off, landing, coverage edge) has no full Doppler curve.
            tj = ts[j]
            full = (tj - s.t[0] >= args.min_side_s) & (s.t[-1] - tj >= args.min_side_s)
            for k in np.flatnonzero((dm <= args.max_cpa) & full):
                passes[a + k].append((float(ts[j[k]] + dm[k] / C_SEARCH), float(dm[k]),
                                      f"{s.icao}:{s.seg}:{s.subseg}", s.cls))
    out = []
    starts = np.arange(w0, w1 - args.duration + 1, 300.0)
    for i, p in enumerate(passes):
        if len(p) < 2:
            continue
        for T0 in starts:
            inside = sorted(q for q in p if T0 + 90 <= q[0] <= T0 + args.duration - 90)
            if len(inside) >= 2 and inside[-1][0] - inside[0][0] >= 150.0:
                out.append((float(lat[i]), float(lon[i]), float(hg[i]), float(T0), inside))
    rng.shuffle(out)
    chosen = []
    for c in out:
        if all(error_m(c[0], c[1], q[0], q[1]) >= args.min_sep_km * 1000 for q in chosen):
            chosen.append(c)
        if len(chosen) == args.runs:
            break
    return chosen, len(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", default="2026-11-06")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="hudson")
    ap.add_argument("--start-utc", default="20:15")
    ap.add_argument("--end-utc", default="22:45")
    ap.add_argument("--duration", type=float, default=1800.0)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-cpa", type=float, default=3000.0, help="slant metres for a qualifying pass")
    ap.add_argument("--truth-half-km", type=float, default=18.0, help="truth points within this of the box centre")
    ap.add_argument("--min-sep-km", type=float, default=4.0)
    ap.add_argument("--min-side-s", type=float, default=60.0,
                    help="ADS-B track must extend this long before and after a qualifying closest approach "
                         "(0 = old behaviour, needed to --reuse-audio batches made before this option)")
    ap.add_argument("--temp-c", type=float, default=10.0, help="air temperature for rendering (search assumes 343 m/s)")
    ap.add_argument("--wind", type=float, default=6.0, help="uniform wind speed, m/s, random direction per run")
    ap.add_argument("--adsb-jitter", type=float, default=0.5, help="per-aircraft ADS-B clock error sigma, s")
    ap.add_argument("--mp3", type=int, default=64, help="MP3 bitrate (0 = keep WAV)")
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--bg-wav", default=None, help="real background recording mixed in (looped)")
    ap.add_argument("--bg-gain", type=float, default=1.0)
    ap.add_argument("--unknown-t0", action="store_true")
    ap.add_argument("--detect-rms", type=float, default=None,
                    help="override audio.detect max_rms_hz (curve-fit tolerance, Hz); default = detector default")
    ap.add_argument("--reuse-audio", default=None,
                    help="batch directory whose run_XX/recording.* to reuse instead of rendering (same seed/settings)")
    ap.add_argument("--d-gate-high-factor", type=float, default=None,
                    help="override SearchConfig.d_gate_high_factor (F4 distance gate high side; 1.0 = symmetric)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    box = PRESETS[args.preset]
    lat_c, lon_c = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
    frame = LocalFrame(lat_c, lon_c, 0.0)
    pts = PROCESSED / "adsblol" / args.day / "points.parquet"
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "g3b" / f"{args.preset}_{args.day}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    c_true = 331.3 * math.sqrt(1 + args.temp_c / 273.15)
    cfg = SearchConfig() if args.d_gate_high_factor is None else \
        SearchConfig(d_gate_high_factor=args.d_gate_high_factor)
    t_all = time.time()

    w0, w1 = utc(args.day, args.start_utc), utc(args.day, args.end_utc)
    segs, st = load_segments_with_stats(pts.as_posix(), frame, *box, t_start=w0, t_end=w1, time_margin_s=60, config=cfg)
    n_small = sum(s.cls in SMALL for s in segs)
    print(f"{args.preset} {args.day} {args.start_utc}-{args.end_utc} UTC: {len(segs)} segments ({n_small} small)")
    terrain = Terrain(box[0] - 0.2, box[1] + 0.2, box[2] - 0.25, box[3] + 0.25)
    chosen, n_cand = pick_truths(segs, frame, terrain, box, w0, w1, args, rng)
    print(f"qualifying (point, start) pairs: {n_cand}; runs: {len(chosen)}")
    if not chosen:
        print("No truth point qualifies: widen the window, the box or --max-cpa.")
        return 1

    bg = None
    if args.bg_wav:
        import librosa
        bg = librosa.load(args.bg_wav, sr=8000, mono=True)[0]

    runs = []
    for i, (tlat, tlon, th, T0, expected) in enumerate(chosen):
        rd = out / f"run_{i:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        rr = np.random.default_rng(args.seed * 1000 + i)
        t_run = time.time()
        # Flights as the search will see them (recording window + margin).
        s_segs, s_st = load_segments_with_stats(pts.as_posix(), frame, *box, t_start=T0, t_end=T0 + args.duration,
                                                time_margin_s=120, config=cfg)
        # The "real world": the same tracks, each shifted by its own ADS-B clock error.
        sources = []
        for s in s_segs:
            jit = rr.normal(0.0, args.adsb_jitter)
            cls = "prop" if s.cls == "unknown" else s.cls
            level = rr.uniform(0.6, 1.5) * (0.7 if s.cls == "unknown" else 1.0)
            sources.append(Source(Track(np.asarray(s.t) + jit, s.enu), cls, level=level,
                                  label=f"{s.icao}:{s.seg}:{s.subseg}:{s.cls}"))
        wdir = rr.uniform(0, 2 * math.pi)
        wind = np.array([args.wind * math.sin(wdir), args.wind * math.cos(wdir), 0.0])
        observer = frame.to_enu(np.array([tlat]), np.array([tlon]), np.array([th]))[0]
        render_seed = int(rr.integers(1 << 30))
        old = Path(args.reuse_audio) / f"run_{i:02d}" if args.reuse_audio else None
        if old is not None:
            # Same seed and settings -> same truths, sources and wind; only re-detect and re-search.
            prev = json.loads((old / "g3b_run.json").read_text())
            assert abs(prev["truth"][0] - tlat) < 1e-9 and prev["t0"] == T0, "reuse dir does not match this seed/settings"
            path = next(old.glob("recording.*"))
            y4 = audio_mod.load(path)
            n_heard = prev["small_within_4km_rendered"]
        else:
            y, synth_truth = render_scene(sources, observer, T0, args.duration, fs=8000, noise_rms=args.noise,
                                          seed=render_seed, c=c_true, wind=wind)
            n_heard = sum(t["cls"] in SMALL and t["d_min_m"] <= 4000.0 for t in synth_truth)
        if old is None and bg is not None:
            reps = int(math.ceil(len(y) / len(bg)))
            b = np.tile(bg, reps)[:len(y)]
            y = y + args.bg_gain * b * (np.std(y) / max(np.std(b), 1e-9))
            y = y / max(1.0, np.max(np.abs(y)) * 1.05)
        if old is not None:
            pass
        elif args.mp3:
            path = write_audio(rd / "recording.mp3", y, 8000, mp3_kbps=args.mp3)
            y4 = audio_mod.load(path)
        else:
            path = write_audio(rd / "recording.wav", y, 8000)
            y4 = audio_mod.load(path)
        perc = audio_mod.detect(y4, **({} if args.detect_rms is None else {"max_rms_hz": args.detect_rms}))
        t_render = time.time() - t_run
        result = run_search(perc, s_segs, box, height_fn=terrain.height,
                            t0_known=None if args.unknown_t0 else T0, config=cfg, out_dir=rd, frame=frame,
                            t_start=T0, t_end=T0 + args.duration, time_margin_s=120)
        lat_m, lon_m, _ = result.peak(which="marg")
        lat_x, lon_x, _ = result.peak(which="max")
        err_m = error_m(lat_m, lon_m, tlat, tlon)
        err_x = error_m(lat_x, lon_x, tlat, tlon)
        in50, in90, in95 = (bool(result.truth_in_region(tlat, tlon, level=q, which="marg")) for q in (0.5, 0.9, 0.95))
        ref = [error_m(r.best_lat, r.best_lon, tlat, tlon) for r in result.refined]
        passed = in90 and err_m < 1000.0
        top_flights = [f"{f['icao']}:{f['seg']}:{f.get('piece', 0)}" for f in (result.modes[0].get("flights", []) if result.modes else [])]
        true_close = [e[2] for e in expected]
        rec = {
            "run": i, "pass": bool(passed), "truth": [tlat, tlon, th], "t0": T0,
            "t0_utc": datetime.fromtimestamp(T0, timezone.utc).strftime("%H:%M"),
            "error_m_marg": err_m, "error_m_max": err_x, "refined_errors_m": ref,
            "truth_in_50_90_95": [in50, in90, in95],
            "expected_close_passes": [{"t_arr_audio": round(e[0] - T0, 1), "d_m": round(e[1]), "flight": e[2], "cls": e[3]} for e in expected],
            "small_within_4km_rendered": n_heard,
            "events_detected": len(perc.events),
            "events": [{"t_c": round(e.t_c, 1), "d": round(e.d), "v": round(e.v, 1), "f0": round(e.f0, 1)} for e in perc.events],
            "silences": len(perc.silences),
            "top_mode_flights": top_flights,
            "top_mode_uses_true_close_flight": bool(set(top_flights) & set(true_close)),
            "segments": len(s_segs), "segments_small": sum(s.cls in SMALL for s in s_segs),
            "c_true": c_true, "wind_enu": wind.tolist(), "mp3_kbps": args.mp3,
            "time_render_detect_s": round(t_render, 1), "time_search_s": result.funnel_counts.get("time_total_s"),
            "funnel": {k: result.funnel_counts.get(k) for k in ("pair_hypotheses", "blobs", "blobs_scored", "cpa_entries")},
        }
        (rd / "g3b_run.json").write_text(json.dumps(rec, indent=2))
        runs.append(rec)
        print(f"run {i:02d} {rec['t0_utc']} UTC  truth {tlat:.4f},{tlon:.4f}  close passes {len(expected)}  "
              f"events {len(perc.events)}  peak err {err_m:6.0f} m  refined {', '.join(f'{e:.0f}' for e in ref[:1]) or '-'} m  "
              f"in90 {in90}  {'PASS' if passed else 'FAIL'}  ({rec['time_search_s']} s)")

    n_pass = sum(r["pass"] for r in runs)
    summary = {
        "day": args.day, "preset": args.preset, "window_utc": [args.start_utc, args.end_utc],
        "settings": {k: v for k, v in vars(args).items() if k not in ("out",)},
        "c_true": c_true, "c_search": C_SEARCH, "qualifying_pairs": n_cand,
        "passed": n_pass, "runs": len(runs),
        "median_peak_error_m": float(np.median([r["error_m_marg"] for r in runs])),
        "median_refined_error_m": float(np.median([r["refined_errors_m"][0] for r in runs if r["refined_errors_m"]] or [np.nan])),
        "runs_detail": runs, "total_s": round(time.time() - t_all, 1),
    }
    (out / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"G3b {args.preset}: {n_pass}/{len(runs)} passed; median peak error {summary['median_peak_error_m']:.0f} m; "
          f"outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
