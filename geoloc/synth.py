"""Render synthetic field recordings from real flight tracks (test T2 / gate G3).

Every aircraft within audible range is rendered, not only the ones we want to find,
so silences in the synthetic audio are as honest as the flight data allows.
Propellers and rotors get a tone stack (fundamental plus harmonics); jets get a
broadband swell with no clear tone.
"""
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from .physics import emission_to_arrival

F0_RANGE = {"prop": (75.0, 125.0), "turboprop": (100.0, 140.0), "rotor": (18.0, 28.0)}
HARMONICS = {"prop": [1.0, 0.6, 0.4, 0.25], "turboprop": [1.0, 0.5, 0.3],
             "rotor": [0.3, 0.6, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]}


@dataclass
class Source:
    track: object          # physics.Track in the scene's ENU frame
    cls: str
    f0: float = 0.0        # 0 = draw from F0_RANGE
    level: float = 1.0     # amplitude at 1 km
    label: str = ""
    harmonics: list = field(default_factory=list)


def render_scene(sources, observer, t_start, duration, fs=8000, noise_rms=0.02, seed=0, c=343.0,
                 excess_db_per_km=4.0, wind=None):
    """Mix all sources as heard at the observer. Returns (audio, truth list).

    Level falls with spherical spreading (1/d) plus an excess loss per km standing in
    for ground effect, absorption and refraction. Its real value is measured in test T3.
    ``wind`` (ENU m/s) changes the effective sound speed along each ray (G3b).
    """
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = t_start + np.arange(n) / fs
    out = np.zeros(n)
    truth = []
    for src in sources:
        t_emit = np.arange(src.track.t_start, src.track.t_end, 0.05)
        t_arr, ratio, dist = emission_to_arrival(src.track, observer, t_emit, c, wind)
        if t_arr[-1] < t[0] or t_arr[0] > t[-1]:
            continue
        r = np.interp(t, t_arr, ratio, left=np.nan, right=np.nan)
        d = np.interp(t, t_arr, dist, left=np.nan, right=np.nan)
        live = ~np.isnan(r)
        d_safe = np.where(live, d, 1.0)
        amp = np.where(live, src.level * 1000.0 / d_safe * 10 ** (-excess_db_per_km * d_safe / 1000 / 20), 0.0)
        # Fade in and out at the ends of the track so its edges do not click.
        edge = np.convolve(live.astype(float), np.ones(fs) / fs, mode="same")
        amp *= edge
        if src.cls == "jet":
            noise = lfilter([1.0], [1.0, -0.97], rng.standard_normal(n))  # rumbly broadband
            out += 0.3 * amp * noise / np.std(noise)
            f0 = None
        else:
            lo, hi = F0_RANGE[src.cls]
            f0 = src.f0 or rng.uniform(lo, hi)
            harmonics = src.harmonics or HARMONICS[src.cls]
            phase = 2 * np.pi * np.cumsum(np.where(live, f0 * r, 0.0)) / fs
            wobble = 1.0 + 0.1 * np.sin(2 * np.pi * rng.uniform(0.2, 0.7) * t + rng.uniform(0, 6.3))
            for k, a in enumerate(harmonics, 1):
                out += amp * wobble * a * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
        i = int(np.nanargmin(np.where(live, d, np.inf)))
        truth.append({"label": src.label, "cls": src.cls, "f0": f0, "t_closest_heard": float(t[i]),
                      "d_min_m": float(d[i]), "peak_amp": float(amp[i])})
    out += noise_rms * np.cumsum(rng.standard_normal(n)) / np.sqrt(fs) * 0.05  # slow low-frequency drift
    out += noise_rms * rng.standard_normal(n)
    return out / max(1.0, np.max(np.abs(out)) * 1.05), truth


def write_audio(path, audio, fs, mp3_kbps=None):
    """Write WAV, or MP3 at the given bitrate via ffmpeg (to test compression)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not mp3_kbps:
        sf.write(path, audio, fs)
        return path
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "x.wav"
        sf.write(wav, audio, fs)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", f"{mp3_kbps}k", str(path)],
                       check=True)
    return path
