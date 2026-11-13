"""M1 checks on synthetic straight passes with known geometry (default detector: matched filter)."""
import numpy as np
import pytest

from geoloc.audio import ANALYSIS_FS, detect
from geoloc.physics import Track
from geoloc.synth import Source, render_scene


def straight_source(t_cpa, speed, lateral, alt, f0, heading=0.0, cls="prop"):
    d = np.array([np.cos(heading), np.sin(heading), 0.0])
    side = np.array([-d[1], d[0], 0.0])
    t = np.arange(t_cpa - 400, t_cpa + 400, 3.0)  # ADS-B-like 3 s sampling
    pos = (t - t_cpa)[:, None] * speed * d + lateral * side + np.array([0, 0, alt])
    return Source(Track(t, pos), cls, f0=f0)


@pytest.fixture(scope="module")
def scene():
    sources = [straight_source(200, 60, 900, 600, 90.0), straight_source(650, 75, 1800, 900, 110.0, heading=1.0)]
    audio, truth = render_scene(sources, np.zeros(3), 0.0, 900.0, fs=ANALYSIS_FS, noise_rms=0.02)
    return detect(audio, ANALYSIS_FS), sources


def test_finds_both_passes(scene):
    p, _ = scene
    assert len(p.events) == 2


@pytest.mark.parametrize("k, speed, lateral, alt, f0", [(0, 60, 900, 600, 90.0), (1, 75, 1800, 900, 110.0)])
def test_recovers_geometry(scene, k, speed, lateral, alt, f0):
    p, _ = scene
    e = p.events[k]
    d_true = np.hypot(lateral, alt)
    t_true = [200, 650][k] + d_true / 343.0
    assert e.t_c == pytest.approx(t_true, abs=1.0)
    assert e.d == pytest.approx(d_true, rel=0.10)
    assert e.v == pytest.approx(speed, rel=0.05)
    assert e.f0 == pytest.approx(f0, abs=1.0)
    assert 1 in e.harmonics and 2 in e.harmonics


def test_finds_silence_between_passes(scene):
    p, _ = scene
    assert any(s < 400 and e > 450 for s, e in p.silences)


def test_wobbly_pass_kept_as_loose():
    # Real tracks wobble: a +-3 m/s speed oscillation breaks the 0.5 Hz tight fit,
    # but the long, well-constrained ridge must still come through as a "loose" event.
    t = np.arange(-200, 600, 3.0)
    v = 60 + 3.0 * np.sin(2 * np.pi * (t - 200) / 45.0)
    x = np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    x -= np.interp(200.0, t, x)
    pos = np.column_stack([x, np.full_like(t, 900.0), np.full_like(t, 600.0)])
    audio, _ = render_scene([Source(Track(t, pos), "prop", f0=90.0)], np.zeros(3), 0.0, 400.0,
                            fs=ANALYSIS_FS, noise_rms=0.02)
    assert detect(audio, ANALYSIS_FS, method="ridge", loose_rms_hz=None).events == []
    (e,) = detect(audio, ANALYSIS_FS, method="ridge").events
    d_true = np.hypot(900, 600)
    assert e.quality == "loose"
    assert e.t_c == pytest.approx(200 + d_true / 343.0, abs=2.0)
    assert e.d == pytest.approx(d_true, rel=0.10)
    assert e.v == pytest.approx(60, rel=0.05)


def test_straight_passes_are_tight():
    sources = [straight_source(200, 60, 900, 600, 90.0), straight_source(650, 75, 1800, 900, 110.0, heading=1.0)]
    audio, _ = render_scene(sources, np.zeros(3), 0.0, 900.0, fs=ANALYSIS_FS, noise_rms=0.02)
    p = detect(audio, ANALYSIS_FS, method="ridge")
    assert len(p.events) == 2 and all(e.quality == "tight" for e in p.events)
