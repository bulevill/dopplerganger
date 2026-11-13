"""Gate G2 (test T1): the forward model matches independent physics to < 0.1 Hz."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy.optimize import brentq

from geoloc.physics import (
    Track, closest_approach, distance_from_slope, f0_from_doppler, heard_curve,
    speed_from_doppler, speed_of_sound, straight_pass_curve,
)

C = 343.0
F0 = 80.0  # Cessna 172 blade-pass frequency: 2400 RPM / 60 * 2 blades
TOL_HZ = 0.1


def straight_line_truth(t_heard, p0, v, observer, f0, c=C):
    """Closed-form heard frequency for constant-velocity flight p(t) = p0 + v t.

    Solves |p(t) - o| = c (t_h - t) for the emission time t, a quadratic in t.
    """
    q = p0 - observer
    a = v @ v - c * c
    b = 2.0 * (q @ v + c * c * t_heard)
    cc = q @ q - c * c * t_heard**2
    disc = np.sqrt(b * b - 4.0 * a * cc)
    t_emit = np.minimum((-b + disc) / (2.0 * a), (-b - disc) / (2.0 * a))  # retarded root
    r = observer - (p0 + np.outer(t_emit, v))
    u = r / np.linalg.norm(r, axis=1)[:, None]
    return f0 / (1.0 - (u @ v) / c)


def sampled_track(fn, t0, t1, interval):
    t = np.arange(t0, t1 + interval / 2, interval)
    return Track(t, fn(t))


def straight(p0, v):
    return lambda t: p0 + np.outer(t, v)


def test_speed_of_sound():
    assert speed_of_sound(0.0) == pytest.approx(331.3)
    assert speed_of_sound(20.0) == pytest.approx(343.2, abs=0.1)


def test_straight_pass_matches_closed_form():
    p0, v = np.array([-6000.0, 800.0, 500.0]), np.array([60.0, 0.0, 0.0])
    obs = np.zeros(3)
    track = sampled_track(straight(p0, v), 0, 200, 1.0)
    t_h = np.linspace(30, 190, 2000)
    err = np.abs(heard_curve(track, obs, t_h, F0) - straight_line_truth(t_h, p0, v, obs, F0))
    assert np.nanmax(err) < TOL_HZ


@settings(max_examples=30, deadline=None)
@given(
    speed=st.floats(30, 120),
    heading=st.floats(0, 2 * np.pi),
    lateral=st.floats(50, 5000),
    altitude=st.floats(150, 3000),
)
def test_straight_pass_any_geometry(speed, heading, lateral, altitude):
    d = np.array([np.cos(heading), np.sin(heading), 0.0])
    side = np.array([-d[1], d[0], 0.0])
    v = speed * d
    p0 = -100.0 * v + lateral * side + np.array([0, 0, altitude])
    obs = np.zeros(3)
    track = sampled_track(straight(p0, v), 0, 200, 1.0)
    t_h = np.linspace(40, 180, 500)
    err = np.abs(heard_curve(track, obs, t_h, F0) - straight_line_truth(t_h, p0, v, obs, F0))
    assert np.nanmax(err) < TOL_HZ


@pytest.mark.parametrize("interval", [1.0, 2.0])
def test_turning_flight_matches_root_solve(interval):
    """A banked circle sampled like ADS-B, checked against an exact per-sample root solve."""
    radius, speed, alt = 1500.0, 60.0, 600.0
    w = speed / radius

    def circle(t):
        t = np.atleast_1d(t)
        return np.column_stack([radius * np.cos(w * t), radius * np.sin(w * t), np.full_like(t, alt)])

    def circle_vel(t):
        return np.array([-radius * w * np.sin(w * t), radius * w * np.cos(w * t), 0.0])

    obs = np.array([2500.0, 400.0, 0.0])
    track = sampled_track(circle, 0, 300, interval)
    t_h = np.linspace(20, 280, 400)

    truth = []
    for th in t_h:
        te = brentq(lambda s: s + np.linalg.norm(circle(s)[0] - obs) / C - th, th - 60, th)
        r = obs - circle(te)[0]
        truth.append(F0 / (1.0 - circle_vel(te) @ (r / np.linalg.norm(r)) / C))

    err = np.abs(heard_curve(track, obs, t_h, F0) - np.array(truth))
    assert np.nanmax(err) < TOL_HZ


def test_closest_approach_and_arrival():
    p0, v = np.array([-6000.0, 800.0, 500.0]), np.array([60.0, 0.0, 0.0])
    track = sampled_track(straight(p0, v), 0, 200, 1.0)
    t_cpa, d, t_arr = closest_approach(track, np.zeros(3))
    assert t_cpa == pytest.approx(100.0, abs=1e-3)
    assert d == pytest.approx(np.hypot(800.0, 500.0), abs=1e-3)
    # On a straight line the heard pitch crosses f0 exactly at the delayed CPA time.
    assert heard_curve(track, np.zeros(3), np.array([t_arr]), F0)[0] == pytest.approx(F0, abs=1e-3)


def test_estimators_recover_speed_f0_distance():
    """G2 criterion: v, f0 and d recovered within 2% from the curve alone."""
    speed, lateral, alt = 60.0, 800.0, 500.0
    p0, v = np.array([-36000.0, lateral, alt]), np.array([speed, 0.0, 0.0])
    track = sampled_track(straight(p0, v), 0, 1200, 1.0)
    t_h = np.arange(110, 1190, 0.01)
    f = heard_curve(track, np.zeros(3), t_h, F0)

    f_a, f_r = f[0], f[-1]
    v_est = speed_from_doppler(f_a, f_r)
    f0_est = f0_from_doppler(f_a, f_r)
    i = int(np.argmin(np.abs(f - f0_est)))
    slope = np.gradient(f, t_h)[i]
    d_est = distance_from_slope(f0_est, v_est, slope)

    assert v_est == pytest.approx(speed, rel=0.02)
    assert f0_est == pytest.approx(F0, rel=0.02)
    assert d_est == pytest.approx(np.hypot(lateral, alt), rel=0.02)


def test_straight_pass_curve_matches_simulator():
    p0, v = np.array([-6000.0, 800.0, 500.0]), np.array([60.0, 0.0, 0.0])
    track = sampled_track(straight(p0, v), 0, 200, 1.0)
    t_cpa, d, t_arr = closest_approach(track, np.zeros(3))
    t_h = np.linspace(30, 190, 500)
    err = np.abs(straight_pass_curve(t_h, t_arr, d, 60.0, F0) - heard_curve(track, np.zeros(3), t_h, F0))
    assert np.nanmax(err) < TOL_HZ


def test_harmonics_share_the_doppler_ratio():
    p0, v = np.array([-6000.0, 800.0, 500.0]), np.array([60.0, 0.0, 0.0])
    track = sampled_track(straight(p0, v), 0, 200, 1.0)
    t_h = np.linspace(30, 190, 200)
    f1 = heard_curve(track, np.zeros(3), t_h, F0)
    f2 = heard_curve(track, np.zeros(3), t_h, 2 * F0)
    np.testing.assert_allclose(f2, 2 * f1)
    # Speed comes out the same from either harmonic.
    assert speed_from_doppler(f1[0], f1[-1]) == pytest.approx(speed_from_doppler(f2[0], f2[-1]))
