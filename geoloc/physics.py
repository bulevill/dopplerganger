"""Forward acoustic model (module M4): what a ground observer hears from a moving aircraft.

Coordinates are local east-north-up (ENU) metres; times are seconds.
The model works forward from emission time: each sound sample leaves the aircraft
at t_emit and reaches the observer at t_emit + distance / c. A finished curve is
never shifted afterwards, because that distorts it (see FRAMEWORK.md section 7).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

C_AT_0C = 331.3  # speed of sound in dry air at 0 degC, m/s


def speed_of_sound(temp_c):
    """Speed of sound in dry air (m/s) for a temperature in degC."""
    return C_AT_0C * np.sqrt(1.0 + np.asarray(temp_c, dtype=float) / 273.15)


class Track:
    """Aircraft trajectory built from timestamped position samples (as from ADS-B).

    A cubic spline through the samples gives smooth position and velocity, so
    turns, climbs and speed changes are kept instead of assuming straight flight.
    """

    def __init__(self, t, pos):
        t = np.asarray(t, dtype=float)
        pos = np.asarray(pos, dtype=float)
        order = np.argsort(t)
        self.t = t[order]
        self._spline = CubicSpline(self.t, pos[order], axis=0)

    @property
    def t_start(self):
        return self.t[0]

    @property
    def t_end(self):
        return self.t[-1]

    def position(self, t):
        return self._spline(t)

    def velocity(self, t):
        return self._spline(t, 1)


def emission_to_arrival(track, observer, t_emit, c=343.0, wind=None):
    """Map emission times to arrival times and Doppler ratios at one observer.

    Returns (t_heard, ratio, distance). The heard frequency of any tone is
    ratio * f_emitted, so every harmonic k * f0 shares the same ratio.
    Wind (ENU m/s) changes the effective sound speed along each ray.
    """
    observer = np.asarray(observer, dtype=float)
    pos = track.position(t_emit)
    vel = track.velocity(t_emit)
    r = observer - pos
    distance = np.linalg.norm(r, axis=1)
    u = r / distance[:, None]  # unit vector, aircraft -> observer
    c_eff = c + (u @ np.asarray(wind, dtype=float) if wind is not None else 0.0)
    v_r = np.einsum("ij,ij->i", vel, u)  # positive when closing on the observer
    t_heard = t_emit + distance / c_eff
    ratio = 1.0 / (1.0 - v_r / c_eff)
    return t_heard, ratio, distance


def heard_curve(track, observer, t_heard, f0, c=343.0, wind=None, dt_emit=0.05):
    """Frequency heard at the given arrival times. NaN where no sound has arrived."""
    t_emit = np.arange(track.t_start, track.t_end, dt_emit)
    t_arr, ratio, _ = emission_to_arrival(track, observer, t_emit, c, wind)
    # Subsonic flight keeps t_arr strictly increasing, so plain interpolation is safe.
    return np.interp(t_heard, t_arr, f0 * ratio, left=np.nan, right=np.nan)


def closest_approach(track, observer, dt=1.0):
    """Closest point of approach: (t_cpa, slant distance, arrival time at c = 343 m/s)."""
    observer = np.asarray(observer, dtype=float)
    t = np.arange(track.t_start, track.t_end, dt)
    d = np.linalg.norm(track.position(t) - observer, axis=1)
    i = int(np.argmin(d))
    lo, hi = t[max(i - 1, 0)], t[min(i + 1, len(t) - 1)]
    res = minimize_scalar(
        lambda s: np.linalg.norm(track.position(s) - observer), bounds=(lo, hi), method="bounded",
        options={"xatol": 1e-4},
    )
    return res.x, res.fun, res.x + res.fun / 343.0


def straight_pass_curve(t_heard, t_c, d, v, f0, c=343.0):
    """Closed-form heard frequency for a straight, level, constant-speed pass.

    t_c is when the closest-approach sound arrives, d the closest slant distance.
    The audio stage fits this to a detected track, which works even when only part
    of the curve is visible.
    """
    t_cpa = t_c - d / c  # emission time of closest approach
    tau = np.asarray(t_heard, dtype=float) - t_cpa
    # Aircraft at (v * s, d) relative to the observer at emission time s (s = 0 at CPA).
    # Solve |p(s)| = c (tau - s): (v^2 - c^2) s^2 + 2 c^2 tau s + (d^2 - c^2 tau^2) = 0.
    a = v * v - c * c
    b = 2.0 * c * c * tau
    cc = d * d - c * c * tau**2
    s = (-b + np.sqrt(b * b - 4.0 * a * cc)) / (2.0 * a)  # a < 0: this root is the retarded one
    v_r = -v * (v * s) / np.hypot(v * s, d)
    return f0 / (1.0 - v_r / c)


# Estimators used by the audio stage (M1) to turn a measured curve into geometry.

def speed_from_doppler(f_approach, f_recede, c=343.0):
    """Aircraft speed from the far approach and recede frequencies. Independent of f0."""
    return c * (f_approach - f_recede) / (f_approach + f_recede)


def f0_from_doppler(f_approach, f_recede):
    """Emitted frequency from the far approach and recede frequencies."""
    return 2.0 * f_approach * f_recede / (f_approach + f_recede)


def distance_from_slope(f0, speed, slope_at_cpa, c=343.0):
    """Slant distance at closest approach from the frequency slope (Hz/s) where f == f0."""
    return f0 * speed**2 / (c * abs(slope_at_cpa))
