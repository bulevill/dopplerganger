"""Synthetic G3 scene fixture (test / driver only, not part of search).

Builds ADS-B-like straight tracks in a truth-centred ENU frame, renders
them with ``synth.render_scene`` at the truth observer, and converts the
tracks into the search frame for ``search.segments_from_tracks``.

Geometry (all CPA points are perpendicular to the heading, so the
chosen point really is the closest point of approach):
  A  prop, heading  90 deg (east),  CPA 1500 m north, alt 1000 m, 60 m/s, f0 90 Hz
  B  prop, heading   0 deg (north), CPA 2000 m east,  alt 1200 m, 55 m/s, f0 110 Hz
  C  prop, heading  90 deg (east),  CPA 8000 m south, alt  900 m, 65 m/s, f0 95 Hz
     (faint: detected by M1, but far from the truth observer)
  D  prop, heading   0 deg (north), CPA 12000 m east, alt 1000 m, 60 m/s, f0 100 Hz
     (inaudible at truth: amp ~3e-4, below the noise; it is the flight
     the silence gate should penalise at wrong cells east of truth)
  E  jet,  heading 180 deg (south), CPA 9000 m west,  alt 3000 m, 150 m/s
     (broadband swell only, no tone)

A is at CPA (emission) T0+250 s; B is ``dt_between`` later.  With
``dt_between=600`` the detected CPA-arrival difference is ~601 s and
there is a ~60 s silence between the events' spans; with 920 s the
silence is ~600 s, as in the video.
"""
from __future__ import annotations

import numpy as np
from datetime import datetime, timezone

from .geodesy import LocalFrame
from .physics import Track
from .synth import Source

TRUTH_LAT = 41.65
TRUTH_LON = -73.85
T0 = int(datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc).timestamp())


def _straight(cpa_enu, heading_deg, speed, alt, t_cpa_emit, t_span=450, dt=5.0):
    h = np.radians(heading_deg)
    direction = np.array([np.sin(h), np.cos(h), 0.0])
    t = np.arange(t_cpa_emit - t_span, t_cpa_emit + t_span + 1, dt, dtype=float)
    pos = np.asarray(cpa_enu, dtype=float)[None, :] + (t - t_cpa_emit)[:, None] * speed * direction[None, :]
    pos[:, 2] = alt
    return Track(t, pos)


def build_scene(search_frame, dt_between=600.0, duration=1100.0, seed=1,
                include_distractors=True):
    """Return a dict with everything a G3 run needs.

    ``search_frame`` is the LocalFrame used for the search grid.
    Tracks are returned both in the truth frame (``sources``, for
    ``render_scene`` with observer (0,0,0)) and in the search frame
    (``segments``, for ``search.run_search``).
    """
    truth_frame = LocalFrame(TRUTH_LAT, TRUTH_LON, 0.0)
    t_a = T0 + 250.0
    t_b = t_a + float(dt_between)
    specs = [
        ("A", "prop", 90.0, np.array([0.0, 1500.0, 1000.0]), 1000.0, 60.0, 90.0, t_a),
        ("B", "prop", 0.0, np.array([2000.0, 0.0, 1200.0]), 1200.0, 55.0, 110.0, t_b),
    ]
    if include_distractors:
        specs += [
            ("C", "prop", 90.0, np.array([0.0, -8000.0, 900.0]), 900.0, 65.0, 95.0, T0 + 550.0),
            ("D", "prop", 0.0, np.array([12000.0, 0.0, 1000.0]), 1000.0, 60.0, 100.0, T0 + 465.0),
            ("E", "jet", 180.0, np.array([-9000.0, 0.0, 3000.0]), 3000.0, 150.0, 0.0, T0 + 400.0),
        ]
    sources = []
    tracks_truth = []
    for label, cls, heading, cpa, alt, speed, f0, t_cpa in specs:
        tr = _straight(cpa, heading, speed, alt, t_cpa)
        tracks_truth.append((label, cls, tr))
        sources.append(Source(tr, cls, f0=f0, label=label))
    # Convert tracks to the search frame via lat/lon (exact, not a shift).
    items = []
    for i, (label, cls, tr) in enumerate(tracks_truth):
        lat, lon, h = truth_frame.to_llh(tr.position(tr.t))
        enu_search = search_frame.to_enu(lat, lon, h)
        items.append((f"TEST{label}", i, cls,
                      Track(np.asarray(tr.t, dtype=float), enu_search)))
    from .search import segments_from_tracks
    segments = segments_from_tracks(items)
    observer_search = search_frame.to_enu(
        np.array([TRUTH_LAT]), np.array([TRUTH_LON]), np.array([0.0]))[0]
    return {
        "truth_lat": TRUTH_LAT, "truth_lon": TRUTH_LON, "t0": float(T0),
        "duration": float(duration), "seed": seed,
        "sources": sources, "segments": segments,
        "observer_search": observer_search,
        "truth_frame": truth_frame,
        "expected": {  # CPA-arrival time in audio seconds, from synth truth
            "A": {"f0": 90.0}, "B": {"f0": 110.0},
        },
    }
