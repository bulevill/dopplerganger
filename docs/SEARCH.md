# M3–M6 search module — G3 build (2026-11-07)

Fresh implementation, written from `FRAMEWORK.md` (§3 funnel, §4 posterior,
§7 formulas, §9.3 tunables) and the source strategy notes (not included).
The old partial file is untouched as `geoloc/search_partial_reference.py`
— it was never run and stops mid-function.

## Files created / changed

- `geoloc/search.py` — **new, from scratch** (~1500 lines; now
  ~2,300 after the 9 fixes and the G3b fixes). Public API:
  `SearchConfig`, `Segment`, `segments_from_tracks`, `load_segments`,
  `load_segments_with_stats`, `trim_segments`,
  `corridor_cpa_estimate`, `make_grid`, `build_cpa_table`,
  `pair_gate`, `single_event_gate`, `audibility_prob`,
  `silence_loglik`, `collapse_blobs`, `score_hypothesis`,
  `refine_blob`, `run_search`, `credible_masks`, `write_outputs`,
  `write_diagnostic_plot`.
- `geoloc/synthetic_scene.py` — **new** test/driver fixture: two target
  props (non-parallel, Δt 600/920 s) + distractors (faint 8 km prop,
  inaudible 12 km prop, jet), truth 41.65 N, -73.85 W.
- `scripts/g3_synthetic.py` — **new** G3 driver (see below).
- `scripts/g3b_real.py` — **new (2026-11-08)** real-day timing
  harness for the Windows machine (see "Real-day timing" below).
- `tests/test_search.py` — **new**, 42 tests (9 original + 11
  regression tests for the 9 fixes + 13 for the G3b fixes below
  + 5 for the v3 notes patch + 4 for jet deferral; all pass).
- `scripts/g3b_pick_window.py` — **new (2026-11-08)** per-hour
  small-plane counter for picking a G3b window (see v3 notes).
- `results/g3_full/` — outputs from the full G3 driver run (see Results).
- `geoloc/audio.py` was modified only in the v3 notes patch
  (note 6: NaN guard in `Spectrogram.value`); `physics.py`,
  `synth.py`, `geodesy.py`, `terrain.py`, `telemetry.py` untouched.

## Decisions recorded (2026-11-07, in code comments too)

1. **G3 first, then G4.** The module is built and tested on a 50 km box;
   the CONUS funnel (per-day CPA cache, tiling) is the G4 step.
2. **Doppler form is `physics.py`:** `f_heard = f0 / (1 - v_r / c_eff)`,
   not the strategy doc's approximate `f0 * (1 - v_r / c)`.
3. **Both max and marginalised scores are computed and kept.**
   `report.json` → `scored_hypotheses[].loglik_max / loglik_marg`, and
   the heat map is the marginalised posterior. Peak locations agree;
   loglik_max − loglik_marg ≈ 10 in the G3 run (marginalised is wider,
   as expected — that width difference is what T4 calibrates).
4. **Soft P(audible) silence, no hard 6 km cutoff** (FRAMEWORK §3
   improvement 3, §4). `audibility_prob(d, cls)` starts from the synth
   model — spherical spreading + `excess_db_per_km=4.0`, SNR against
   `noise_rms=0.02`, logistic scale 3 dB. At 2 km P≈0.999, 6 km P≈0.13,
   10 km P≈0.001. T3 replaces it with measured values.

## How it works (one paragraph per funnel step)

- **F3/F4 CPA table:** per grid cell, per segment, the global CPA of the
  piecewise-linear track (edge projection, numba, two-pass count/fill
  CSR; pure-python fallback if numba is absent). Stored with slant
  distance, CPA time, speed, arrival time `t_arr = t_cpa + d/c`, and a
  bitmask of which events it could match (distance gate asymmetric:
  low side −20%/3σ, high side ×2 that; speed gate ±20%/3σ — see
  "F4 asymmetric distance gate" below).
  The spline `Track` is used only for scoring, not gating.
- **F5 pair gate:** at one cell, CPAs for different events whose implied
  T0 values agree within 5 s (or, with known T0, whose predicted audio
  CPA times agree). Hypotheses are tried for *every* event subset of
  size >= 2 (and size 1 with known T0) — pairs, triples, quads, ...
  up to 6 events, pruned by span above that and recorded. Candidate
  lists are sorted by CPA distance before the per-event cap (12), and
  a combination budget overrun (500 per cell/subset) is evaluated
  closest-first and recorded, never silently skipped.
- **F6 silence:** `Σ log(1 − P(audible))` over silence windows, closest
  CPA per segment per window, matched flights excluded. The silence
  value is cached by (cell, T0, matched flights) — a second hypothesis
  at the same cell with a different T0 or different flights gets its
  own score.
- **F7 collapse:** cells grouped by (events, flights) into blobs.
- **F8 scoring:** forward curve via `physics.emission_to_arrival`
  (dt_emit 0.5 s), sweep f0 per event (±3 Hz, 0.5 step) and a *common*
  dt (±5 s, locked offset), harmonics weighted 1/k, mean whitened
  spectrogram score → loglik = mean × N_eff / 10 dB, N_eff capped at 50
  (frames are correlated; uncapped N_eff ≈ 300 gave loglik ≈ 2900 and
  a one-cell posterior — see Calibration note).
- **F9 refine:** 100 → 25 → 5 m windows (500/150/50 m half-width) around
  each surviving blob, locked (dt, f0), small re-sweep at the final peak.
  Known T0 is threaded through (pair gate with t0_known, or
  single-event gate for the 1-event case), so refinement works for
  the single-event + known-T0 variant too, and the refined T0 stays
  at t0_known. A refinement failure raises and is recorded in
  `funnel_counts.refine_errors` / `report.refine_errors` — never
  silently swallowed.
- **M6 posterior:** per cell, logsumexp over hypotheses of
  (event loglik + silence loglik), uniform prior, Gaussian smoothing
  σ = 1 grid cell (systematic error + grid discretisation), then
  normalise. BOTH maps are true per-cell sweeps: each scored-blob
  cell is scored with a full dt/f0 sweep, giving a genuine
  loglik_max and loglik_marg per cell (measured: 12.9 s for 486 cells
  on the full G3 run). Outputs: `heatmap.tif` in azimuthal-equidistant
  (+proj=aeqd, centred on the search frame origin, in metres — the
  ENU grid is regular in this CRS; see Fix 6c), band 1 = probability,
  band 2 = log-posterior, marginalised; `credible_regions.geojson`
  (50/90/95%, EPSG:4326, reprojected from the projected CRS),
  `report.json`, `diagnostic.png` (spectrogram + modelled curves for
  the best mode). Mode `mass` is the whole peak's mass (Voronoi sum
  over cells assigned to that mode, in ENU metres), mode `log_prob_*`
  are log posterior probabilities of the mode cell, mode `loglik_*`
  are the true scored logliks of that mode's best hypothesis, and
  mode `flights` is populated from that hypothesis (icao, seg, cls,
  event, CPA time, slant distance, speed).

Time bases: flight time = Unix, audio time = seconds from recording
start, `t_audio = t_arr_unix − T0`. Documented in the module docstring.

## Results — G3 (actually run, 2026-11-07)

### Full driver (50 km box, 200 m grid, Δt = 920 s, duration 1600 s)

```
python scripts/g3_synthetic.py --out results/g3_full --duration 1600 --dt-between 920 --grid 200
```

- Render 31.1 s, `audio.detect` 2.5 s (4 events found, incl. the two
  targets at f0 90.01 / 110.00 Hz, t_c within 0.03 s of truth, and a
  spurious 191 Hz harmonic event), `run_search` 22.6 s, total 56 s.
  (Search is slower than the first G3 run's 7.4 s because the
  marginalised map is now a true per-cell sweep: 12.9 s for 486 cells.)
- Funnel: 121,170 CPA entries, 56,690 cells with a CPA, 486 pair
  hypotheses (all subsets, not just pairs + full set), 433 cells,
  6 blobs, all 6 scored, 3 refined, 0 refine errors.
- **Coarse peak (max and marginalised, same cell): error 144 m.**
- **Refined peak: error 2 m** (41.65002, -73.85001 vs truth 41.65, -73.85).
- **Truth in 50/90/95% credible regions: True / True / True.**
- **G3 PASS** (truth in 90%, top-mode error < 1000 m; full S2 target
  is ≤500 m regional — the refined peak is well inside that).
- Best hypothesis: flights TESTA + TESTC + TESTB (all three real
  events), loglik_max 250.1, loglik_marg 240.4, silence loglik 0.0 —
  and this triple is now actually present in `report.json`
  `scored_hypotheses` (it was quoted in the first README but missing
  from the report; see Fix 4 below). Top mode: mass 0.99995,
  log_prob_marg -2.11, flights TESTA/TESTC/TESTB with CPA times.

Re-run 2026-11-08 (after the G3b fixes below): same command,
**G3 still PASSES** — coarse peak error 144 m, refined peak error
2 m (41.65002, -73.85001), truth in 50/90/95%. Render 46.8 s,
`audio.detect` 3.1 s, `run_search` 27.7 s (funnel `time_total_s`
24.3 s; write outputs 3.4 s), total 77.7 s. Timing breakdown:
CPA table 1.08 s, pair gate 3.08 s, scoring 0.24 s, refinement
3.45 s, marginalised sweep 16.3 s, silence 0.02 s. Corridor
estimate for the 5-segment scene: 122,750 (candidate 100,250,
silence-only 22,500 — the jet TESTE); product estimate 312,500.
Modes now carry `"piece"` in every flight dict (see check 3 below).

### Pytest

```
python -m pytest tests/test_search.py -q
# 33 passed in 210.2 s
```

Tests: the original 9 (audibility, CPA table, pair gate, silence,
scoring, `load_segments` NaN-alt, single-event + known-T0 gate,
small G3 end-to-end) plus 11 regression tests for the 9 fixes:
silence cache keyed by T0/flights, pair-gate closest-12 + combo
truncation recording, true marginalised sweep shape, load_segments
interpolation / no-alt drop / baro-only-when-allowed, grid NaN
height, mode clustering in metres, grid distortion, scale guard,
refine with known T0 + single-event refinement + refine failure
recorded, and the G3 triple-in-report + correct mode fields test
(incl. the projected GeoTIFF check).

### Single-event + known-T0, and the scale guard (manually verified)

- Single event + known T0, 500 m coarse grid: refined result present,
  refined peak within tolerance of truth or its 3000 m mirror
  (mirror ambiguity, FRAMEWORK §1). On a 1000 m grid the nearest cell
  is ~680 m from truth and its CPA distance falls just outside the
  ±20% gate, so the gate finds nothing — a grid-discretisation limit,
  noted in the test.
- Large box (CONUS, 24–50 N, -125–-66 W): `run_search` raises
  `ValueError` telling the user to tile / use the G4 per-day CPA
  cache, instead of OOMing or silently truncating.

## Fixes — 9 bugs fixed 2026-11-07 (none affected G3)

1. **Silence cache (Fix 1).** Was keyed by cell only; now keyed by
   (cell, round(T0, 3), matched flights). Test:
   `test_silence_cache_keyed_by_t0_and_flights`.
2. **Refinement (Fix 2).** `t0_known` is threaded through `refine_blob`
   (pair gate with t0_known, single-event gate for 1 event); the
   single-event + known-T0 case is now refined; refinement failures
   raise and are recorded in `funnel_counts.refine_errors` /
   `report.refine_errors` with a warning, never swallowed. The
   re-sweep silence term is now computed at the refined cell.
   Tests: `test_refine_with_known_t0_and_single_event`,
   `test_refine_failure_is_recorded_not_swallowed`.
3. **Flights dropped in busy airspace (Fix 3).** Candidate lists are
   sorted by CPA distance before the cap (config
   `max_candidates_per_event=12`); a >500-combo group is evaluated
   closest-first up to `max_pair_combos=500` and the truncation is
   recorded (`pair_gate_truncated_groups`,
   `pair_gate_combos_evaluated`). Test:
   `test_pair_gate_closest_12_survives_and_combos_recorded`.
4. **Three-flight answer lost (Fix 4).** All event subsets of size
   >= 2 are now enumerated (plus size 1 with known T0), up to
   `max_subset_events=6`, pruned by span above that and recorded.
   The G3 triple TESTA+TESTC+TESTB is now the top entry in
   `report.json` `scored_hypotheses`. Test:
   `test_g3_triple_in_report_and_modes_correct`.
5. **Report fields (Fix 5) — MIGRATION NOTE.** Mode fields were
   mislabelled: the old `loglik_max` / `loglik_marg` in `modes` were
   actually log posterior probabilities. They are now
   `log_prob_max` / `log_prob_marg` (<= 0, equal to log(prob) of the
   mode cell), and `loglik_max` / `loglik_marg` in `modes` are now
   the TRUE scored logliks of that mode's best hypothesis — same
   field names, different (correct) meaning. `modes[].flights` is
   now populated (icao, seg, cls, event, CPA/arrival time, slant
   distance, speed), and `mass` is the whole peak's mass (Voronoi
   sum over cells assigned to the mode in ENU metres), not one
   cell's probability. Any code reading the old `modes[].loglik_*`
   as a log-probability must switch to `log_prob_*`.
6. **(a) Summed vs best map (Fix 6a).** The marginalised map was
   the max map shifted by a per-blob constant. It is now a true
   per-cell full dt/f0 sweep for every scored-blob cell
   (`posterior_cells_scored`, `posterior_scoring_seconds` in the
   funnel counts; 12.9 s / 486 cells on the full G3 run). Test:
   `test_marginalised_map_is_true_per_cell_sweep`.
   **(b) Height mismatch (Fix 6b).** `load_segments` no longer
   substitutes `alt_baro_m` for missing `alt_geom_m`. Missing
   geometric altitude inside a segment is interpolated in time;
   a segment with no geometric altitude at all is dropped and
   counted (`segments_dropped_no_alt`), unless
   `allow_baro_fallback=True` is set explicitly (then counted and
   warned). NaN ground heights in `make_grid` are filled with the
   median finite height and counted (`height_nan_count`), never
   left as NaN ENU cells. New function:
   `load_segments_with_stats`. Tests:
   `test_load_segments_nan_alt_interpolated`,
   `test_load_segments_no_alt_dropped_and_baro_only_when_allowed`,
   `test_make_grid_nan_height_filled_not_nan`.
   **(c) Heat-map distortion (Fix 6c).** `heatmap.tif` is now in
   azimuthal-equidistant (+proj=aeqd, centred on the frame origin,
   in metres), in which the ENU grid is regular — NOT EPSG:4326
   with a mean dlat/dlon transform. The CRS is stored in
   `report["output_crs"]`; `credible_regions.geojson` stays
   EPSG:4326 (reprojected). Mode clustering, `truth_in_region`,
   and refined-mode matching now use ENU metres, not degrees.
   A distortion diagnostic (`report["grid_distortion"]`,
   dlon variation) warns for large boxes. Tests:
   `test_find_modes_clusters_in_metres_not_degrees`,
   `test_grid_distortion_grows_with_box`, plus the GeoTIFF CRS
   check in the triple test.
   **(d) Regional-only sizing (Fix 6d).** Limits are config, not
   hard-coded slices: `refine_top_n=3` (was `scored[:3]`),
   `max_blobs_to_score` (already config, now with a recorded
   truncation count), `max_grid_cells=500_000`,
   `max_cpa_entries=5_000_000`, `allow_large_box=False`. Every
   truncation is recorded in `funnel_counts` (blobs, refine,
   pair-gate, subsets). A box above the scale limits raises a
   clear `ValueError` (tile / G4 cache) instead of OOMing.
   The full G4 tiled per-day CPA cache is NOT implemented here —
   that is the G4 step. Test: `test_scale_guard_refuses_large_box`.

## G3b real-flight fixes (2026-11-08)

The first real-flight run (6 Nov, 50 km boxes) was blocked by the
scale guard: the loader returned 311 segments (Hudson Valley) and
3,274 segments (near New York) for the whole day, and the guard
estimated grid cells x segments = 19M / 205M against a 5M cap. Even
keeping only segments within an hour of the recording left New York
at 7.9M. All three fixes below are in place in `geoloc/search.py`
(now ~2,300 lines); 13 new tests (33 total, all passing).

### Fix 1 — corridor-based size estimate, not the product

`corridor_cpa_estimate(segments, grid, rmax)` sums, per segment,
the grid cells inside that segment's bbox expanded by `rmax`,
clipped to the grid (counted from the grid's own x/y centres with
`searchsorted`). That is an upper bound on `_best_cpa` calls. The
old estimate `n_cells * n_segments` was the cost of a brute-force
grid x segments scan — exactly the O(N·M) the funnel was designed
to avoid. The guard in `run_search` now uses the corridor estimate
(same caps: `max_grid_cells=500,000`, `max_cpa_entries=5,000,000`);
both estimates are recorded in `funnel_counts`
(`cpa_entries_estimated_corridor`, `cpa_entries_estimated_product`,
plus `corridor_cells_max_per_segment` / `_median_per_segment`).
The split is reported in three buckets — candidate
(prop/turboprop/rotor), silence-only (jet only), and unknown —
because unknown-class segments are treated as possible matches
(weight 0.5), not silence-only (v3 note 3 below); the three
buckets sum to the total.

Measured, 50 km box at 200 m = 62,500 cells (synthetic realistic
traffic: turning tracks, reporting gaps, airport pattern work,
out-and-back re-entries; segment times spread over the whole day):

| case | product (old) | corridor (new) | guard |
|---|---|---|---|
| 311 segs, trimmed, no time filter | 19,437,500 | 3,518,567 | PASS |
| 311 segs, time filter + trim | 500,000 | 66,428 | PASS |
| 3,274 segs (390 prop + 2,884 jet), trimmed, no time filter | 204,625,000 | 50,049,038 | REFUSE (by design — see below) |
| 3,274 segs, time filter + trim | 5,937,500 | 1,763,355 | PASS |

The unfiltered 3,274-segment case is *correctly* refused: its CPA
table really is 21.2M entries and takes 85 s. The corridor estimate
is not a free pass — it measures the actual work. The time filter
is what makes the New York box pass, which is the designed order
of operations (cheap geometry/timing first). Note the filtered
product estimate (5.9M) would *still* refuse — only the corridor
estimate lets the guard pass.

### Fix 2 — time filter in the loader

`load_segments` / `load_segments_with_stats` take `t_start` /
`t_end` (Unix seconds) + `time_margin_s`. The SQL hit CTE *and*
the main select filter points to `[t_start - margin, t_end +
margin]`; a segment touching the box at a different time of day
is not loaded at all. A segment partly overlapping the window is
kept, trimmed to the window, plus one bracketing point either side
(via `LAG`/`LEAD` in SQL) so the Track spline at the window edge is
anchored. If both are `None`, behaviour is unchanged (whole day).
Stats: `segments_dropped_by_time`, `points_time_filtered_out`,
`load_seconds`.

**The margin is an assumption, and it is recorded (check 2).**
A margin of ±X means the method assumes the recording start time
is known to within ±X — *when the margin is additional to the
window*. If the caller already folded the T0 uncertainty into
`t_start`/`t_end` itself, pass `margin_already_in_window=True`
to `run_search` and the report says so instead of claiming the
margin is the uncertainty (v3 note 5 below). With no filter the
report says "No time filter (whole day)" and records
`margin_s` as `None`, never "±0 s". `run_search` takes
`t_start` / `t_end` / `time_margin_s` (the window the segments
were loaded with — it does not re-apply the filter) and records
`report["time_filter"] = {t_start, t_end, margin_s, filtered,
margin_already_in_window, assumption}`.
If only the date is known, pass all three as `None`: the loader
keeps the whole day and the size question comes back — the corridor
guard is the only thing standing between a whole-day load and the
5M cap.

**Margin rule / worked example.** The window must cover the T0
uncertainty + the recording duration + sound delay (~d/c, up to
~30 s at 10 km) + the Δt between events. Example: T0 known to
±600 s, recording 1600 s long, events up to 920 s apart → call
with `t_start = T0 - 600`, `t_end = T0 + 1600 + 920`,
`time_margin_s = 60` (or fold the 600 s into `t_start` directly).
If T0 is known exactly, `t_start = t0_known`,
`t_end = t0_known + duration`, margin ~60 s is enough.

### Fix 3 — trim long segments to the search area

`trim_segments(segments, frame, lat_min, lat_max, lon_min, lon_max,
rmax_m=10_000, min_points=5, trim_extra_m=3000)` (also in the
loader by default, after the time filter, before the Track is
used): each segment is trimmed to the box expanded by `rmax_m`
**plus a 3 km safety margin** (see check 5), plus one bracketing
point either side of each kept run. If trimming splits a track
into disjoint runs (leaves and re-enters), it is split into
separate `Segment` objects (same `icao`/`seg`, `subseg` = run
index) rather than joined across the gap. Runs shorter than
`min_points` are dropped. Stats: `points_trimmed_spatially`,
`segments_split`, `runs_dropped_too_short`,
`segments_dropped_by_spatial_trim`. A 300 km straight flight
crossing a 50 km box is trimmed to ~76 km of points (box + 2x
rmax + margin + bracketing ≈ 255 points at 5 s sampling);
CPA / truth recovery is unchanged (pair-gate test with 300 km
tracks).

### Fix 4 — CPA loop: kept, not rewritten

Measured first, per the brief. With time filter + trim (the real
pipeline), the existing cell-first numba loop (`_cpa_count` /
`_cpa_fill`, product bbox checks, `_best_cpa` only inside the
sbox) is fast enough, so it was **kept** and the times documented
— no rewrite for its own sake:

| case (50 km box, 200 m grid, time filter + trim) | CPA entries | CPA table time | run_search total |
|---|---|---|---|
| 311 segs → 8 in window | 36,464 | 0.9 s | 0.9 s |
| 3,274 segs → 95 in window (in-memory filter) | 720,807 | 3.0 s | 4.3 s |
| 3,274 segs → 54 in window (real loader, SQL filter) | 627,057 | 2.7 s | — (timing-only) |
| 311 segs, trimmed, no time filter (worst case) | 2,135,863 | 9.4 s | 14.6 s |

All well under the ~30 s threshold. CPA semantics unchanged
(global CPA per segment per cell, same gates). The unfiltered
3,274-segment table (21.2M entries, 85 s) is refused by the guard,
not by the loop — the fix for that case is the time filter, not a
faster loop. If G4 ever needs more, the segment-first restructure
is documented in the task notes, not implemented here.

### The six additional checks (2026-11-08)

1. **Real day as well as synthetic counts.** Matching segment
   counts (311 / 3,274) does NOT reproduce real flights. The
   synthetic G3b tests now mimic real behaviour: turning tracks,
   reporting gaps (~10% dropped points, 12 s gaps), airport
   pattern work (circles), out-and-back re-entries
   (`_realistic_track`, `_g3b_segments` in `tests/test_search.py`).
   **The real 2026-11-06 timing was NOT run in this environment**
   — the original development sandbox had no real `points.parquet`. Instead,
   `scripts/g3b_real.py` is the timing harness for the Windows
   machine (see "Run on Windows" below): it loads the real Parquet
   with the time filter + trim, prints the guard estimate (old
   product vs new corridor), load time, CPA table time, and the
   full run_search timing breakdown. It was exercised end-to-end
   here against a synthetic Parquet built from the same generator
   (Hudson preset: 54 segments loaded, corridor 1,437,144 vs
   product 3,375,000, CPA 627,057 entries in 2.7 s, load 0.8 s).
2. **Time margin recorded.** See Fix 2 above.
   Test: `test_time_filter_recorded_in_report`.
3. **Split segments get unique IDs; repeat passes expected.**
   The unique segment identity is `(icao, seg, piece)` — `piece`
   is the `subseg` run index, now in every flight dict in
   `report.json` (`segments`, mode `flights`, `scored_hypotheses`
   flights, `refined` flights). Audit of `pair_gate` /
   `single_event_gate` / `collapse_blobs` / silence: all dedup
   and exclusion use `seg_idx`, which are *indices into the
   segments list*, not `(icao, seg)` — so two pieces of one
   aircraft are already two different entries and CAN be two
   events. The only gap was the report, which made two pieces
   look identical; that is fixed. Test: an out-and-back track
   (one aircraft, two passes) is split into two pieces and the
   pair gate finds a hypothesis using one piece per event:
   `test_out_and_back_two_pieces_two_events`
   (also `test_trim_splits_reentering_flight`).
4. **Corridor estimate counts jets.** Jets are not candidates, but
   they still sit in the CPA table for the silence check — in the
   New York box 2,884 of 3,274 segments are jets, most of the
   work. The estimate counts EVERY segment at full cost and
   reports the split separately (`cpa_entries_estimated_candidate`
   vs `cpa_entries_estimated_silence_only`, `segments_candidate`
   vs `segments_silence_only`). In the filtered NY case the
   silence-only part is 1,624,445 of 1,763,355 corridor cells
   (92%). No slimming of jet table entries was implemented:
   slimming was considered and deliberately NOT done — it would
   change CPA semantics for the silence check, and the full-entry
   estimate already passes the guard, so there is nothing to gain
   by risking it. If G4 ever needs it, it must be measured
   against the full-entry baseline first.
   Test: `test_corridor_estimate_user_cases_and_jets`.
5. **Trim edges.** The trim uses box + `rmax` + `trim_extra_m`
   (default 3000 m ≈ one reporting interval at cruise speed,
   15 s x ~150–200 m/s, plus margin), so a CPA right at the trim
   edge keeps its bracketing points. The bracketing points are the
   NEAREST points outside the expanded box on each side (the
   point immediately before the run starts / after it ends),
   not just the first kept point — including across a reporting
   gap that straddles the boundary. Tests:
   `test_trim_keeps_cpa_at_corridor_edge` (CPA 9.5 km from an
   edge cell, gap straddling the boundary, found with the correct
   distance/time) and `test_cpa_beyond_rmax_excluded_by_gate_not_trim`
   (a flight 11 km from every cell is KEPT by the trim but
   correctly EXCLUDED by the CPA gate — the exclusion is the
   gate's job, not the trim's).
6. **Search time, broken down.** `run_search` now records
   per-stage times in `funnel_counts`: `time_cpa_table_s`,
   `time_pair_gate_s`, `time_scoring_s`, `time_refine_s`,
   `time_marginalised_sweep_s` (same value as
   `posterior_scoring_seconds`), `time_silence_s`, `time_total_s`,
   plus `time_write_outputs_s`. Silence time is a SUBSET of the
   scoring / posterior / refine stages (it is called inside
   them), so it must NOT be added to the stage sum. Load time is
   NOT in `run_search` (the loader is outside it) — it is in the
   loader stats as `load_seconds`, and the G3b harness prints it.
   Test: `test_timing_breakdown_present_and_consistent`.
   G3 numbers: CPA 1.08 s, pair gate 3.08 s, scoring 0.24 s,
   refine 3.45 s, marginalised sweep 16.3 s, silence 0.02 s,
   total 24.3 s (see the re-run above).

New tests (13): `test_corridor_estimate_user_cases_and_jets`,
`test_load_segments_time_filter`, `test_trim_long_flight_to_box`,
`test_trim_preserves_cpa_and_truth`,
`test_trim_splits_reentering_flight`,
`test_trim_drops_runs_shorter_than_min_points`,
`test_trim_keeps_cpa_at_corridor_edge`,
`test_cpa_beyond_rmax_excluded_by_gate_not_trim`,
`test_out_and_back_two_pieces_two_events`,
`test_time_filter_recorded_in_report`,
`test_timing_breakdown_present_and_consistent`,
`test_g3b_like_311_segments_50km_box` (rendered scene + ~300
realistic distractors, 50 km box, 200 m grid — guard passes,
truth in 90%, refined error < 1000 m),
`test_g3b_like_3274_segments_guard_passes` (390 prop + 2,884 jet,
time filter + trim — guard passes, run_search completes,
CPA table < 30 s).

### Are we following the same order of operations as the video?

Short answer: yes — and the guard/loader bug was a deviation
from that order, which these fixes restore.

| Video / original strategy order | Our funnel |
|---|---|
| Filter telemetry by type (prop vs jet; ICAO type code) | F1 class split (prop/turboprop/rotor candidates; jets silence-only) |
| 200 m grid, only in a 10 km corridor around the flight path | F2 grid (200 m) + corridor: CPA only within `rmax` = 10 km of a cell |
| CPA + slant distance + delayed CPA (sound-propagation delay) | F3/F4 CPA table: global CPA per segment per cell, `t_arr = t_cpa + d/c` |
| Δt filter, speed/distance filter, silence filter | F5 pair gate on Δt (+ feature gate on d/v), F6 silence gate |
| Collapse to flight-pair hypotheses | F7 `collapse_blobs` |
| Forward-model only the survivors; locked-offset score against the spectrogram | F8 quick score (`score_hypothesis`, locked dt/f0) |
| Refine 200 → 100 → 25 → 5 m | F9 refine 100 → 25 → 5 m |

The computing-capacity solution IS that order: cheap
geometry/timing first (type filter, corridor grid, CPA,
Δt/speed/silence gates), expensive spectrogram scoring last, and
only on survivors. The old guard and loader deviated from it in
exactly the wrong direction: they estimated and loaded as if the
search were a brute-force grid × segments scan — the O(N·M) the
funnel was designed to avoid. The guard refused a normal 50 km
box because it priced the work at 19M/205M; the loader loaded
every segment of the whole day that touched the box, whole
tracks hundreds of km long, regardless of when it flew. These
fixes restore the order: the estimator prices the corridor work
only; the loader keeps only the recording window (time filter)
and only the part of each track near the box (trim); the CPA
loop still does corridor work only; spectrogram scoring still
runs only on pair-gate survivors. The remaining capacity work
for G4 is the same idea at CONUS scale: a reusable per-day CPA
cache + tiling (see Limitations below) — not implemented here.

## Calibration note (important for T4)

The first G3 run failed on credible-region coverage: mean scores are
~20 dB, so uncapped log-likelihoods were ~2900 and the posterior put
98% of its mass on one 500 m cell whose centre is ~350 m from the
truth (grid discretisation alone). The fix, now in the code, is
`n_eff_cap=50`, `score_scale_db=10`, and posterior smoothing
σ = 1 cell. That is a *starting* calibration, not a measured one:
T4 must fit the temperature / score scale on held-out real recordings
(FRAMEWORK §9.3). Do not quote the 90% region as calibrated until T4.

## Run on Windows (target machine)

In a Miniforge Prompt, from `C:\Projects\Geolocation` (copy the new
files there — `geoloc\search.py`, `geoloc\synthetic_scene.py`,
`scripts\g3_synthetic.py`, `tests\test_search.py`, this README):

```
conda activate geoloc
cd C:\Projects\Geolocation
python -m pytest tests\test_search.py -q
python scripts\g3_synthetic.py --out results\g3_full --duration 1600 --dt-between 920 --grid 200
```

Notes: run from the project root (scripts import `geoloc`);
`load_segments` needs `D:\geoloc-data\processed\adsblol\<day>\points.parquet`
or set `GEOLOC_DATA` if the drive letter changed; `search.py` itself
makes no network calls and never calls `Terrain` — pass
`terrain.Terrain(...).height` as `height_fn` to `run_search` in
production (tests use a flat 0 m height). The driver currently uses
the in-memory synthetic scene by default; `--day 2026-11-06` is
accepted but, if a real Parquet exists, it only notes it — building
the G3-from-real-tracks path is a G4-era step (pick real tracks from
the Parquet, render from them, same `run_search`).

### Real-day timing (G3b) — run this on Windows

The real 2026-11-06 timing was NOT measured in this environment
(no real `points.parquet` here). Run the harness against the real
data, same Hudson Valley and New York 50 km boxes as the blocked
run, with the recording window (adjust `--t-start` / `--t-end` to
the actual recording; `--time-margin` is the T0 uncertainty, see
Fix 2 above):

```
conda activate geoloc
cd C:\Projects\Geolocation
python scripts\g3b_real.py --day 2026-11-06 --preset hudson --t-start 2026-11-06T12:00:00Z --t-end 2026-11-06T12:42:00Z --time-margin 60 --timing-only
python scripts\g3b_real.py --day 2026-11-06 --preset newyork --t-start 2026-11-06T12:00:00Z --t-end 2026-11-06T12:42:00Z --time-margin 60 --timing-only
python scripts\g3b_real.py --day 2026-11-06 --preset hudson --t-start 2026-11-06T12:00:00Z --t-end 2026-11-06T12:42:00Z --time-margin 60 --audio <recording.wav> --out results\g3b_hudson
```

It prints: segments loaded (after time filter + trim), loader
stats, grid cells, OLD product estimate vs NEW corridor estimate
(candidate / silence-only / unknown), guard PASS/REFUSE, load time, CPA
table time, and (with `--audio`) the full run_search timing
breakdown. The presets are EXAMPLE boxes (Hudson centred on
41.65 N, -73.85 W; New York centred on Manhattan) — replace with
the boxes actually used, via `--box lat_min lat_max lon_min
lon_max`. Expected (synthetic, same segment counts): Hudson —
~54–95 segments in the window, corridor ~1.4–1.8M, CPA ~0.6–0.7M
entries in ~3 s, load < 1 s; New York — corridor must be < 5M
after the filter. If the real corridor is much larger, report the
harness output — it means the real tracks are longer / denser
than the synthetic ones, and the per-day CPA cache moves up the
G4 list.

## v3 notes patch (2026-11-08) — six notes from the user's real-day check

None of these blocks G3b. Small patch only; G3 results unchanged.

**User's real-day numbers (2026-11-06, 1 h window 11:30–12:30 UTC):**
Hudson Valley 10 segments, corridor estimate 0.13M (old product
0.6M), actual table 0.09M, load/table 2.2 s / 0.4 s; Manhattan
152 segments, 3.1M / 9.6M / 2.2M, 2.4 s / 1.5 s; Teterboro 124
segments, 2.9M / 7.8M / 1.8M, 2.4 s / 1.4 s; Hudson whole day
323 segments, 8.1M (refused) / 20.2M / 4.7M, 1.8 s / 2.5 s.
All three 1 h boxes pass the 5M cap. Out-and-back flights split
cleanly: 15 in Manhattan, 6 at Teterboro. Tests: 46 pass on the
user's machine (33 in `tests/test_search.py` here plus the other
test files); G3a batch 8/8 pass, refined error exactly as v2
(2–13 m), runs 18–50 s.

1. **Little headroom near New York — comment corrected, cap NOT
   changed.** A 1 h window already uses 3.1M of the 5M cap, ~99%
   of it jets for the silence check, so with time known only to
   ~1 h those boxes will probably be refused. The code comment
   estimating "~100k" for that case was ~30x too low; it now
   quotes the measured Manhattan (152 segs, corridor 3.1M,
   actual 2.2M) and Teterboro (124 segs, 2.9M / 1.8M) figures.
   The cap is NOT raised and the jet table is NOT slimmed here —
   that is a G4 decision (see note 2, and the jet-deferral change
   that follows this patch).
2. **Limit calibration (G4) — now measured, see "Limit
   calibration — full-search benchmark" below.** The 5M cap was
   uncalibrated: the whole-day Hudson box is refused at 8.1M
   estimated, but its table is actually 4.7M entries and builds
   in 2.5 s. The cap must be set from memory and time measured
   at G4, not guessed. The benchmark (2026-11-08) times the
   full search at 4.7M / 19.5M / 48.9M entries: total 114 s /
   438 s / 1,296 s, peak RSS 786 MB / 1,329 MB / 2,745 MB, and
   the pair gate — not the table — is the bottleneck at 50M.
   The default cap is NOT changed in this patch; the
   recommendation is the user's decision.
3. **Unknown aircraft mislabelled — fixed.** The corridor
   estimate used to label unknown-class segments "silence-only",
   but the search treats them as possible matches (weight 0.5).
   They now have their own bucket: candidate + silence-only
   (jet only) + unknown = total, in both the estimate and the
   segment counts (`cpa_entries_estimated_unknown`,
   `segments_unknown`). Test: `test_corridor_estimate_unknown_bucket`
   (also asserts the unknown segment is still matchable by the
   pair gate).
4. **Loader missed near-edge flights — fixed.** `load_segments`
   only picked up flights entering the box itself; flights
   within 10 km of the edge that never cross it were left out
   (6 jets Hudson, 3 Manhattan on the real day). The SQL hit
   now uses the box expanded by `rmax` (10 km, in degrees with
   cos(lat)); the trim to box+rmax+3 km is unchanged. Recorded
   in stats as `segments_added_by_expanded_hit`. Test:
   `test_loader_expanded_hit_near_edge_flight` (5 km outside:
   loaded and in the CPA table; 20 km away: not loaded).
5. **Time-filter wording — fixed.** With no filter the report
   used to say "±0 s"; it now says "No time filter (whole day)"
   and records `margin_s` as `None`. And the margin is now
   described as the window margin, not automatically as
   start-time uncertainty: if the caller already folded the
   uncertainty into `t_start`/`t_end`, pass
   `margin_already_in_window=True` and the report records that.
   Test: `test_time_filter_no_filter_wording_and_margin_basis`.
6. **audio.py:29 warning — fixed at the source, then rewritten
   safely (2026-11-08).** A NaN modelled frequency (interpolated
   ratio outside the emission-time range, in
   `_score_event_matrix`) reached `Spectrogram.value`, and
   `np.rint(NaN).astype(int)` raised "invalid value in cast".
   The first fix replaced non-finite inputs with a value one step
   outside the grid and relied on the index then being out of
   range — correct today, but the user noted (2026-11-08) it could
   quietly return a real value if the audio settings ever changed.
   The rewrite makes the non-finite mask part of the validity
   check itself: NaN/inf in -> NaN out for any grid, with no
   `rint`/cast ever applied to a non-finite value — same result,
   no global warning suppression. Test:
   `test_spectrogram_value_nan_no_warning` (runs with warnings
   as errors, including a second grid with 0.5 s / 2 Hz steps to
   prove the guard does not depend on the grid settings).

**Picking a G3b window with more small-plane traffic.**
The 1 h noon window had only one prop/turboprop per box — too
few events for a real-track test. `scripts/g3b_pick_window.py`
counts prop / turboprop / rotor (plus jet / unknown) segments
per UTC hour in a box, so a busier hour can be picked first:

```
conda activate geoloc
cd C:\Projects\Geolocation
python scripts\g3b_pick_window.py --day 2026-11-06 --preset hudson
python scripts\g3b_pick_window.py --day 2026-11-06 --preset newyork
python scripts\g3b_pick_window.py --day 2026-11-06 --box 41.425 41.875 -74.15 -73.55
```

It prints one row per UTC hour (prop, turboprop, rotor, jet,
unknown, small, total) and the hour with the most small-plane
traffic. A segment spanning hours is counted in each hour it is
in the box. Tested here against a synthetic Parquet (see
`test_g3b_pick_window_script`).

## Jet deferral (2026-11-08) — high jets out of the early stage

**Problem (user's real-day numbers).** In the New York 1 h box
jets are ~99% of the early CPA work (~30,000 entries for props
and similar vs ~3.1 million for jets; Manhattan 152 segments,
corridor estimate 3.1M, actual table 2.2M), yet jets are only
used for the silence check, which runs on the ~450 surviving
cells. The 5M cap was nearly exhausted by work that only matters
late.

**Design.** `SearchConfig.defer_jets = True` (default;
`defer_jets=False` reproduces the old behaviour exactly).
`split_early_deferred(segments, config)` splits the load before
the guard and the CPA table:

- **Early** = prop / turboprop / rotor / unknown, PLUS jets with
  any point at or below `jet_early_max_alt_m` (3000 m). FRAMEWORK
  F1 already says jets below ~3 km AGL are candidates, so a low
  jet near an airport can genuinely be the plane heard and stays
  early. Unknown-class aircraft stay early (many are small planes;
  the FAA registry is incomplete) — they are never deferred.
- **Deferred** = jets with every point above the threshold
  (high jets: silence-pool only). They are NOT in the early CPA
  table, NOT in the pair gate / scoring, and NOT counted in the
  corridor estimate or the guard. The guard counts the early
  stage only.

**Late stage — silence only, on surviving cells only.** The
silence for a hypothesis is the early-table silence (unchanged,
cached by cell + T0 + matched flights) PLUS a deferred-jet
silence computed on demand for that cell only, with the same
`_best_cpa` the full table would have used (`_jet_cpa_entries`,
cached per cell — the CPA does not depend on T0, only the
silence-window check does). No full jet CPA table is ever built
over the whole grid; that would defeat the purpose. Every place
silence is used was audited: F8 blob-centroid scoring, the
marginalised re-sweep over posterior cells, and both silence
calls inside `refine_blob` (via its new `silence_fn` parameter)
all get the jet contribution. A jet kept early as a low jet is
never double-counted: it is not in the deferred list, so the
early pass handles it (and excludes it when matched).

**The altitude proxy, and its limitation.** "Low" means any
point at or below 3000 m in the segment's ENU height, which is
ellipsoid height, not height above ground — in high terrain it
keeps more jets early (safe direction). It is also
frame-dependent: ENU height equals ellipsoid height only at the
frame origin and falls by ~d²/(2R) away from it (up to ~80 m
across a 50 km box), so a jet near the threshold can be kept
early in one frame and deferred in another. That only ever
keeps MORE jets early, never defers a truly low jet. Terrain-
based AGL is a G4 refinement, not built here.

**Accepted limitation (recorded).** A high jet's sound could in
principle be mistaken for a prop's near an airport, and
deferring high jets removes the chance of matching it as an
event. Accepted: a jet loud enough to be heard as an event is
almost always low, and low jets stay early. This is also
recorded in `report["jet_deferral"]`, with the deferred
segments listed in `report["deferred_segments"]`.

**Identity proof (measured).** With `defer_jets=True` vs `False`
on the same inputs, results must be identical whenever no high
jet is matched as an event:

- `test_jet_deferral_identity_g3_scene`: on the G3 scene (frame
  at the truth, TESTE deferred), modes, pair hypotheses and
  refined peaks are identical on vs off.
- `test_jet_deferral_high_jet_silence_identical`: a high jet in
  a silence window penalises the truth hypothesis by exactly
  the same amount on vs off; with the jet removed entirely the
  penalty is 0.
- `test_jet_deferral_low_jet_still_matched`: a low jet (2000 m)
  is still matched as an event with deferral on.
- G3 driver, deferral on vs off (`--no-defer-jets`): both PASS,
  coarse 144 m, refined 2 m, truth in 50/90/95%, run_search
  29.9 s (on) / 29.7 s (off), time_total 26.6 s / 26.5 s,
  identical modes. Honest caveat: in the driver's frame (box
  centre offset 8 km east / 6 km north of the truth) TESTE's
  ENU height is 2917 m, below the threshold, so it is kept early
  in both runs and the driver comparison does not exercise
  deferral — the pytest identity test above is the one that does.
- New funnel_counts fields: `segments_early`,
  `segments_deferred_jets`, `cpa_entries_estimated_deferred_jets`
  (what the estimate WOULD have been),
  `cpa_entries_deferred` (actual deferred-jet CPAs computed on
  demand), `time_deferred_jet_silence_s`.

**New York-like measured numbers** (synthetic, 50 km box, 200 m,
1 h window, 390 props + 2,884 jets over the day → 110 segments
in the window: 18 early, 92 deferred high jets):

| | corridor estimate | CPA entries | CPA time |
|---|---|---|---|
| all segments | 2,129,781 | 953,193 | 3.7 s |
| early only | 238,660 | 179,490 | 0.6 s |
| deferred jets | 1,891,121 | (on demand, per surviving cell) | — |

The early table is 5.3x smaller here, not the ~100x of the real
day: the synthetic props are long tracks crossing the whole box
(~25k entries each), while the real 1 h window had very few,
short props (~30k entries total) vs ~150 jets (~3.1M). The ratio
depends on the prop/jet mix; what deferral guarantees is that
the jet part is gone from the early stage. The guard now passes
on the early estimate (238,660) with real headroom.

**Real-world result (user, 2026-11-08): jet deferral gives ZERO
saving on the real 2026-11-06 data, in all four test areas — 0
jets moved out of the early stage.** Two reasons, both measured:

- The flight files stop at 10,000 ft (~3,050 m). That limit was
  set when the data was converted (`telemetry.convert_day`,
  `max_alt_ft=10_000`). A jet only counts as "high" if it stays
  above 3,000 m the *whole* time, which almost never happens
  below that limit — a jet that dips below 3,000 m at any point
  is kept early.
- Near New York the jets really are low: most are taking off or
  landing at the big airports. Even a 500 m cutoff would move
  only 12 of 134 jets.

The feature is correct (the silence check gives exactly the
same answers early vs late, verified by the user) and harmless,
so it stays — but it is NOT the solution to the size limit, and
the ~100x saving expected from the synthetic test does not
happen on real data. Do NOT try to fix this by lowering the
threshold: that would risk matching low jets as events for
almost no saving. The binding constraint is the 10,000 ft
conversion cap; raising it is a data-conversion decision for
G4, not a search-code change. What helps instead is the size
limit itself — see "Limit calibration (G4)" below.

## Limit-calibration benchmark (2026-11-08) — `scripts/bench_large.py`

Times the FULL `run_search` at large CPA-table sizes, to set the
5M cap from measured time and memory — NOT to raise it. The user
measured 48 bytes per CPA entry, so 5M is only ~240 MB (Hudson
whole day 4.8M / 233 MB / 2.5 s to build; New York 1 h 2.2M /
105 MB / 1.1 s; his machine has 64 GB), but the table is not the
whole search — the pair gate, scoring and marginalised sweep may
slow faster than the table grows. The script uses the real G3
perception (render + `audio.detect`, 1600 s, Δt 920 s) plus
synthetic distractor segments in the recording window, 50 km box
at 200 m, jet deferral OFF by default so the table actually holds
the jets, ~12% prop / ~88% jet like the real New York box. It
reports per-stage times, CPA entries and MB, peak RSS, peak error
and whether the truth is in the 90% region, and writes JSON with
`--out`. Run on Windows (geoloc env), one scale at a time:

```
python scripts\bench_large.py --n-prop 40 --n-jet 310 --out results\bench_5m.json
python scripts\bench_large.py --n-prop 170 --n-jet 1230 --out results\bench_20m.json
python scripts\bench_large.py --n-prop 420 --n-jet 3080 --out results\bench_50m.json
```

`--only-cpa` builds the table only; `--defer-jets` runs with
deferral on. Traffic density at the top end is NOT realistic
(~3,500 segments in one 50 km box in one hour is ~20x New York)
— the point is table size and how each stage scales, not the
traffic level. Smoke-tested in this workspace 2026-11-08
(`--n-prop 2 --n-jet 5 --only-cpa`: 149,042 entries in 6.6 s,
7.2 MB); the 5M/20M/50M runs themselves have NOT been executed
yet — run them on Windows and set the cap from the measured
per-stage times. Note the perception render costs ~210 s per run
in this environment, so each scale takes several minutes before
the search itself starts.

## Limit calibration — full-search benchmark (2026-11-08)

Note 2 above is now measured. The user measured one CPA entry
at 48 bytes, so the 5M cap is only ~240 MB (Hudson whole day
4.8M entries / 233 MB / 2.5 s to build; New York 1 h 2.2M /
105 MB / 1.1 s; his machine has 64 GB), and asked: before
raising the cap, time the FULL search at that size, because the
later stages may slow down faster than the table grows.

**Method** (`scripts/bench_large.py`). The real G3 perception
(render + `audio.detect`, 1600 s, Δt 920 s) plus N synthetic
distractor segments in the recording window, in a 50 km box at
200 m (62,500 cells), jet deferral OFF so the table actually
holds the jets. Distractors are the realistic generator from
the tests (turning tracks, 3 s reporting, ~10% dropped points,
pattern work, out-and-back), ~12% prop / ~88% jet like New
York. Measured on this VM (4 cores, 7.9 GB RAM) — slower than
the user's machine, so the absolute times are upper bounds and
the *scaling shape* is what transfers. Render + detect (~49 s)
is excluded from the `run_search` times below.

| actual CPA entries | segments in window | table | CPA table | pair gate | scoring | refine | marginalised sweep | total `run_search` | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,672,541 | 705 | 224 MB | 36.3 s | 35.0 s | 0.7 s | 6.9 s | 34.9 s (897 cells) | 114 s | 786 MB |
| 19,509,333 | 3,005 | 936 MB | 153.7 s | 216.7 s | 0.7 s | 25.1 s | 41.7 s (1,087 cells) | 438 s | 1,329 MB |
| 48,856,841 | 7,505 | 2,345 MB | 480.2 s | 745.5 s | 1.0 s | 49.6 s | 18.1 s (412 cells) | 1,296 s | 2,745 MB |

**How each stage scales** (entries ×4.18, then ×2.51):
- **CPA table: linear.** ×4.23, then ×3.12 (~7.8 µs/entry
  at 5M, ~9.8 µs at 50M). 50M entries ≈ 8 min for the table.
- **Pair gate: faster than linear — the bottleneck.** ×6.19,
  then ×3.44 (exponent ~1.3 vs entries); 58% of the total at
  50M. Hypotheses 4,426 → 30,267 → 73,713; blobs 138 → 1,189
  → 4,254; pair-gate groups truncated at the 500-combo cap:
  0 → 380 → 22,822. That last number is a *correctness*
  concern, not just speed — at 50M the gate is silently
  capping work in 22,822 groups.
- **Scoring: flat** (~1 s). It is capped by
  `max_blobs_to_score=20`, so it costs the same at any size.
- **Refine: sublinear** (×3.6, then ×2.0).
- **Marginalised sweep: flat vs table size.** 18–42 s,
  scaling with scored-blob cells (412–1,087), not with
  entries. It is NOT the bottleneck at large size — the pair
  gate is.
- **Total: roughly linear in entries** (×3.84, then ×2.96;
  ~26.5 s per million entries at 50M).
- **Memory: linear, no surprise.** Peak RSS ≈ ~550 MB fixed
  + 48 B/entry: 50M ≈ 2.75 GB here.

**The cap is on the estimate, not the table.** The corridor
estimate over-counts the actual table: 13.1M / 55.7M / 140.9M
estimated vs 4.67M / 19.5M / 48.9M actual here (~2.9×,
turning tracks have big bounding boxes). On the real day it
was 1.4× (Manhattan 3.1M vs 2.2M) to 1.7× (Hudson whole day
8.1M vs 4.7M). So a 50M cap on the *estimate* admits roughly
17M actual entries (synthetic factor) to ~30M actual (real
factor). To admit 50M *actual* entries, the cap on the
estimate would have to be ~145M.

**Truth-recovery caveat — read this before raising the cap.**
At all three benchmark densities the truth was NOT recovered
(peak error ~22–23 km; top mode a spurious distractor pair
with mass 1.0; at 50M the true TESTA/TESTB pair was not even
among the 12 scored blobs). Mechanism: 138 → 4,254 blobs, but
only the top 20 by Δt error are scored, so chance pairs with
tiny Δt errors crowd the truth out. BUT the benchmark density
is 5–50× real: 705–7,505 in-window segments in one 50 km box
vs ~150 in real New York 1 h (~782 entries/cell at 50M vs ~35
real). A real 50M table would come from a whole day over a
much bigger area, where per-cell density is like the 5M case —
and at real densities G3a passes 8/8. So the timing numbers
above are about table size and transfer; the truth failure is
a warning about **per-cell density and the top-20 scoring
budget**, not a verdict on a 50M cap. The G4 repeat on real
data must check truth recovery, not just timing.

**Recommendation (default cap NOT changed in this patch —
the user's decision).** A 50M cap is feasible on the user's
64 GB machine for a single regional search: ~2.4 GB table,
~3 GB peak RSS here, ~22 min full search here (faster on his
machine). The conservative step is 20M (~1 GB, ~7 min here).
Before raising it, the G4 work should be the pair gate:
its superlinear scaling and the 22,822 truncated groups at
50M, and whether the top-20 blob budget needs to grow with
density. `scripts/g3b_real.py` now takes `--max-cpa-entries`
so the repeat can be run without editing code.

**Repeat it.** Synthetic (this workspace):

```
python scripts/bench_large.py --n-prop 80 --n-jet 620 --out results/bench_5m.json
python scripts/bench_large.py --n-prop 360 --n-jet 2640 --out results/bench_20m.json
python scripts/bench_large.py --n-prop 900 --n-jet 6600 --out results/bench_50m.json
python scripts/bench_large.py --n-prop 80 --n-jet 620 --only-cpa   # table only
```

Real data (Windows, 2026-11-06; `--timing-only` first, then a
full run with a recording):

```
python scripts\g3b_real.py --day 2026-11-06 --preset hudson --timing-only --max-cpa-entries 50000000
python scripts\g3b_real.py --day 2026-11-06 --preset newyork --t-start 2026-11-06T11:30:00Z --t-end 2026-11-06T12:30:00Z --timing-only --max-cpa-entries 50000000
python scripts\g3b_real.py --day 2026-11-06 --preset hudson --audio <recording.wav> --out results\g3b_hudson --max-cpa-entries 50000000
```

## F4 asymmetric distance gate (2026-11-09) — turning aircraft

`SearchConfig.d_gate_high_factor` (default 2.0; 1.0 restores the old
symmetric gate). `event_gates` now computes
`d_tol = max(0.20 * ev.d, 3 * sigma_d)` as before, but
`d_lo = max(30, ev.d - d_tol)` while `d_hi = ev.d + factor * d_tol`.
The low side and the speed gate are unchanged.

Why: M1 fits a straight-pass Doppler model, but a real aircraft that
turns or changes speed near its CPA sounds like a straight pass at a
CLOSER distance than its geometric CPA, so M1's sound-implied distance
is systematically below the geometric CPA the gate compares against.
Evidence (G3b tuning set, 24 renders of real 2026-11-06 Hudson/New
York ADS-B tracks; 86 matched passes): M1 d vs sound-implied d median
error 5.5% (4/86 off by 2× or more), geometric d vs sound-implied d
18% (9/86), M1 d vs geometric d 19% (8/86). The bias is one-sided:
geometric / sound-implied ratio 1.05 median, 1.36 at the 75th
percentile, 2.48 at the 90th, 5.5 at most; only 6/55 below 0.8. A
local correction from the range acceleration at the CPA does NOT work
(the bend acts over the whole audible pass, not just at the CPA).
Gate variants on the 55 expected G3b passes M1 matched: symmetric
44/55; high side ×1.5 → 50/55; **×2 → 52/55**; ×3 → 52/55;
compared against sound-implied d/v instead → 51/55.

Cost, measured here on the G3 synthetic scene (same 4 detected events,
factor 1.0 vs 2.0): CPA entries unchanged (121,170 — the gate marks
entries, it does not create them); marked entries 53,662 → 58,632
(+9.3%, 44.3% → 48.4% of entries); pair-gate combinations evaluated
23,535 → 39,407 (+67%); pair hypotheses 486 → 729 (+50%); cells with
a pair hypothesis 433 → 646 (+49%); pair-gate time 2.54 s → 2.71 s
(+7%); CPA-table time 0.87 s → 1.07 s. The top hypothesis is
identical under both factors (triple TESTA+TESTC+TESTB), coarse peak
144 m, refined peak 2 m, truth in 50/90/95% — straight-track G3a is
unaffected. The fuller option — comparing against the straight-pass
fit of the flight's forward Doppler curve instead of the geometric
CPA — is deferred; it costs a curve fit per surviving CPA.

Checks still to run on Windows (not done here): re-run the G3b
batches with the new gate and compare truth-in-region and error with
RESULTS.md (New York runs should gain most):
`python scripts\g3b_render.py --reuse-audio <batch dir>` (same audio,
new search), plus the large benchmark (`scripts/bench_large.py`) for
the wider gate's cost at scale — the G3-scene cost above is small, but
the pair gate is superlinear, so the +50% hypotheses here may grow
faster at 20–50M entries.

## Density-scaled scoring budget (2026-11-09) — v5 gate on real-density G3b

End-to-end G3b re-run on Windows (same audio, matched detector,
factor 1.0 vs 2.0): Hudson 7/8 both, hudson_v2 5/8 both, New York
7/8 vs 6/8 — total **19/24 vs 18/24**. The gate is right in
principle (NY run_00: FAIL 553 m -> PASS 87 m, the turning case),
but at New York density it costs more than it gains: run_01 PASS
220 m -> FAIL 8.9 km, run_04 PASS 132 m -> FAIL 5.2 km.

What the reports show (read from
`results/g3b/newyork_2026-11-06_gate1.0/` and `_gate2.0/`):
blobs are ranked for scoring by min dt_err, and dt_err is a
near-tie at city density (scored blobs' dt_err 0.00-0.06 s), so
the fixed top-20 cut is near-arbitrary. The blobs that located
truth at factor 1.0 are absent from `scored_hypotheses` at 2.0:
run_01's winning single-event blob (ada936:4:0, event 3,
centroid loglik 151.3, the highest scored) and run_04's winning
single-event blob (a0feaf:2:0, event 0, centroid loglik 14.1).
Their exact dt_err ranks are NOT in the reports (no blob list
is recorded), only that they fell outside the top 20
(run_01: 43 -> 46 blobs, cut 23 -> 26; run_04: 84 -> 103 blobs,
cut 64 -> 83). Two nuances: the cut blobs are single-event
blobs of *other* flights whose geometry happens to place the
observer at truth, not the true-flight pair blobs — the
true-flight singles (a45915:0:0 in run_01, a1fce0:4:0 in run_04)
were scored identically under both factors. And in run_04 at
2.0 a truth-location blob WAS scored and refined to 575 m
(a04a38:7:0, mass 0.16) but lost the posterior to a wrong blob
(0d1401:0:0, mass 0.82) — so run_04 is crowding at the cut
*plus* a lost score battle, not a pure never-scored case.

The fix (option 2, user preference): the scoring budget now
scales with density. `SearchConfig.max_blobs_to_score` (20) is
the floor; the effective budget is
`min(max_blobs_to_score_cap=100, max(20, ceil(blobs_to_score_frac=1.0
* n_blobs)))`, recorded per run as
`funnel_counts["max_blobs_to_score_effective"]`. Sparse scenes
are unchanged (G3a: 6 blobs -> 20). `d_gate_high_factor` stays
at its 2.0 default — the budget fix is what lets it ship at 2.0;
the interim default of 1.0 is no longer needed. Deliberately
NOT scaled: `max_pair_combos` (500). Candidate lists are capped
at 12 per event first, so a 2-event combination is at most
144 < 500 — the cap only bites subsets of >=3 events, and the
run_01/run_04 failures were single-event blobs lost at the blob
cut. Pair gate is also ~83% of run_04's 602 s, so scaling it
would blow the runtime ceiling.

Runtime ceiling (measured stage times, run_04 gate 2.0):
pair gate 499 s, marginalised sweep 68 s over 3,515 cells
(~10-20 ms/cell), quick scoring 0.44 s for 20 blobs
(~0.02 s/blob), total 602 s. Scoring up to the 100-blob cap
adds ~2 s of quick scoring but the sweep grows with scored
cells (est. ~10-20k cells -> ~100-400 s), so the worst run is
expected at roughly 750-1,000 s. That estimate is NOT measured
here — the NY-density validation (success criterion: keep
run_00's fix, recover run_01 and run_04, >= 20/24) can only be
run on Windows with the real renders:
`python scripts\g3b_render.py --preset newyork --start-utc 21:15 --end-utc 23:45 --min-side-s 0 --reuse-audio results\g3b\newyork_2026-11-06 --d-gate-high-factor 2.0 --out results\g3b\newyork_2026-11-06_gate2.0`
(Hudson batches: `--preset hudson`, `--min-side-s 0` for
`hudson_2026-11-06`, `60` for `hudson_2026-11-06_v2`.)

## Limitations / next steps for G4

- CPA is the *global* minimum per segment per cell (piecewise-linear);
  a turning flight with two local minima in one segment is represented
  once. Fine for G3; revisit for G4 if turning flights matter.
- Hypotheses are all event subsets; a 3-event hypothesis
  has a larger loglik than a 2-event one by construction (each real
  event adds evidence) — comparable because loglik is absolute vs the
  noise baseline, but spurious-event robustness should be checked in T5.
- Full-CONUS: F3 must become a reusable per-day CPA cache with tiling
  (FRAMEWORK §3 improvement 1); the current in-memory CSR table for a
  50 km box is 121k entries, CONUS at 200 m is ~10–25 GB by the
  framework's own estimate — `run_search` now refuses a box above
  `max_grid_cells` / `max_cpa_entries` with a clear error instead of
  running it. Benchmark at G4 first.
- `load_segments` is implemented and unit-tested (altitude
  interpolation, no-alt drop, baro-only-when-allowed, time filter,
  spatial trim), and the G3b harness (`scripts/g3b_real.py`) was
  exercised end-to-end against a synthetic Parquet in this
  environment — but it has NOT been run against a real
  `points.parquet` here. Run it on Windows against 2026-11-06
  before G4 (see above).
- The G3 driver does not yet pick real flights from a converted day
  (see above); the in-memory scene is ADS-B-like (5 s sampling,
  straight tracks) rather than real ADS-B. The G3b tests use
  realistic synthetic traffic (turns, gaps, pattern work,
  out-and-back) as a stand-in, not real tracks.
- The true per-cell marginalised sweep costs ~13 s for 486 cells on
  the full G3 run (vs 7.4 s total search before); at CONUS scale
  that cost must be paid only inside scored blobs, or the sweep
  must be coarsened — a G4 benchmarking question, not a G3 one.
- Single-event + known-T0 on a coarse grid is gate-sensitive: on a
  1000 m grid the nearest cell can fall just outside the ±20%
  distance gate (see the refine test note). Use a 500 m coarse grid
  for single-event searches, or widen the gate with a recorded
  systematic-error term, before the real-flight single-event test.
