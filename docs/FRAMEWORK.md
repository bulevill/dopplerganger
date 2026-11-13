# Acoustic Geolocation — Research Framework

Status: written 2026-11-07 as the plan, then kept up to date as gates passed (see §8). Results to date: [RESULTS.md](RESULTS.md).
Source strategy: a publicly demonstrated acoustic-geolocation technique (my notes on it are not included). §3 compares its claimed reduction funnel with corrected estimates.

---

## 0. Goal and success criteria

**Goal:** Given an audio recording plus coarse metadata (a date, and optionally a region and an absolute time), output a **posterior probability heat map** of where it was recorded. Coverage should reach CONUS first and the world later, with calibrated credible regions (50/90/95%).

**Success means:**
| # | Criterion | How measured |
|---|---|---|
| S1 | True location falls inside the 90% credible region ~90% of the time (calibration) | Coverage test on own GPS-tagged recordings |
| S2 | Median localization error ≤ 500 m (two-event, regional search) | Own recordings, known truth |
| S3 | The method says "no confident answer" (flat posterior) rather than a wrong sharp peak when evidence is weak | Recordings with no or one faint flyover |
| S4 | A CONUS-day search finishes on available hardware in ≤ 24 h | Benchmark (§3) |

---

## 1. Problem formulation

### Unknowns
- Observer position **x** = (lat, lon); elevation follows from the DEM.
- Recording start time **T₀**: unknown unless timestamped; the date is assumed known.
- For each acoustic event *k*: which flight caused it, its emitted frequency **f₀ₖ**, and which harmonic is being tracked.

### Observables (from audio)
| Observable | Constrains |
|---|---|
| Arrival time of CPA crossing per event | Along-track position (given T₀) |
| Approach/recede frequencies f_a, f_r | Speed **v** = c·(f_a − f_r)/(f_a + f_r) (independent of f₀) |
| Slope df/dt at the CPA crossing | Slant distance **d** ≈ f₀·v² / (c·|df/dt|) |
| Interval Δt between events | Ties both events to one observer |
| Silence windows | Excludes positions where other flights would have been heard |
| Absolute level / harmonic richness | Rough range (weak; needs audibility model) |
| Lloyd's-mirror interference fringes (ground reflection) | Source height above ground (optional, advanced) |

### Identifiability (critical; missing from the source strategy)
- **One event, straight flight:** CPA distance gives two lines parallel to the track, one on each side (**mirror ambiguity**). Arrival time picks a point along them only if T₀ is known. Without T₀, one event yields a **line pair**, not a point.
- **One event, T₀ known:** two mirror candidates per matching flight. The ambiguity is broken only by turns or climbs during the pass, or by terrain.
- **Two events from non-parallel flights:** the two mirror line pairs intersect in ≤ 4 points, and Δt then selects among them. This is the minimal well-posed case without T₀.
- **Implication:** the heat map must be allowed to be **multimodal**. Report modes separately rather than a single point with an error ellipse.

---

## 2. Pipeline architecture

| Stage | Input | Output | Notes |
|---|---|---|---|
| **M1 Audio perception** | WAV/FLAC/MP3 + metadata | Event list: t_center, f_a, f_r, df/dt, harmonics, SNR, silence windows; STFT matrix S(f,t) | Fixed STFT params stored with output |
| **M2a Telemetry ingest** | ADS-B daily dumps | Cleaned tracks (prop/rotor only), resampled to 1 s, local ENU/ECEF coordinates, geometric altitude | Partitioned Parquet by date/H3 cell |
| **M2b Terrain** | DEM tiles | Z_ground(x,y) interpolator, heights in WGS84 ellipsoid | Convert geoid → ellipsoid |
| **M2c Atmosphere** | HRRR (US) / ERA5 (global) | c(x,y,z,t), wind (x,y,z,t) | Effective sound speed |
| **M3 Coarse screening** | Tracks + grid + event features | Per grid cell: arrival-event list → candidate flight pairs | The funnel, §3 |
| **M4 Forward simulation** | Candidate (x, flight) | f_modeled(t_heard) per harmonic | Emission-time integration, wind-aware |
| **M5 Scoring and refinement** | f_modeled + S(f,t) | Log-likelihood per (x, hypothesis); refined grids 100→25→5 m | Locked-offset rule |
| **M6 Posterior and output** | Log-likelihoods + prior | Heat map GeoTIFF, interactive HTML map, JSON report (modes, credible regions, flights) | §4 |
| **V Validation harness** | Own recordings + GPS truth | Error stats, calibration curves | Runs on every pipeline change |

---

## 3. Computational Reduction Funnel

The key design choice is **not** to evaluate audio against every grid point. Geometry and timing prune the search first. Only a small set of **flight-pair hypotheses** ever touch the spectrogram.

Order-of-magnitude estimates for **one CONUS day**. These are assumptions to be replaced by benchmarks at gate G4.

| Step | Operation | Count in → out (estimate) | Cost driver | Where it runs |
|---|---|---|---|---|
| F0 | Area of interest | CONUS 8.1M km² | — | — |
| F1 | **Track split by acoustic class** (not a hard jet drop). *Positive-event pool:* prop, turboprop and rotorcraft at any audible altitude, plus jets below ~3 km AGL (departure/approach, where fan tones are audible). *Silence pool:* **all** airborne aircraft, every type, so the silence gate (F6) can reject cells where any loud aircraft should have been heard. Drop ground/taxi | ~100k+ tracks → ~15–40k positive-pool tracks; silence pool keeps everything airborne | I/O and type lookup | DuckDB on Parquet |
| F2 | **Corridor grid:** 200 m cells within R_max (10 km) of kept tracks; H3 or tile index | 202M cells total → ~50–120M cells covered | Spatial index | CPU |
| F3 | **CPA events:** per (cell, track) compute t_cpa, d_slant, v, t_arrival = t_cpa + d/c. Keep only d < R_audible(type) | ~1–3B (cell, track) pairs → event table ~10–25 GB | Main CPU cost; vectorized segment search | Numba, tiled by region |
| F4 | **Feature gate:** keep events with v in [v_min, v_max] and d in [d_min, d_max] from M1 | ×0.05–0.2 | Cheap | CPU |
| F5 | **Pair gate (Filter A):** per cell, event pairs with \|Δt_pair − Δt\| ≤ ε. With T₀ known, use an absolute window instead. Sort events by time per cell and two-pointer scan | → ~10⁵–10⁷ cell-pairs | Cheap after sort | CPU |
| F6 | **Silence gate (Filter C):** reject a cell-pair if any other event with P(audible) > p* arrives inside a silence window | ×0.1–0.3 | Reuses F3 table | CPU |
| F7 | **Collapse to hypotheses:** group surviving cells by (flight1, flight2) into connected blobs | → ~10²–10⁴ hypotheses | Trivial | CPU |
| F8 | **Quick score:** M4+M5 at blob centroid with a coarse f₀/δt sweep; drop hypotheses below threshold | → ~10–10² hypotheses | Spectrogram lookups | CPU |
| F9 | **Fine refine:** 100 → 25 → 5 m grids, full sweep, locked offset, wind-aware | → final posterior | Heaviest per-point cost, applied to tiny area | CPU (GPU optional) |

### Comparison with the source funnel
The source funnel's counts are for **one specific recording** (two events, Δt = 920 s). They are reference points to reproduce at G4, not general constants.

| Source step | Source claim | Framework step | Assessment |
|---|---|---|---|
| 1. Metadata filter | Props are ~10–20% of flights; removes 80–90% of telemetry | F1 | Plausible by flight count. Low-altitude GA dominates *track points near the ground*, so measure the reduction in **corridor area**, not rows. Add an altitude/AGL filter, plus rotorcraft (category A7), which the source omits |
| 2. 10 km corridor | Grid only near tracks | F2 | Agree. Set width from the audibility model, not a constant |
| 3. Δt + speed bounds | "Lightweight arithmetic" → ~73,000 points (≈ 2,900 km², 0.036%) | F3–F5 | **Underestimated cost.** The CPA table (F3) is the single most expensive step: about 1–3B cell-track evaluations. It is cheap per item, not cheap overall. Apply speed/distance bounds (F4) *before* pairing (F5); this reorders the source |
| 4. Silence filter | Removes ~80% → ~15,000 points (≈ 600 km², 0.007%, below the stated "< 0.05%") | F6 | Source math is internally consistent but the 80% figure is recording-specific (dense airspace). Use soft audibility probabilities |
| 5. Forward simulation | Run on ~15,000 nodes | F7–F8 | Improve: group nodes by flight pair first (15k nodes → probably tens of hypotheses), simulate at blob centroids, then refine |
| 6. Mesh refinement | 100 → 25 → 5 m; text says "sub-meter" | F9 | **Sub-meter is not physically meaningful.** ADS-B position error (~10–30 m), 1 s timestamp error (~60 m along-track), and sound-speed error (~1% of slant range) set a floor of roughly 25–100 m. Stop refining when the grid spacing is below ~⅓ of the posterior width |
| Complexity | O(N·M·T) brute force → "weeks to minutes" | §3 Hardware fit | Unverified. Define N = grid cells, M = tracks, T = samples per track. Funnel cost is about O(M · cells_per_corridor · log T) for F3 plus O(survivors · sweep) for F9. Benchmark at G4 |

**What the source funnel leaves out:**
- the reusable per-day CPA cache (improvement 1 below)
- the case where the recording's absolute time is known, which tightens F5 sharply
- weather-corrected c
- output as a posterior heat map, rather than a single "target coordinate" (mirror ambiguity, §1)

### Improvements over the source strategy
1. **Precompute F1–F3 once per day and reuse it for every recording from that day.** The event table is independent of the audio, which makes it a reusable cache.
2. **Collapse to flight-pair hypotheses (F7) before any spectrogram work.** The source strategy scores per grid point. Neighbouring cells share the same flights, so score per hypothesis and refine spatially only inside winners.
3. **Make audibility probabilistic in the silence gate.** A hard 6 km cutoff kills true locations when another aircraft passed faintly (wind, masking noise). Use P(audible | d, type, ambient noise) and a soft penalty.
4. **Use the speed formula that ignores f₀:** v = c(f_a − f_r)/(f_a + f_r). The source's Δf ≈ 2f₀v/c needs an unknown f₀.
5. **Handle the case with no T₀ by sweeping T₀ implicitly through Δt**, not across the whole day for each event separately.
6. **Use a coarse DEM for F2–F6.** 90 m resolution (Copernicus GLO-90) is enough when cells are 200 m. Load the 30 m or 10 m DEM only for F9.
7. **Make the funnel adaptive:** if F6 leaves too many hypotheses (dense airport areas), tighten ε or add a third event before scoring.

### Hardware fit (tested on this machine)
8 CPU cores, 64 GB RAM, Intel UHD 630 (no CUDA), 147 GB free disk.
- F1–F8 for one CONUS day: **feasible on CPU** if tiled (estimate: tens of minutes to a few hours). Verify at G4.
- F9: fine for ≤ 10² hypotheses on CPU. A world-scale or multi-day search may need a rented cloud GPU or many cores.
- Disk: a few ADS-B days plus a regional DEM fit. CONUS 30 m DEM plus a week of global ADS-B will not. Plan an external drive or cloud bucket for world scale.

---

## 4. Probabilistic model and heat map

**Posterior over the grid:**

log P(x | audio) = log prior(x) + max or Σ over hypotheses h [ Σₖ log L(eventₖ | x, h) + log L(silence | x, h) ] + const

- **Event likelihood:** a matched-filter score along f_modeled, normalized against a noise baseline. Raw pixel sums are not probabilities. Convert score to log-likelihood with a **calibration curve learned from own recordings** (true vs. false hypotheses).
- **Silence likelihood:** Π over unheard flights of (1 − P(audible)).
- **Nuisance parameters** (f₀, δt, harmonic index): marginalize (sum) over the sweep instead of taking the max. This gives honest uncertainty.
- **Prior (optional):** uniform, or weighted by population or land cover (people record where people are). Keep it switchable and report both.
- **Hypothesis mixing:** the posterior is a mixture over flight-pair hypotheses. Multiple modes are expected and must be shown.

**Outputs:**
| Artifact | Format |
|---|---|
| Heat map (log-posterior and probability) | Cloud-optimized GeoTIFF, EPSG:4326 |
| Credible regions 50/90/95% (highest-posterior-density contours) | GeoJSON polygons |
| Interactive map | HTML (folium/deck.gl) with mode list |
| Report | JSON: modes (lat, lon, mass), flights matched (ICAO24, type, CPA time), parameters, data versions |
| Diagnostics | Spectrogram with modeled curves overlaid per mode |

**Calibration check (S1):** over N own recordings, plot the fraction with truth inside the p% region against p. It must lie on the diagonal. If it doesn't, scale the likelihood temperature.

---

## 5. Data source inventory

Endpoint checks run 2026-11-07 from this machine. "Reachable" means the server responded; it does not mean data was downloaded.

### 5.1 Flight telemetry (ADS-B)
| Source | Coverage / format | Cost / access | Test result | Fit |
|---|---|---|---|---|
| **adsb.lol globe_history** | Global, daily readsb tarballs on GitHub releases (`adsblol/globe_history_YYYY`); includes MLAT, `alt_geom` | Free, ODbL (attribution + share-alike) | ✅ GitHub API 200 | **Primary candidate.** Verify per-day size and position cadence |
| **OpenSky Network historical (Trino)** | Global, `state_vectors_data4` (high cadence), crowd-sourced | Free for academic/non-commercial use, **requires account + approved access** | ✅ trino endpoint reachable (303 → login); REST live API 200 | Primary for research credibility; apply now (lead time) |
| OpenSky REST API | Live only (no deep history) | Free, rate-limited | ✅ 200 | Testing only |
| ADS-B Exchange historical | Global readsb-hist | **Paid**; free samples (1st of each month) | ✅ samples site reachable (301) | Cross-check against samples |
| FlightAware AeroAPI / Firehose | Global, fused | **Paid** (expensive) | not tested | Not needed |
| FAA SWIM (ASDE-X/STDDS) | US surveillance feeds | Free but application + infrastructure | not tested | Out of scope |

**G1 measured, adsb.lol 2026-11-06** (report: `D:\geoloc-data\reports\g1_adsblol_2026-11-06.json`)
- Archive 3.8 GB/day (2 parts), 75,277 aircraft worldwide. Read directly from the tar; no extraction needed.
- Low-altitude (< 10,000 ft) aircraft touching the CONUS box: 15,831 piston, 2,783 turboprop, 2,592 rotorcraft, 13,021 jet, **9,865 with no type code**.
- Position gap for low-altitude piston aircraft: median 2.9 s, 10th percentile 0.6 s, **90th percentile 12.5 s**. Tracks are not 1 Hz; T1 must test 5–15 s sampling.
- Geometric altitude present on 96% of low piston points.
- Piston point sources: ADS-B 87%, ADS-R (rebroadcast UAT) 9%, MLAT 2.5%, TIS-B 0.8%. Timing quality likely differs by source; tag every point with it.
- Storage: one year ≈ 1.4 TB raw. Keep raw tars on D:, write a CONUS low-altitude Parquet subset for processing.

### 5.2 Aircraft type classification
| Source | Use | Test result |
|---|---|---|
| OpenSky aircraft metadata DB | ICAO24 → typecode, model | ✅ page reachable |
| FAA Releasable Aircraft Database | US N-numbers → engine type (reciprocating/turboprop/turboshaft), blades not included | ⚠️ 403 to scripted download (bot protection). **Download manually in a browser** |
| ICAO Doc 8643 type designators | Typecode → engine type (P/T/J), engine count, WTC | Public list; obtain once |
| Prop parameters (blades, RPM ranges) | Predict blade-pass frequency f₀ = RPM/60 × blades | **Gap:** no clean database; build a table for the top ~50 GA types by hand |
| readsb-provided type and `category` fields (A1 light, A7 rotorcraft) | Fallback when ICAO24 is unknown | In ADS-B dumps |

### 5.3 Terrain
| Source | Resolution / datum | Cost | Test result |
|---|---|---|---|
| USGS 3DEP | 1 arc-sec (~30 m) CONUS, 1/3 arc-sec (~10 m), 1 m lidar partial; NAVD88 | Free | ✅ TNM API 200 |
| Copernicus DEM GLO-30 / GLO-90 | Global 30/90 m; EGM2008 geoid | Free (AWS open data) | ✅ S3 200 |
| SRTM / NASADEM | 30 m, 60°N–56°S | Free (Earthdata login) | not tested |
| Geoid model EGM2008 / GEOID18 | Orthometric → ellipsoid height conversion | Free (PROJ grids) | via pyproj |

### 5.4 Atmosphere (sound speed, wind)
| Source | Coverage | Cost | Test result |
|---|---|---|---|
| NOAA HRRR | CONUS, 3 km, hourly, vertical levels | Free (AWS) | ✅ S3 200 |
| ERA5 (Copernicus CDS) | Global, 0.25°, hourly, pressure levels | Free, account + licence acceptance | ✅ site 200 |
| METAR/ASOS (Iowa Environmental Mesonet) | Surface T, wind, altimeter (QNH) for baro-altitude correction | Free | ✅ reachable (422 = missing params, expected) |

### 5.5 Validation / priors (optional)
| Source | Use | Cost | Test result |
|---|---|---|---|
| Own GPS logs | **Ground truth** | Free | n/a |
| NAIP aerial imagery / Mapillary | Visual check of top modes | Free (Mapillary needs token) | Mapillary 500 without token (expected) |
| Google Street View API | Same | **Paid** beyond free credit | Not needed (truth known) |
| WorldPop / NLCD | Population/land-cover prior | Free | not tested |

---

## 6. Software stack

Environment: `environment.yml` (conda-forge). ffmpeg is needed for compressed audio.

| Layer | Choice | Why |
|---|---|---|
| Environment | **Miniforge (conda-forge)**, Python 3.12 | GDAL/rasterio/PROJ install cleanly on Windows only via conda-forge |
| Audio | soundfile, librosa, scipy.signal; **ffmpeg** (MP3/AAC decode) | STFT, decoding |
| Tables | pyarrow, **DuckDB**, polars | Out-of-core queries on daily dumps |
| Geo | pyproj, rasterio, GDAL, shapely, geopandas, **h3** | Projections, DEM, indexing |
| Atmosphere | xarray, cfgrib/eccodes, cdsapi | HRRR/ERA5 GRIB |
| Compute | numpy, **numba** (CPU JIT); optional JAX/CuPy on cloud GPU | F3 and F9 kernels |
| ADS-B helpers | `traffic`, `pyopensky` | Track cleaning, Trino client |
| Output | matplotlib, folium or pydeck, rio-cogeo | Heat maps |
| Testing | pytest, hypothesis | Physics unit tests, synthetic end-to-end |
| Provenance | Data manifest (source, version, hash, date) per run | Reproducibility for publication |

---

## 7. Calculation reference (corrected)

| Quantity | Formula | Note |
|---|---|---|
| Sound speed | c = 331.3·√(1 + T/273.15) m/s | 0 °C → 331, 30 °C → 349 (±5%). **Do not hard-code 343** |
| Effective sound speed | c_eff = c + **w**·û | Wind along the propagation direction; 10 m/s ≈ 3% |
| Moving-source Doppler | f_heard = f₀ / (1 − v_r/c_eff), v_r at **emission** time, positive toward the observer | Source doc §4 had the wrong form |
| Emission → arrival | t_heard = t_emit + ‖**r**(t_emit)‖ / c_eff | Integrate forward; never shift a finished curve |
| Speed from curve | v = c·(f_a − f_r)/(f_a + f_r) | Asymptotic approach/recede; f₀ cancels |
| Emitted frequency | f₀ = 2·f_a·f_r/(f_a + f_r) | |
| Distance from slope | d ≈ f₀·v² / (c·\|df/dt\|_CPA) | Straight-line approximation; seeds the search |
| CPA (Closest Point of **Approach**) | argmin_t ‖P(t) − x‖ over 1 s resampled track | Segment-wise closed form |
| Blade-pass frequency | BPF = RPM/60 × n_blades; harmonics at k·BPF | e.g. C172: 2400 RPM × 2 blades → 80 Hz |
| Altitude | Prefer `alt_geom` (WGS84 ellipsoid). Else baro + QNH correction (METAR) | Baro error: hundreds of feet |
| Ground height | Z_ellipsoid = Z_DEM + N_geoid | N ≈ −10 to −40 m in CONUS |
| STFT resolution | Δf_bin = fs/N_fft; window = N_fft/fs | 48 kHz, N = 16384 → 2.9 Hz bins, 0.34 s window |
| Spreading loss | L = 20·log₁₀(d/d₀) + α(f)·d (ISO 9613-1) | Audibility model input |

---

## 8. Gates (go / no-go)

| Gate | Pass condition | Unblocks |
|---|---|---|
| **G0 Environment** | Miniforge env with all §6 packages importable; ffmpeg works | Everything |
| **G1 Data access** | Download and parse 1 day of adsb.lol; 1 DEM tile; 1 HRRR hour; FAA DB downloaded manually. OpenSky is optional, not required to pass | M2 |
| **G2 Physics unit tests** | Synthetic straight-line flyover: simulator matches the analytic Doppler curve to < 0.1 Hz; v, f₀, d recovered within 2% | M4 |
| **G3 Synthetic end-to-end** | Synthesize audio from a **real** ADS-B track at a chosen point, add noise and MP3 compression; pipeline recovers the point in a 50 km box | M1, M3–M6 |
| **G4 Funnel benchmark** | F1–F8 on one real CONUS day: measured runtime, disk, and survivors per step; update the §3 table | Scale decisions |
| **G5 First real recording** | A real flyover is detected in own audio and matched to the correct ADS-B flight at the known location | Real-data claims |
| **G6 Regional search** | Own recordings located within a ~200 km box; S2 met | — |
| **G7 Calibration** | Coverage curve (S1) on ≥ 20 recordings | Credible-region claims |
| **G8 CONUS search** | Full-CONUS day search meets S4 | — |
| **G9 Global** | Coverage-limited; report per-region expected performance | — |

Recommended order: G0 → G1 (start OpenSky application on day 1) → G2 → G3 → G4, collecting recordings in parallel for G5+.

**Status (2026-11-07).** G0 and G2 passed. G1 passed except the manual FAA download. G3 is split in two:
- **G3a, passed.** Synthetic ADS-B-like tracks (straight, 5 s sampling, flat ground, no MP3) in a 50 km box, with M1 run on the rendered audio. Re-run on this machine: 8 of 8 runs passed (seeds 1–5; event spacing 300/600/920/1500 s). Refined peak error 2–7 m. Coarse 200 m peak error 144 m, the grid offset in every run. Truth was inside the 50/90/95% regions in every run. The 90% region was 16–18 cells (about 0.7 km²). `run_search` took about 6 s, and the test suite was 22 tests. Results: `results/v1/`. Method notes: `README_G3.md`.
  *Re-run 2026-11-08 after the round-1 fixes:* 8 of 8 passed again, refined error 2–13 m. The three-flight hypothesis (A+B+C) now wins every run, with mode mass ≈ 1.0. The 90% region is 18 cells. `run_search` takes 21–57 s, and the suite is 33 tests. Results: `results/v2/`.
  *Re-run 2026-11-08 after the round-2 fixes:* 8 of 8 passed, with refined errors identical to v2. The suite is 46 tests. Results: `results/v3/`.
- **G3b, 19/24 runs pass as of 2026-11-09** (matched-filter detector + density-scaled scoring budget; see [RESULTS.md](RESULTS.md)). History: first batches run 2026-11-08. Hudson 5–6/8 with a 2 Hz detector tolerance (median 150–330 m); NYC 1–3/8. The limit was M1: 0.5 Hz misses wobbly real passes, 2 Hz floods busy airspace. *2026-11-09:* the two-tier M1 (tight + gated loose fits) gives Hudson 5/8 + 5/8 and NYC 3/8 with about 5 events per clip. The remaining NYC failures are about ranking among many candidate areas (search side) and rotor recall. Details in RESULTS.md. *History:* Tracks taken from the converted 2026-11-06 day, rendered with noise and MP3, plus wind and a sound-speed mismatch. This is what the gate asks for. The round-2 fixes (time filter, trim to box + rmax + 3 km, corridor size estimate) let a real 50 km box through the guard with a 1 h window: Hudson Valley has 10 segments and 0.13 M estimated entries; NYC has 124–152 segments and 2.9–3.1 M estimated entries against a 5 M cap. Loading takes 2.4 s and the CPA table 1.5 s (RESULTS.md). A whole-day load is still refused. *v3 (jet deferral, 55 tests, G3a identical, `results/v4/`):* deferral defers no jets on the real data (10,000 ft ceiling; NYC jets are low), so the busy-airspace limit stays. The practical lever is the size cap. Entries measure 48 B, so the 5 M cap is only 240 MB; recalibrate it at G4 (RESULTS.md). G3a shares its sound speed, audibility model and straight tracks with the renderer, so it tests the code, not the physics assumptions.
- **Caveat.** The credible-region size in G3a comes mostly from the fixed smoothing (σ = 1 cell) and the evidence cap (§9.3). Coverage is not calibrated until T4.

---

## 9. Testing phase

Testing runs alongside the gates. Each test has a question to answer, metrics, and a list of parameters it is allowed to tune. **Rule: tune on a tuning set, report on a held-out set.** Recordings are split before tuning starts (about 60/40) and the held-out set is never used to choose parameters. This prevents overfitting to our own recordings.

### 9.1 Test ladder

| Test | Gate | Data | What we aim to discover | Metrics |
|---|---|---|---|---|
| **T1 Physics check** | G2 | Synthetic straight and turning flights | Is the simulator correct? How fine must tracks be resampled and time-stepped? | Curve error vs. analytic (Hz); recovered v, f₀, d error (%) |
| **T2 Synthetic audio** | G3 | Audio rendered from **real** ADS-B tracks at chosen points, plus noise, MP3/AAC, wind | Detection limits (SNR, distance); best STFT settings; how many false hypotheses survive the funnel; how often mirror ambiguity appears | Detection recall/precision; rank of true location; survivors per funnel step |
| **T3 Field calibration** | G5 | Own recordings at known GPS point and synced time | Audibility range by aircraft type and weather; which harmonics survive on our mic; ADS-B timestamp offset; share of heard aircraft missing from ADS-B | Detected vs. expected passes; timing residual (s); harmonic survival table |
| **T4 Blind localization** | G6, G7 | Held-out own recordings, truth hidden from the pipeline | Real accuracy; whether confidence outlines are honest; false-peak rate | Median and 90th-percentile error (m); calibration curve (S1); top-mode hit rate |
| **T5 Stress tests** | G6–G7 | Variants of T3/T4 recordings | Where it breaks: single event, unknown time, cuts, compression, phone mic, busy airport vs. rural, wind | Error and "not sure" rate per condition |
| **T6 Scale benchmark** | G4, G8 | Full CONUS days | Runtime, memory, disk; where CPU vs. GPU pays off | Wall time per funnel step; peak RAM; cache size |

### 9.2 Recording campaign (feeds T3–T5)
- **Sites:** next to a GA airport (busy), suburban, rural. Note distance to the nearest airport.
- **Sessions:** ≥ 30 min continuous, uncut. Target ≥ 20 sessions for calibration, then more for the held-out set.
- **Equipment:** field recorder with windscreen as the main source. Optionally a phone recording at the same time for the mic comparison.
- **Ground truth log:** GPS position, start time from a GPS- or NTP-synced clock, weather notes (wind, temperature), and a written note of each plane heard or seen.
- **Truth labels afterwards:** match each heard pass to its ADS-B flight by hand. This gives true f₀, v, d and CPA time for every event.

### 9.3 Tunable parameters

| Parameter | Starting value | Tuned by | Tuned to optimize |
|---|---|---|---|
| Track resample interval | 1 s | T1 | Curve error < 0.1 Hz at least cost |
| STFT window N_fft / hop | 16384 / 2048 at 48 kHz | T2, T3 | Detection recall at low SNR |
| Analysis band | 40–500 Hz | T3 | Covers the surviving harmonics on our mic |
| Harmonic set scored | 1×, 2×, 3× | T3 | Score separation, true vs. false |
| Speed / distance bounds width | ±20% around M1 estimate | T2, T3 | Keep true flight in ≥ 99% of cases, prune the most |
| Δt tolerance ε | 5 s | T2, T3 | Same, driven by measured timing error |
| ADS-B timing offset prior | 0 ± 1 s | T3 | Measured residual per data source |
| Corridor width R_max | 10 km | T3 | Measured audibility range |
| Audibility model (source level, falloff, noise floor) | Synth model: 1/d spreading + 4 dB/km excess, SNR vs. noise 0.02, logistic scale 3 dB; class levels prop/turboprop/rotor 1.0, jet 0.3, unknown 0.5 | T3 | Predicted vs. actual "heard" passes |
| Silence handling | Soft P(audible) penalty, no hard cutoff (replaces threshold p*) | T3, T4 | False rejections of true location ≈ 0 |
| Evidence cap N_eff per event | 50 frames | T4 | Calibration (S1); uncapped values gave a one-cell posterior in G3a |
| Score scale (dB per log-likelihood unit) | 10 dB | T4 | Calibration (S1) |
| Posterior smoothing σ | 1 grid cell | T4, T6 | Coverage without over-widening; shrink once systematic errors are measured |
| Coarse grid spacing | 200 m | T2, T6 | Miss rate vs. runtime |
| f₀ / δt sweep range and step | ±3 Hz / ±5 s | T2, T4 | Smallest sweep that still finds the peak |
| Sound-speed model | Weather-corrected vs. fixed 343 m/s | T4 | Does weather improve error? Keep only if yes |
| Score → likelihood calibration curve | Fitted | T4 (tuning set) | Calibration curve on the diagonal |
| Likelihood temperature | 1.0 | T4 | Calibration (S1) |
| Prior (uniform vs. population) | Uniform | T4 | Report both; pick if consistently better |
| Refinement stop rule | Grid < ⅓ posterior width | T4, T6 | No accuracy gain from finer grids |
| "Not sure" threshold (minimum evidence) | Fitted | T5 | Wrong-but-confident answers ≈ 0 (S3) |
| Tile size, cache format, CPU/GPU split | Measured | T6 | Fits in 64 GB RAM; meets S4 |

### 9.4 Decision rules from test results
- **T1 fails:** fix the physics before anything else. No real data is used until T1 passes.
- **T2 recovers the true point less than ~80% of the time** at realistic SNR: improve M1 detection or the funnel before field work.
- **T3 shows the low harmonics never survive on our mic:** switch the recorder or change the band, then re-run T3.
- **T3 timing residual > 2 s:** tighten track interpolation or prefer the better ADS-B source. Widen ε if neither helps.
- **T4 calibration curve is off the diagonal:** refit the likelihood temperature on the tuning set only.
- **T5 shows a condition always fails:** document it as a limitation and make the pipeline answer "not sure" in that condition.

---

## 10. Gaps in understanding (research questions)

1. **Audibility range** of GA props, turboprops and helicopters vs. distance, ambient noise and weather. This drives the silence gate and corridor width. Measure it from own recordings.
2. **Harmonic identification:** is the tracked line the BPF, an engine firing harmonic, or a higher multiple? **Lower risk than first thought:** v and d come out the same from any harmonic, so localization geometry is unaffected. It matters for matching an aircraft type's predicted f₀, for scoring the whole harmonic stack, and for telling one aircraft's harmonics from two aircraft. Handle it by scoring 1×/2×/3× hypotheses; measure harmonic survival in T3.
3. **ADS-B timestamp accuracy:** receiver clock errors of 0.5–1 s become ~170–340 m along-track error. Quantify per source (MLAT vs. direct, OpenSky vs. adsb.lol).
4. **How many relevant aircraft are invisible:** GA without ADS-B Out (not required in most Class E/G below 10,000 ft), LADD-blocked, or PIA addresses (type lookup fails). An unseen aircraft breaks the silence gate and creates unexplained events.
5. **Refraction and shadow zones:** temperature inversions and wind gradients bend sound. Is straight-ray propagation good enough at < 6 km?
6. **Ground-reflection fringes:** noise or a usable signal (source-height estimation)?
7. **Score → likelihood calibration:** how many recordings are needed for stable calibration?
8. **Uniqueness statistics:** in busy airspace, how often do false hypotheses outscore truth? Simulate with G3.
9. **Recorder effects:** phone mics often high-pass at ~80–100 Hz and apply AGC and noise suppression, which can erase or distort the band of interest. Compare phone vs. field recorder.
10. **Claims in the source video are unverified** (e.g. "~80% false-positive removal" from the silence gate, backyard-level precision). Treat them as hypotheses to test.
11. **Opportunity: how often can a recording be located at all?** First estimate, 2026-11-06 (weekday, November), from `scripts/opportunity_map.py`. A pass is a prop, turboprop or rotor aircraft within about 3 km horizontally and ≤ 1,500 m AGL, counted in the 08–20 local daytime window. Only **0.8%** of CONUS land gives a 30-minute recording a ≥ 50% chance of catching ≥ 2 passes, and **4.2%** gives at least a 10% chance. 10-minute recordings get 0.1% and 1.1%.

    | Place | Passes per hour | P(≥ 2 passes in 30 min) |
    |---|---|---|
    | Orlando | 7.2 | 85% |
    | Chicago | 3.4 | 46% |
    | Denver | 2.8 | 46% |
    | Manhattan | 2.8 | 35% |
    | Richmond | 0.6 | 8% |
    | Rural Kansas / Nebraska | 0 | 0% |

    Hub airports are holes, because small planes avoid them. Florida flight schools dominate. The method is opportunistic: validation recordings should be planned with the map, and in-the-wild applicability must be reported as this fraction.

    Known biases:
    - one day only;
    - ADS-B gaps;
    - the 10,000 ft MSL ceiling undercounts mountain areas;
    - a fixed 3 km hearing rule;
    - one pass per aircraft per spot.

    Map generated by `scripts/opportunity_web.py` (outputs are not distributed).

---

## 11. Limitations and barriers

### Computational
- Single-day CONUS is CPU-feasible (estimate). World scale and an **unknown date** multiply cost by ~5–10× and by N_days respectively.
- No CUDA GPU locally. F9 at scale may need a rented GPU (order of $1–3/h) or a many-core VM.
- Disk: 147 GB free is not enough for global multi-day ADS-B plus a 30 m global DEM. Budget 1–2 TB for world scale.

### Paid or access barriers
| Item | Status |
|---|---|
| OpenSky historical | Free but **approval required**; lead time unknown. Apply now |
| ADS-B Exchange full history | Paid; avoidable (adsb.lol) |
| FlightAware | Paid; avoidable |
| Google Street View | Paid; avoidable (truth known; Mapillary/NAIP free) |
| ERA5 | Free, account required |
| Field recorder + windscreen | ~$100–300 one-time; strongly recommended over phone |
| Cloud compute | Optional; pay-as-you-go |

### Data availability
- ADS-B is **crowd-sourced**, so low-altitude coverage is poor in rural and mountain areas. That is exactly where prop aircraft fly low, so performance will vary strongly by region. Produce a coverage map alongside every result.
- Outside the US and Europe, GA ADS-B equipage and receiver density drop sharply. "World" capability is coverage-limited, not compute-limited.
- Atmospheric models are coarse (3 km / 25 km) relative to local microclimate.

### Physical / method
- Mirror ambiguity and single-event degeneracy (§1). Some recordings are fundamentally non-unique.
- Needs ≥ 1 clear prop or rotor pass, ideally 2 non-parallel ones. Jet-only audio is out of scope (broadband, high altitude).
- Edits or cuts in the audio break Δt continuity. Require continuous recordings.
- Wind noise on the microphone masks the low-frequency band.

### Legal / ethical / licensing
- adsb.lol is ODbL: derived databases must be shared under ODbL with attribution.
- OpenSky terms: non-commercial use, no raw-data redistribution, citation required.
- Own recordings may capture bystanders' voices. Low-pass or trim speech before sharing datasets. Ask your institution whether IRB review applies (usually not for self-recorded environmental audio).
- Scope: the method is validated on self-collected recordings with consent.

---

## 12. Immediate next steps

1. Install Miniforge and create the env (G0).
2. Apply for OpenSky research access and create Copernicus CDS and Mapillary accounts. Download the FAA Releasable Aircraft DB manually.
3. Download one adsb.lol day; measure its size, cadence and `alt_geom` availability (G1).
4. Build the M4 simulator plus analytic tests (G2), then the synthetic end-to-end harness (G3). These need no recordings. *Done through G3a (2026-11-07).*
5. In parallel: get a recorder and run the first field sessions near a GA airport, with a GPS log and a synced clock.
6. Fix the search-module review items listed in [RESULTS.md](RESULTS.md). *Rounds 1 and 2 done 2026-11-08. Round 3 notes (non-blocking) are in RESULTS.md.*
7. G3b (first batches done 2026-11-08, see §8): make the M1 fit tolerance adapt to event density or tier event quality, then re-run Hudson and NYC with `--min-side-s 60`, plus real background noise and unknown T0. Original plan: render from real tracks in the 2026-11-06 day, with MP3, wind and a c mismatch. The real-Parquet load works (`scripts/g3b_real.py`). Pick a window and box with several props: 21:00 UTC in Hudson (18 small-plane segments) or 22:00 UTC in New York (39), from `scripts/g3b_pick_window.py`. Noon had only 1–4. Run a multi-seed batch, including a busy-airspace box, for gap 8 (uniqueness).
8. G4: benchmark the funnel on one CONUS day. The current in-memory CPA table and pure-Python pair gate are sized for a 50 km box, so don't run CONUS as-is.
