# Results

Measured results to date (through 2026-11-09), the conditions behind them, and what is still open.
Method and gate definitions: [FRAMEWORK.md](FRAMEWORK.md) (§8 gates, §9 test plan).

**Rule used throughout:** tune on tuning sets, report on a held-out set, and run the held-out set once.

---

## 1. Data facts (adsb.lol, 2026-11-06)

- Raw day: 75,277 aircraft worldwide, 3.8 GB.
- After conversion (CONUS box, below 10,000 ft): 34.9 M points, 44,092 aircraft, 219,209 segments, 747 MB, ~9 min.
- Acoustic classes: 16,818 prop, 2,783 turboprop, 2,733 rotor, 13,021 jet, 8,737 unknown type.
- Low-altitude piston aircraft report every 2.9 s at the median, but every 12.5 s at the 90th percentile.
  Tracks are not 1 Hz, so the physics was tested with samples up to 15 s apart.
- Geometric (GPS) altitude is present on 96% of low piston points.

## 2. Physics (G2)

The forward model integrates sound from emission time to arrival time, with optional wind.
Error against closed-form and root-solved references is < 0.1 Hz, and ≤ 0.03 Hz on a turning flight
with track samples 15 s apart. Speed, f₀ and distance are recovered within 2%.

## 3. Synthetic end-to-end (G3a)

Five synthetic flights around a known point, audio rendered and run through detection and search in a 50 km box.

- 8/8 runs located; refined peak error 2–13 m; truth inside the 50/90/95% regions in every run.
- **Caveat:** the renderer and the search share the same sound-speed, audibility and straight-track models,
  so this tests the code, not the physics assumptions. That is what G3b is for.

## 4. Realistic renders from real flight tracks (G3b)

Harness: `scripts/g3b_render.py`. It picks ground points where ≥ 2 small aircraft passed within 3 km in a
30-minute window of the real 2026-11-06 day, renders *every* aircraft in the window, and runs detection and
search as black boxes.

Deliberate mismatches between the renderer and the search: sound speed from 10 °C (337 vs. 343 m/s assumed),
6 m/s wind, per-aircraft ADS-B clock offset σ 0.5 s, MP3 at 64 kbps.

**Pass** = truth inside the 90% credible region **and** top peak within 1 km.

| Batch | Ridge detector, best tuning | Matched filter + density-scaled budget |
|---|---|---|
| Hudson Valley (sparse traffic) | 5/8 | 7/8, median 140 m |
| Hudson Valley, second set of points | 5/8 | 5/8, median 308 m |
| New York City (busy) | 3/8 | 7/8, median ~100 m |
| **Total** | **13/24** | **19/24** |

Changes between the two columns:

1. **Detection was the bottleneck, not the search.** In runs where ≥ 2 close passes were detected, the search
   passed 5 of 6. A fixed fit tolerance couldn't work: 0.5 Hz rejected wobbly real passes, 2 Hz flooded
   New York with false events that crowded the truth out of the search's top-20 scoring budget.
2. **Replacing the per-frame ridge tracker with a whole-curve matched filter** (log-frequency template over
   the full pass) took the total from 13 to 19.
3. **A density-scaled scoring budget** (score more candidate areas when there are more) was needed alongside a
   wider distance gate. The gate alone fixed one New York run and broke two others.

Still failing: New York run 4 (a near-truth candidate is scored but loses the posterior, 5.2 km off),
and three Hudson runs that are near misses (the right area is refined, within 150–740 m, but not ranked first).

## 5. Detector evaluation

Harness: `scripts/m1_eval_build.py`, `scripts/m1_eval.py`. "Search-usable" means the detection passes the
search's speed/distance gate for the right aircraft, which is stricter than a timing match.

| Detector | Set | Search-usable / expected | False positives per hour |
|---|---|---|---|
| ridge (two-tier) | G3b renders (tuning) | 28 / 123 | 3.4 |
| matched filter | G3b renders (tuning) | 44 / 123 | 3.7 |
| ridge (two-tier) | **real phone flyovers (held out)** | 5 / 31 | – |
| matched filter | **real phone flyovers (held out)** | **22 / 31** | – |

The held-out set is 40 real phone recordings of single flyovers (Pathak, Zenodo 23424278, CC BY 4.0),
31 of which have trustworthy labels. It was not looked at during tuning and was run once.

**Why the old detector failed on real audio:** in 37 of 40 held-out clips a ridge covering the true pass existed,
but the acceptance gates rejected it. Close, fast passes smear inside a 1 s analysis window, and the clips are
shorter than the gates' minimum ridge length. The gates had been tuned on 30-minute renders and did not transfer.

**Rejected change.** Searching for a second aircraft hidden under an accepted pass's harmonics
found 6 more true passes but nearly doubled false positives, cost one end-to-end run (18/24) and made New York
searches 2–40× slower. It was reverted.

## 6. Distance bias from turning aircraft

About 1 in 6 detected distances on G3b came out 2× or more too close. Fitting the straight-pass model to each
real track's own noise-free Doppler curve showed the detector was right about the *sound*: it agrees with the
sound-implied distance to 5.5% (median). Turning or accelerating aircraft, mostly New York helicopters,
produce Doppler curves that fit a closer straight pass. The bias is one-sided (geometric / sound-implied distance:
1.05 median, 2.48 at the 90th percentile).

Consequence: comparing the detector's distance with the geometric closest approach rejects the true location
for these passes. The long-term fix is for the search to compare whole curves rather than closest-approach
numbers. The forward model already does this at the scoring stage.

## 7. Scale

Benchmark at 50 km box, 200 m grid (`scripts/bench_large.py`):

| Traffic in window | Search time | Truth |
|---|---|---|
| ≈ 1 New York hour | 27–40 s | found, 3/3 seeds |
| ≈ 2× | 49–58 s | found, 3/3 |
| ≈ 5× | 85 s | lost |
| ≈ 9× | 127 s | lost |

Memory was not the limit (peak ~0.8 GB). Density was, which led to the density-scaled scoring budget.
A full-CONUS day (gate G4) has not been run.

## 8. Real recordings

| Clip | Length | Aircraft pass? | Outcome |
|---|---|---|---|
| 1 | 3 min | no | correctly nothing detected |
| 2 | 19 min, indoors | no (nearest small aircraft 12–16 km) | correctly nothing detected |
| 3 | 17 min | no (road traffic only) | correctly nothing detected |
| 4 | 36 min | **one clear pass** | missed by the detector at the time: a household noise burst split the curve |

Clip 4's pass, extracted by hand: 117 m/s, 4.35 km, f₀ 90 Hz. The best ADS-B match is an Embraer Phenom 300,
a light *jet*, not a propeller plane. That match implies a start time ~30 min earlier than remembered, so it is
unconfirmed. One event is not enough to locate a recording anyway (mirror ambiguity).

## 9. How often can a recording be located at all?

`scripts/opportunity_map.py`, one weekday of national ADS-B. A "pass" is a prop, turboprop or rotor aircraft
within ~3 km horizontally and ≤ 1,500 m above ground, during 08–20 local time.

- **0.8%** of US land gives a 30-minute recording a ≥ 50% chance of ≥ 2 passes; 4.2% gives ≥ 10%.
- 10-minute recordings: 0.1% and 1.1%.

| Place | Passes per hour | P(≥ 2 passes in 30 min) |
|---|---|---|
| Orlando | 7.2 | 85% |
| Chicago | 3.4 | 46% |
| Denver | 2.8 | 46% |
| Manhattan | 2.8 | 35% |
| Richmond, VA | 0.6 | 8% |
| Rural Kansas / Nebraska | 0 | 0% |

Hub airports are holes (small planes avoid them). Biases: one day, ADS-B coverage gaps, a 10,000 ft ceiling
that undercounts mountain areas, a fixed 3 km hearing rule.

## 10. Open problems

1. **A real recording, located.** Needs an outdoor session somewhere the opportunity map rates well.
2. **Turning aircraft** (§6): move the search's gate from closest-approach numbers to curve comparison.
3. **Far passes.** Nothing beyond 7 km is detected on G3b, but the renders contain no usable tone there
   (4 dB/km excess loss). Real held-out flyovers keep 15–26 dB of tone SNR out to 5.7 km slant, so the
   renderer is probably pessimistic. Not yet measured.
4. **G4:** a full-CONUS day. The current in-memory closest-approach table is sized for a 50 km box.
5. **Calibration (G7):** whether the 90% region contains the truth 90% of the time needs ≥ 20 real recordings.
