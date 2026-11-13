# Dopplerganger

Research pipeline for acoustic geolocation: estimates where an audio recording was made by matching the
Doppler curves of small aircraft in the audio against ADS-B flight data and terrain. Input is a recording and
a date; output is a probability heat map with credible regions and the matched flights.

A propeller aircraft passing a microphone produces a tone (blade-pass frequency and harmonics) whose Doppler
shift over time gives the aircraft's speed and closest distance. Two or more such events, matched to logged
flights and the time gap between them, constrain the observer's position.

**Status: research prototype.** Realistic renders from real flight tracks: 19 of 24 runs located
(median error ~100–300 m). Real recordings: none located to date (see [Results](#results)).

---

## How it works

```
 audio ──► M1 detect Doppler curves ──► events: time, speed, closest distance (+ uncertainty)
                                              │
 ADS-B day ──► M2a tracks (prop/rotor/jet) ───┤
 DEM ────────► M2b ground height ─────────────┤
                                              ▼
              M3 funnel: closest-approach table → speed/distance gate → pair timing gate
                         → silence gate → flight-pair hypotheses
                                              ▼
              M4 forward acoustic model (emission-time Doppler, wind)  ──►  M5 score vs. spectrogram
                                              ▼
              M6 posterior heat map, 50/90/95% credible regions, matched flights (GeoTIFF, GeoJSON, JSON)
```

The search is a **reduction funnel** (`docs/FRAMEWORK.md` §3): geometry and timing gates reduce the grid to a
small set of flight-pair hypotheses before any spectrogram scoring. The forward acoustic model runs only on
those hypotheses.

Modelling choices:

- **Mirror ambiguity.** One straight pass fixes a distance, not a side, so a single event gives two parallel
  lines. The posterior is multimodal and modes are reported separately.
- **Unknown start time.** Without a timestamp, only the gap between events constrains position.
- **Silence as evidence.** Locations where a loud aircraft should have been heard during a quiet interval are
  penalised.
- **Precision floor.** ADS-B position and timing error limit accuracy to roughly 25–100 m.

## Results

| Gate | What it tests | Result |
|---|---|---|
| G2 physics | Forward Doppler model vs. closed form | < 0.1 Hz error; speed, f₀, distance within 2% |
| G3a synthetic | Synthetic tracks, rendered audio, 50 km box | 8/8 located, 2–13 m error |
| **G3b realistic** | **Real ADS-B tracks** (Hudson Valley, New York City), rendered with MP3 compression, wind, a wrong sound speed and per-aircraft clock error | **19/24** pass (truth inside the 90% region and top peak < 1 km). Was 13/24 before the detector rewrite |
| Detector, held out | 31 real phone recordings of flyovers ([Pathak, Zenodo](https://zenodo.org/records/23424278), CC BY 4.0), run once with frozen settings | 22/31 usable detections vs. 5/31 for the previous detector |
| Real recordings | 4 field recordings, 3–36 min | 0 located. Three contain no aircraft pass; one contains a single pass |

**Applicability.** From one day of US ADS-B data, 0.8% of US land gives a 30-minute recording a ≥ 50% chance
of containing two usable small-aircraft passes (`scripts/opportunity_map.py`). Coverage is highest around
general-aviation training areas and near zero in much of the country.

Full results and open problems: [`docs/RESULTS.md`](docs/RESULTS.md).

## Repo layout

```
geoloc/          the pipeline
  audio.py         M1  Doppler-curve detection (whole-curve matched filter; older ridge tracker kept)
  telemetry.py     M2a raw adsb.lol day -> Parquet (points, aircraft, acoustic class)
  adsblol.py           reader for adsb.lol daily archives (no extraction needed)
  terrain.py       M2b ground height above the WGS84 ellipsoid (Copernicus GLO-90 + EGM2008)
  physics.py       M4  forward acoustic model and Doppler estimators
  search.py        M3-M6 funnel, scoring, posterior, outputs
  geodesy.py, synth.py, synthetic_scene.py, paths.py
scripts/         gate drivers, evaluation harnesses, benchmarks, opportunity map
tests/           physics, detector and search tests (pytest + hypothesis)
docs/
  FRAMEWORK.md     method, formulas, funnel, gates, test plan, limitations
  SEARCH.md        design notes for the search module (M3-M6)
  RESULTS.md       measured results and open problems
```

## Running it

```bash
conda env create -f environment.yml
conda activate geoloc
python check_env.py          # environment check (gate G0)
python -m pytest -q          # 62 tests, ~4 min; 1 known failure (see below)

# G3a: synthetic end-to-end, no downloads needed
python scripts/g3_synthetic.py --out results/g3_full --duration 1600 --dt-between 920 --grid 200
```

Real-data runs need data that isn't in this repo:

| Data | Source | Notes |
|---|---|---|
| ADS-B | [adsb.lol globe_history](https://github.com/adsblol/globe_history_2026) daily releases | ~3.8 GB/day. Convert with `scripts/convert_day.py <day>` (~9 min). ODbL |
| Terrain | Copernicus GLO-90 (AWS open data) | Fetched on demand by `terrain.py` |
| Weather (optional) | NOAA HRRR (AWS open data) | `scripts/g1_fetch_terrain_weather.py` |

Set `GEOLOC_DATA` to where the bulky data should live (default `D:/geoloc-data`).

**Known test failure.** `test_g3_triple_in_report_and_modes_correct` expects the synthetic scene's third
aircraft, 8 km away, to be detected. The matched-filter detector does not detect it. The test is left
unchanged because detection range beyond 7 km is an open question (`docs/RESULTS.md` §10).

## Not included

- **Personal recordings** (privacy). The evaluation harness references them as `bg1`–`bg3` background noise;
  any long recording without aircraft can be substituted.
- **Run outputs** (~1 GB of rendered audio and heat maps) and downloaded data.

## Use

Developed and tested only on synthetic audio, a public research dataset, and the author's own recordings.
Applicability is limited (see the 0.8% figure above), and cuts or edits in the audio break the method.
Do not apply it to recordings of people without their consent.

## Acknowledgements

- Method inspired by a publicly demonstrated acoustic-geolocation technique; `docs/FRAMEWORK.md` §3 checks its
  claims against measurements.
- Detector evaluation approach informed by the open-source doppler-locate and dopplertrack projects (both MIT).
  No code copied.
- Flight data © adsb.lol contributors (ODbL). Terrain: Copernicus DEM. Held-out flyovers: Pathak (CC BY 4.0).

## License

MIT. See [LICENSE](LICENSE).
