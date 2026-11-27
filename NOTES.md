# Project diary

What I did, why, what I found and what went wrong, in the order it happened.
The first part (exploration on a single race) corresponds to `notebooks/journey.py`;
the rest to `src/` and `scripts/`.

## Part 1 — Exploration on one race: Monaco 2023

### Data
- Source: the OpenF1 API. Race session `9094` (Monaco 2023).
- Five tables: `sessions`, `laps` (one row per driver per lap), `stints` (one row per set of tyres),
  `race_control` (flags, safety car, official messages), `weather` (a measurement roughly every minute).
- Every table is cached as a CSV in `data/`, so the API is called only once.
- Rule I kept from the start: **never modify the downloaded tables**, build new DataFrames for each step.

### Joining laps and stints
To study degradation I need, for every lap, which tyre was fitted and how old it was.
- `merge` on `driver_number`, `meeting_key`, `session_key`. These keys are not enough by themselves,
  because a driver has several stints, so I then keep only the rows where `lap_start ≤ lap_number ≤ lap_end`.
- Result: 1516 laps and 57 stints gave 1513 rows. No lap matched more than one stint.
  **3 laps had no stint**. A likely cause is anomalous stint rows in the source data, e.g. `lap_end = 0`.
  I left the original tables untouched.
- Tyre age: `tyre_age = tyre_age_at_start + lap_number − lap_start`, so it is 0 on the first lap of a stint
  when the tyre is new. The simulator later uses the same convention.

### Adding the weather
- Timestamps: some ISO 8601 strings have milliseconds and some don't, so a fixed format failed.
  The fix is `pd.to_datetime(..., format="ISO8601", utc=True)`, applied to both laps and weather.
- Weather and laps are not measured at the same instants, so an exact `merge` is not possible.
  I used `pd.merge_asof(..., direction="backward")`: each lap gets the **last measurement taken before it started**,
  never a later one (that would use information from the future).
- I printed the columns before writing code. The rain column is `rainfall`, not `rain`. Others include
  `air_temperature`, `track_temperature`, `humidity`, `pressure`, `wind_speed` and `wind_direction`.

### Cleaning rules
| Rule | Why |
|---|---|
| drop pit-out laps | cold tyres and the pit-lane exit make them slower for reasons unrelated to wear |
| drop pit-in laps (the lap before a pit-out) | the driver slows down to enter the pit lane |
| drop lap 1 | standing start, the whole field bunched up |
| drop laps with a `YELLOW` / `DOUBLE YELLOW` flag | drivers must slow down, the lap time says nothing about the tyre |
| keep only the **dry** phase | rain changes lap times far more than tyre age does |
| drop outliers (see below) | traffic, mistakes, anything slower than the driver's own local pace |

General principle: **before blaming the tyre for a slow lap, rule out every external explanation.**

### Dry / transition / wet phases
Weather is a confounder. A lap 3 s slower in the rain is not a worn tyre. Each lap of the race is classified
using the median lap time of the whole field and the share of measurements with `rainfall`:
- **wet**: field median ≥ dry reference + 6 s;
- **transition**: some rain measured, or field median ≥ dry reference + 2 s;
- **dry**: everything else.

The dry reference is the median of the field medians on rain-free laps up to lap 49.
`fase_id` numbers consecutive blocks of the same phase, so the outlier detection never compares laps across a phase change.

### Outliers: rolling median
There is no single "normal" lap time for a whole race: it changes with tyre age, fuel and phase.
So each lap is compared with a **local** reference:
- `rolling(window=7, center=True, min_periods=5).median()` of the lap times,
  computed per `driver_number`, `stint_number`, `fase_id`. Each driver is compared only with themself,
  on the same set of tyres and in the same conditions. The window is centred, so it uses 3 laps before and 3 after.
- Median rather than mean, because the mean gets pulled by the very outliers I want to find.
- `scostamento = lap_duration − rolling_median`. Positive means slower than the local pace.
- A lap is an outlier if it is slower than the reference by more than a threshold. This is a heuristic, not a statistical law.
  I started with 2.5 s; `journey.py` now uses 2 s.

### Investigating the outliers instead of deleting them
- First run: 52 outliers out of 1320 laps. After changing the filters: 45.
  I did not record which change caused the difference.
- Plotting them by lap showed they are **not spread randomly**. They cluster at laps 49–54, lap 57 and laps 63–77,
  with some extreme deviations (about 16, 26 and 40 s).
- Count per lap: lap 52 → 11, lap 57 → 6, laps 51 and 53 → 3, lap 50 → 2, lap 49 → 1.
- One slow lap from one driver can be a mistake, traffic or a technical problem.
  **Eleven drivers slow on the same lap point to a common cause.** I listed the hypotheses:
  safety car or VSC, weather, tyre management, pit stops, or a mix.
- Decision: **do not touch the threshold or the window until I know what the outliers are.**
  Otherwise I would only be tuning the method to get a cleaner-looking plot.

#### What happened around lap 52 (checked later in the data)
| Lap | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 |
|---|---|---|---|---|---|---|---|---|
| field median lap time (s) | 78.0 | 78.2 | 78.6 | 84.2 | 89.6 | 96.0 | 107.9 | 105.7 |
| pit-out laps | 1 | 0 | 0 | 1 | 2 | 3 | 3 | 8 |

- Race control: "TRACK SURFACE SLIPPERY" on lap 52, "LOW GRIP CONDITIONS" on lap 53.
- From lap 51 drivers start switching to **intermediates** (2 on lap 51, 3 on lap 52, 3 on lap 53, 7 on lap 54).
- The weather station's `rainfall` only becomes positive around **lap 54**, two or three laps *after* the slowdown.

So the lap-52 cluster is **the rain arriving**, not tyre wear and not a safety car. The later clusters
(laps 57 and 63–77) fall after the switch to intermediates, while the track was wet.
Lessons:
1. `rainfall` lags behind the track. This is why the phase classification also uses the **pace of the whole field**,
   and doesn't rely on the rain sensor alone. With that rule these laps are classified as transition/wet and removed
   *before* the outlier step.
2. **Detecting** an anomaly is a statistical step, **explaining** it is a separate one. The first can't replace the second.
3. The rolling median is a local reference, not a measure of degradation. A statistically valid outlier
   can still have nothing to do with the tyres.

Monaco 2023 is therefore not a good race for measuring dry degradation. In Part 2 it is excluded automatically.

## Part 2 — Several races

### Choosing the races
- Candidates: Sakhir (7953), Monaco (9094), Catalunya (9102), Monza (9157), Mexico City (9181).
- **Monaco is excluded automatically**: rain in 15% of the weather measurements. The filter is `MAX_QUOTA_PIOGGIA = 0`
  (a dry race has no measurement with `rainfall`).
- **Catalunya**: the download failed on 27/11/2026 (the API returned 401 because a live session was running).
  The program skips it with a message and will include it once the API is available again.
- Mexico City has a **red flag on lap 34**. I keep the race, but take the flag into account in the pit-stop loss and in the back-testing.

### From absolute to relative thresholds
With one race, thresholds in seconds were fine. With several circuits they are not: a lap is about 78 s at Monaco
and about 100 s at Sakhir, and "up to lap 49" only made sense for Monaco. In `src/clean.py`:
- dry reference = 25th percentile of the field medians on rain-free laps (no hand-picked lap);
- transition: field median ≥ +3%, wet: ≥ +8% of the reference;
- outlier: more than 2.5% slower than the rolling median.

### Cleaning table
Printed for every race by `scripts/02_pulisci.py`. Laps kept: Sakhir 843/1058, Monza 876/968, Mexico City 1080/1286.

### Degradation by race and compound (linear model, s per lap of tyre age)
| Race | Compound | Degradation | Laps | Drivers |
|---|---|---|---|---|
| Sakhir | SOFT | 0.114 | 325 | 20 |
| Sakhir | HARD | 0.096 | 510 | 19 |
| Monza | MEDIUM | 0.065 | 402 | 19 |
| Monza | HARD | 0.060 | 474 | 19 |
| Mexico City | MEDIUM | 0.047 | 596 | 19 |
| Mexico City | HARD | 0.054 | 477 | 19 |

Plot: `figures/degrado_per_gara.png`. Data: `data/degrado_per_gara.csv`.

- Sakhir has the highest degradation (very abrasive asphalt, heat), which matches what is known about the circuit.
- Compounds with fewer than 50 clean laps or fewer than 3 drivers are excluded because the estimate is unreliable
  (MEDIUM at Sakhir: 8 laps; SOFT at Mexico City: 7 laps).
- SOFT/MEDIUM/HARD are not the same tyre on different circuits,
  so races should be compared circuit by circuit, not by compound name.

### "One race out" validation (MAE in s, times relative to the race median)
| Model | without Sakhir | without Monza | without Mexico | mean |
|---|---|---|---|---|
| baseline (median) | 1.012 | 0.599 | 0.857 | 0.823 |
| linear (pace + degradation per compound) | 0.871 | 0.730 | 0.708 | 0.769 |
| gradient boosting | 0.827 | 0.773 | 0.800 | 0.800 |

- On average the linear model beats the baseline, but **not on Monza**: the (higher) degradation learnt on the other races does not transfer.
  Degradation depends on the circuit, which is why the simulator uses **one model per race**.
- Gradient boosting does not beat the linear model: with this data the extra complexity does not help.

## Part 3 — Simulator

### Model used by the simulator
`time = pace[compound] + degradation[compound] × age + fuel × lap`, one model per race.
At first I used a HistGradientBoosting model with weather features, but the predicted curve was not monotonic:
at Monaco a 40-lap-old tyre came out faster than a 15-lap-old one, and the optimiser exploited that noise.
The linear model has three advantages: every coefficient has a meaning, time always increases with age,
and the sanity checks can be done by changing a single number.

### Age convention
Age 0 on the first lap of the stint, both in the data (`tyre_age_at_start + lap_number - lap_start`) and in the simulator.
A stint of L laps ends at age L-1, so the limit is `L - 1 ≤ maximum observed age`.

### Number of laps
`n_giri` = the last lap with at least one recorded time. OpenF1 adds a row after the finish
(the in-lap to the pits), so `lap_number.max()` gave 52 laps for Monza (it has 51) and 72 for Mexico City (it has 71).

### Pit-stop loss
- Reference: median of the driver's last 5 clean laps before the stop (at least 3).
- `loss = (in-lap − ref) + (out-lap − ref)`, then the median over all stops.
- **Discarded stops**: those whose in-lap or out-lap is "neutralised": yellow or red flags,
  a safety car message, or a field median more than 3% slower than race pace
  (this catches SC, VSC and red flags). Without this filter Mexico City gave 95.6 s (red flag)
  and Monaco 62.3 s (rain).
- Results: Sakhir 23.3 s (37 stops used, 13 discarded), Monza 24.1 s (25 / 0), Mexico City 21.4 s (17 / 21).

### Optimal strategies (two-compound rule, stints ≥ 5 laps, age ≤ maximum observed)
| Race | 1 stop | 2 stops | Best |
|---|---|---|---|
| Sakhir | impossible (HARD max 28, SOFT max 23 laps, race of 57) | SOFT(17) > HARD(37) > HARD | 2 stops |
| Monza | MEDIUM(24) > HARD, 4456.1 s | +9.7 s | 1 stop |
| Mexico City | MEDIUM(38) > HARD, 5989.7 s | HARD(21) > MEDIUM(46) > MEDIUM, −1.1 s | practically a tie |

At Mexico City 1 and 2 stops differ by 1.1 s, much less than the model error, so it is a tie.

### Sanity checks
All 4 pass in all 3 races (they run every time, without the age limit):
1. the pit-stop loss (15/20/25 s) doesn't move the stop lap and adds exactly the difference to the total time;
2. "2 stops − 1 stop" grows by exactly 10 s when the loss goes from 10 to 20 to 30 s;
3. with zero degradation, 2 stops = 1 stop + loss (20.00 s);
4. doubling the degradation of a compound that is used makes the time worse and changes the plan.

### Back-testing
| Race | Winner: gap to optimum | Model error (clean laps) | Simulated vs real error (whole race, median) | Spread between real strategies (10th-90th pct.) |
|---|---|---|---|---|
| Sakhir | +9.3 s | 0.62 s/lap | 24.4 s | 7.4 s |
| Monza | +1.0 s | 0.48 s/lap | 25.4 s | 15.5 s |
| Mexico City | −8.7 s | 0.69 s/lap | n/a (red flag) | 25.4 s |

- At Monza the winner (MEDIUM until lap 20, then HARD) is 1 s from the optimum: a practically optimal strategy.
- At Mexico City the winner looks faster than the "optimum" because they changed tyres for free during the red flag
  (in the back-testing those stops cost 0). Stints that OpenF1 splits at the restart are merged back together.
- **The whole-race error (~25 s) is larger than the difference between strategies (7-15 s).**
  Much of that error is driver pace: the model describes an "average" driver,
  so the front-runners are faster and the back-markers slower. The pace gap is almost the same whatever the strategy,
  so comparing **strategies for the same driver** is more reliable than the absolute time. It is still a limitation.

### What the simulator doesn't know
- **Traffic**: it simulates a driver running alone.
- **Safety car and flags**: a stop under safety car costs less; the simulator always uses the same loss
  (except for the red flag in the back-testing).
- **Weather**: the model is trained on dry races and dry phases only.
- **Fuel**: the lap number is a proxy and also contains track evolution.
- **Cold tyres**: the slower out-lap is absorbed in the pit-stop loss.
- **Used tyres**: the simulator always starts from new tyres.
- **Undercut and overcut**: these would need two drivers simulated together.
- **Driver pace**: there is a single average driver per race.

## Part 4 — Repository structure

All the code used to be in a single `main.py`. Now it is split like this:

| File | Contents |
|---|---|
| `src/data.py` | `scarica` (CSV cache), `carica_gara` (skips races with rain or that can't be downloaded), paths and candidate races |
| `src/clean.py` | `segna_soste`, `prepara_gara` (cleaning rules + cleaning table), `numero_giri`, thresholds |
| `src/models.py` | `ModelloDegrado`, `mescole_stimabili`, `allena_gara`, `mae_giri_puliti`, one-race-out validation, plot |
| `src/simulate.py` | pit-stop loss, `migliore_strategia`, sanity checks, back-testing |
| `scripts/01..04` | the four steps, run in order |
| `notebooks/journey.py` | the first exploration on Monaco (Part 1) |

- The scripts pass results to each other **through files** in `data/`: script 02 writes `giri_puliti_<race>.csv` and
  `scarti_<race>.csv`, scripts 03 and 04 read them. So I can re-run only the step I need.
- Each script adds the project folder to `sys.path`, so `from src.clean import prepara_gara`
  also works when running `python scripts/02_pulisci.py` from the terminal (PyCharm does this by itself).
- The paths of `data/` and `figures/` are computed in `src/data.py` (`Path(__file__)`), not from the current folder.
- **Check**: I compared the output of the 4 scripts with the old `main.py`. All the numbers are identical,
  and `data/degrado_per_gara.csv` is byte-for-byte the same.
- `requirements.txt` lists only the 5 libraries the code uses (with the `.venv` versions), not the whole `pip freeze`,
  which also included Jupyter and FastF1.
