# F1 Tyre Strategy

A model of Formula 1 tyre degradation and a pit-stop strategy simulator, built on public data from [OpenF1](https://openf1.org).

## The problem

Pit-stop strategy is largely decided by **tyre degradation**, the time a tyre loses on every lap.
If a compound degrades slowly it pays to extend the stint and stop once. If it degrades quickly, an extra stop
(about 20-25 s) can be worth it. But lap time depends on much more than the tyre: fuel, traffic, safety car, rain, the driver.
The work is in separating the effect of the tyre from everything else, and in knowing how far the result can be trusted.

## Data

- **Source**: the public [OpenF1](https://openf1.org) API (laps, stints, weather, race control messages).
  The repository contains the **script that downloads the data**, not the data: `python scripts/01_scarica_dati.py` saves it in `data/`.
- **Candidate races (2023)**: Sakhir, Monaco, Catalunya, Monza, Mexico City.
- **Races used**: Sakhir, Monza, Mexico City.
  - Monaco is excluded automatically: it rained in 15% of the weather measurements, and the model covers dry running only.
  - Catalunya: when I downloaded the data the API answered 401. The script skips it with a message and will include it
    once the API is available again.

## Method

**1. Building the dataset** (`src/clean.py`).
- Each lap is joined to its stint by driver and by lap interval (`lap_start ≤ lap_number ≤ lap_end`), which gives the
  compound and the **tyre age** (0 on the first lap of the stint).
- Weather is attached with `merge_asof(direction="backward")`: each lap gets the last measurement taken *before* it started.
  Timestamps are parsed with `format="ISO8601"` because OpenF1 mixes formats with and without milliseconds.

**2. Cleaning.** For each race, in this order, the code drops:
- laps without a time, pit-out and pit-in laps, and lap 1;
- laps with yellow flags;
- laps that are slow for the whole field: safety car, VSC, or weather transition. A lap counts as slow when the
  field median is more than 3% above dry race pace (8% = wet);
- each driver's outliers: laps more than 2.5% slower than the rolling median (window of 7 laps, centred) of
  *their own* laps on the same stint and in the same weather phase.

Script 02 prints a cleaning table for each race:

| Race | Laps downloaded | Clean laps |
|---|---|---|
| Sakhir | 1058 | 843 |
| Monza | 968 | 876 |
| Mexico City | 1286 | 1080 |

*Why the weather filter also looks at pace.* In the first exploration (Monaco 2023, `notebooks/journey.py`) 11 drivers
had an anomalous lap on lap 52. Instead of tuning the threshold away, I checked the data. The field median went from 78.6 s
to 89.6 s in two laps, race control reported a slippery track, and drivers started switching to intermediates.
The weather station only recorded rain about two laps later. So the rain sensor lags behind the track, and
the field's pace is a more reliable signal. The full investigation is in [`NOTES.md`](NOTES.md).

**3. Degradation model** (`src/models.py`), one per race:

```
lap_time = pace[compound] + degradation[compound] × tyre_age + fuel × lap_number
```

It is linear on purpose. Every coefficient has a meaning (seconds, seconds per lap of tyre age, seconds per race lap),
time always increases with age, and the sanity checks can be done by changing a single number.
A gradient boosting model I tried earlier produced non-monotonic curves (a 40-lap tyre "faster" than a 15-lap one),
and the optimiser exploited that noise. Compounds with fewer than 50 clean laps or fewer than 3 drivers are not estimated.

**4. Group validation.** Laps from the same stint are similar to each other, so a random train/test split would
give an over-optimistic error. Each model is trained on all races but one and tested on the one left out
(*leave-one-group-out*). The comparison is against a **baseline** that always predicts the median.

**5. Simulator** (`src/simulate.py`):
- **Pit-stop loss**: the median over all "clean" stops of (in-lap − reference) + (out-lap − reference), where the
  reference is the median of the driver's last 5 clean laps. Stops under safety car, VSC or flags are discarded.
- **Optimum**: exhaustive search over stop laps and compounds, with 1 and 2 stops. Constraints: the two-compound rule,
  stints of at least 5 laps, tyre age no higher than the maximum observed in the data.
- **Sanity checks**: four properties the simulator must satisfy. For example, with zero degradation
  2 stops must cost exactly 1 stop + one pit-stop loss. They are checked on every run.
- **Back-testing**: simulates the drivers' real strategies and compares them with the optimum and with the real race times.

## Results

### Degradation by race and compound

![Degradation by race and compound](figures/degrado_per_gara.png)

| Race | Compound | Degradation (s per lap of age) | Laps | Drivers |
|---|---|---|---|---|
| Sakhir | SOFT | 0.114 | 325 | 20 |
| Sakhir | HARD | 0.096 | 510 | 19 |
| Monza | MEDIUM | 0.065 | 402 | 19 |
| Monza | HARD | 0.060 | 474 | 19 |
| Mexico City | MEDIUM | 0.047 | 596 | 19 |
| Mexico City | HARD | 0.054 | 477 | 19 |

Sakhir, with its abrasive asphalt and heat, has the highest degradation. Note that SOFT/MEDIUM/HARD are *relative* names:
Pirelli brings different compounds to each circuit, so comparisons should be made circuit by circuit.

### "One race out" validation (MAE in seconds, times relative to the race median)

| Model | without Sakhir | without Monza | without Mexico City | Mean |
|---|---|---|---|---|
| baseline (median) | 1.012 | **0.599** | 0.857 | 0.823 |
| linear (pace + degradation per compound) | 0.871 | 0.730 | **0.708** | **0.769** |
| gradient boosting | **0.827** | 0.773 | 0.800 | 0.800 |

On average the linear model beats the baseline, but **not on Monza**: the degradation learnt on the other races does not transfer.
Degradation depends on the circuit, which is why the simulator uses one model per race.
Gradient boosting does not beat the linear model: with this data the extra complexity does not help.

### Strategies and back-testing

| Race | Pit-stop loss | Optimal strategy | Winner: gap to optimum | Model error (clean laps) | Spread between real strategies (10th-90th pct.) |
|---|---|---|---|---|---|
| Sakhir | 23.3 s | SOFT(17) > HARD(37) > HARD | +9.3 s | 0.62 s/lap | 7.4 s |
| Monza | 24.1 s | MEDIUM(24) > HARD | +1.0 s | 0.48 s/lap | 15.5 s |
| Mexico City | 21.4 s | HARD(21) > MEDIUM(46) > MEDIUM | −8.7 s | 0.69 s/lap | 25.4 s |

- **Monza**: the winner (MEDIUM until lap 20, then HARD) is 1 s from the simulated optimum.
- **Sakhir**: a one-stop is impossible with the maximum tyre ages observed (HARD 28, SOFT 23 laps out of 57).
- **Mexico City**: 1 and 2 stops differ by 1.1 s, much less than the model error, so it is a tie.
  The winner looks "faster than the optimum" because they changed tyres for free during the red flag.
- All 4 sanity checks pass in all 3 races.

## Limitations

- **Little data**: 3 dry races, and in each race only two compounds have enough laps.
- **Whole-race error**: the simulated total time is about 25 s off the real one (Sakhir, Monza).
  That is more than the difference between strategies. Most of the error is driver pace, which is about the same
  whatever the strategy, so comparing *strategies* is more reliable than the absolute time. It is still a limitation.
- **Traffic**: the simulator assumes a driver running alone. Undercut and overcut would need two drivers simulated together.
- **Safety car**: a stop under safety car costs less, but the simulator always uses the same loss.
- **Weather**: the model is trained on dry races and dry phases only.
- **Fuel**: the lap number is a proxy and also contains track evolution.
- **Tyres**: the simulator always starts from new tyres. The cold-tyre out-lap is absorbed in the pit-stop loss.
- **Relative compounds**: the same label means different tyres on different circuits.
- **Heuristic thresholds**: the cleaning thresholds (3%, 8%, 2.5%) are reasoned choices, not statistical laws.

## How to reproduce

Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/01_scarica_dati.py   # downloads from OpenF1 into data/ (cached: existing files are not downloaded again)
python scripts/02_pulisci.py        # clean laps + cleaning table -> data/giri_puliti_*.csv, data/scarti_*.csv
python scripts/03_modelli.py        # degradation per race + validation -> data/degrado_per_gara.csv, figures/degrado_per_gara.png
python scripts/04_strategia.py      # pit-stop loss, optimum, sanity checks, back-testing -> data/riepilogo_strategie.csv
```

Every number in this README is printed by one of the scripts.

### Structure

```
src/
  data.py       download from OpenF1 with a file cache
  clean.py      cleaning rules, tyre age, cleaning table
  models.py     degradation model, one-race-out validation, baseline, plot
  simulate.py   pit-stop loss, simulator, sanity checks, back-testing
scripts/        the 4 steps, run in order
notebooks/      journey.py: the first exploration on Monaco 2023 (produces the other plots in figures/)
figures/        saved plots
data/           not in the repository (recreated by script 01)
NOTES.md        project diary: what I tried, what went wrong, why each choice was made
```

## Next steps

- More races and seasons, to estimate all three compounds on every circuit.
- A **confidence interval** on the degradation (bootstrap by driver or by stint) and a sensitivity test on the cleaning thresholds.
- A **driver pace** term, to reduce the error on the race time.
- A reduced pit-stop loss under safety car, and a simulation with safety-car probabilities.
- Starting on used tyres, and a few automated tests (`pytest`) on the delicate parts: tyre age, pit-in lap, weather join.
