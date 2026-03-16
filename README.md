# March Madness Machine Learning (MMML) Model

## Summary

This project builds win-probability predictions for NCAA Division I tournament matchups using recency-weighted regular-season features, a **HistGradientBoostingClassifier**, and **IsotonicRegression** calibration. It writes prediction CSVs for every possible matchup, then uses **madness.py** to fill and print the 2026 men’s and women’s 68-team brackets.

---

## Technologies Used

- **Language**: Python
- **Data & numerics**: **pandas**, **numpy**
- **ML & calibration**: **scikit-learn** — **HistGradientBoostingClassifier** (log-loss, tree-based), **IsotonicRegression** (probability calibration)
- **Validation**: walk-forward cross-validation over seasons to tune **recency-weighted** season importance
- **Data sources**: NCAA men’s and women’s regular-season and tournament CSVs in `data/` (compact and detailed results, seeds, and men’s rankings)

---

## Model Overview

The pipeline uses **gradient-boosted decision trees** (**HistGradientBoostingClassifier**) for **binary classification**: given a matchup, predict whether Team1 beats Team2. Training is **supervised** on historical tournament outcomes. Each game yields **two training examples** (winner vs loser and loser vs winner) so the model sees both orderings; the input vector is **differential features** (Team1 stats minus Team2 stats), which makes the problem symmetric and focuses the learner on relative strength. **Sample weighting** applies **exponential decay** by season (recent years weighted more) and by game order within a season (late-season games weighted more) to emphasize recency. **Walk-forward cross-validation** over seasons tunes the decay so that season weights minimize **log-loss** on held-out years. This is **tabular ML**—no natural language or computer vision; all inputs are numeric per-team and per-matchup summaries.

Raw classifier outputs are **probability-calibrated** with **IsotonicRegression** on a single held-out season so predicted win chances are better calibrated. The calibrated model scores all team pairs for the target season and writes prediction CSVs; **madness.py** then fills the 68-team men’s and women’s brackets from those probabilities.

---

## How to Run

Run everything from the `MMML` directory. Install dependencies: `pip install pandas numpy scikit-learn`.

1. **Generate 2026 prediction CSVs** (writes `MNCAATourneyPredictions.csv` and `WNCAATourneyPredictions.csv` into `predictions/`):

   ```bash
   python3 make_predictions.py --season 2026 --train_end_season 2025 --out_dir predictions
   ```

2. **Fill and print the 2026 brackets** (loads predictions, or generates them if missing; prints men’s and women’s brackets to the console):

   ```bash
   python3 madness.py
   ```