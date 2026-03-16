# March Madness Machine Learning (MMML) Model

This project trains a gradient-boosted model on recent NCAA Division I seasons (with heavy weight on recent years), calibrates win probabilities, generates matchup prediction CSVs, and then uses those predictions to fill and print 2026 men’s and women’s tournament brackets.

---

## Technologies Used

- **Language**: Python
- **Core libraries**: `pandas`, `numpy`, `scikit-learn`
- **Models**: `HistGradientBoostingClassifier` (tree-based classifier with log-loss) + `IsotonicRegression` (probability calibration)
- **Data**: NCAA men’s and women’s regular-season and tournament CSVs in `data/`

---

## Model Overview

- **Goal**: Predict win probabilities for NCAA tournament matchups and use them to fill 2026 men’s and women’s brackets.
- **Inputs / features** (per team and season):
  - Aggregated, **recency-weighted** box-score stats from `*RegularSeasonDetailedResults.csv` (offense and defense), where late-season games receive higher weight to capture momentum.
  - Seed information from `*NCAATourneySeeds.csv` (seed difference between teams).
  - For men, extra ranking features from `MRankings.csv`.
- **Training setup**:
  - Builds paired rows from historical tournament games: winner vs loser and loser vs winner.
  - Uses feature differences (Team1 − Team2) plus seed difference as the input vector.
  - Trains on recent seasons, with heavier weights on the most recent years (season-level) and more emphasis on late-season games within each season (game-level momentum).
- **Calibration and predictions**:
  - Calibrates raw gradient-boosting probabilities with isotonic regression on a held-out season.
  - For a target season (e.g., 2026), scores all team pairs and writes:
    - `predictions/MNCAATourneyPredictions.csv`
    - `predictions/WNCAATourneyPredictions.csv`
  - `madness.py` then reads these files and fills a 68-team bracket for men and women.

---

## How to Run

From the `MMML` directory (use `python3` or `python` depending on your setup). Make sure you have `pandas`, `numpy`, and `scikit-learn` installed (for example: `pip install pandas numpy scikit-learn`).

1. **Generate or refresh 2026 prediction CSVs**:

   ```bash
   python3 make_predictions.py --season 2026 --train_end_season 2025 --out_dir predictions
   ```

   This writes `MNCAATourneyPredictions.csv` and `WNCAATourneyPredictions.csv` into `predictions/`.

2. **Print the 2026 men’s and women’s brackets**:

   ```bash
   python3 madness.py
   ```

   This loads predictions (and generates them if missing), seeds the brackets, and prints both brackets to the console.