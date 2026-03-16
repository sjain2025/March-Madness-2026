## March Madness ML – 2026 Brackets

This project trains a gradient-boosted model on recent NCAA Division I seasons (with heavy weight on recent years), calibrates win probabilities, generates matchup prediction CSVs, and then uses those predictions to fill and print 2026 men’s and women’s tournament brackets.

---

## How the model works

- **Tech / libraries**
  - Python, `pandas`, `numpy`
  - `scikit-learn`:
    - `HistGradientBoostingClassifier` – tree-boosting model with `loss="log_loss"`
    - `IsotonicRegression` – post-hoc probability calibration

- **Features**
  - Source data (under `data/`):
    - `*RegularSeasonDetailedResults.csv` – per-game boxscores (offense & defense).
    - `*NCAATourneyCompactResults.csv` – tournament results (labels).
    - `*NCAATourneySeeds.csv` – seeds, used to add a seed-difference feature.
    - `MRankings.csv` – men’s Massey-style rankings, used as extra features.
  - For each `(Season, TeamID)`:
    - Per-game offensive stats: points, FGM/FGA, 3PM/3PA, FTM/FTA, OR, DR, assists, TOs, steals, blocks, fouls.
    - Per-game defensive stats: same stats allowed to opponents.
    - Average scoring margin and a win-rate proxy.
  - For **men**, adds late-season ranking features from `MRankings.csv`.
  - Training rows:
    - Each tourney game produces two rows: `(winner, loser, y=1)` and `(loser, winner, y=0)`.
    - Features are **differences**: `Team1_feature − Team2_feature` for all above.
    - Adds a **seed-difference** feature from `*NCAATourneySeeds.csv`.

- **Training & weighting**
  - Uses all seasons up to a chosen `train_end_season` (2025 by default).
  - Trains only on the **last 5 seasons before the calibration year**, with explicit season weights:
    - last season: 75%
    - year −1: 18%
    - year −2: 5%
    - year −3: 1%
    - year −4: 1%
  - Applies these as `sample_weight` when fitting `HistGradientBoostingClassifier`.
  - Uses **season-based walk‑forward validation**, optimizing **log loss** instead of accuracy.

- **Calibration**
  - Fit the boosted model on weighted data from pre‑calibration seasons.
  - Get raw probabilities on the calibration season (2025).
  - Fit `IsotonicRegression` so calibrated probabilities better match observed win frequencies.

- **Prediction CSVs**
  - Implemented in `make_predictions.py`.
  - For a target season (e.g. 2026), it:
    - Builds season features for all teams in `MTeams.csv` / `WTeams.csv`.
    - Evaluates every unordered pair of teams with the calibrated model.
    - Writes:
      - `predictions/MNCAATourneyPredictions.csv`
      - `predictions/WNCAATourneyPredictions.csv`
    - Format: columns `WTeamID,LTeamID`, where `WTeamID` is whichever team has calibrated `P(Team1 wins) ≥ 0.5`.

- **Brackets**
  - Implemented in `madness.py`.
  - Loads predictions from `predictions/` and fills a standard 68‑team bracket.
  - For **Men’s 2026**, seeding is forced to match the ESPN 2026 bracket (including correct First Four teams).
  - For **Women’s 2026**, seeding comes from `WNCAATourneySeeds.csv` (with a simple fallback if 2026 seeds are missing).

---

## How to run

From the `MMML` directory:

- **Generate / refresh 2026 prediction CSVs**

  ```bash
  python3 make_predictions.py --season 2026 --train_end_season 2025 --out_dir predictions
  ```

  This writes:

  - `predictions/MNCAATourneyPredictions.csv`
  - `predictions/WNCAATourneyPredictions.csv`

- **Print the 2026 brackets (men’s and women’s)**

  ```bash
  python3 madness.py
  ```

  This will:

  - Ensure the prediction files exist in `predictions/` (and generate them if not).
  - Seed and print the **men’s 2026** bracket using the ESPN first‑round matchups and your model’s winners.
  - Seed and print the **women’s 2026** bracket using `WNCAATourneySeeds.csv` (or a simple synthetic seeding) and your model’s winners.

