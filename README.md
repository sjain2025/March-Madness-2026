## March Madness Machine Learning (MMML) – 2026 Brackets

This project trains a probabilistic model to predict **all possible NCAA Division I matchups** and uses those predictions to fill out **2026 men’s and women’s brackets**.

The code is based on the MMML competition framework provided for the CMU course.

---

## Data used

All inputs live under `data/`:

- `MTeams.csv`, `WTeams.csv`: team IDs and names.
- `MRegularSeasonDetailedResults.csv`, `WRegularSeasonDetailedResults.csv`: boxscore stats for regular-season games (used for features).
- `MNCAATourneyCompactResults.csv`, `WNCAATourneyCompactResults.csv`: historical tournament results (used for labels).
- `MNCAATourneySeeds.csv`, `WNCAATourneySeeds.csv`: historical tournament seeds (used to add seed-based features).
- `MRankings.csv`: historical men’s team rankings (Massey-style), used as extra features.

Predicted matchup winners are written to:

- `predictions/MNCAATourneyPredictions.csv`
- `predictions/WNCAATourneyPredictions.csv`

Each file has:

- **Columns**: `WTeamID,LTeamID`
- **Rows**:
  - Men: 72,390 (381 choose 2)
  - Women: 71,631 (379 choose 2)

---

## Model overview

All modeling code is in `make_predictions.py`.

### 1. Per-team season features

For each season and each team, `_season_feature_snapshot` builds a feature vector from **regular-season detailed results**:

- Offensive stats (per game): points scored, FGM/FGA, 3PM/3PA, FTM/FTA, OR, DR, assists, turnovers, steals, blocks, fouls.
- Defensive stats (per game): same stats allowed to opponents.
- Extra summary stats:
  - **Average scoring margin** (`TeamScore - OppScore`).
  - **Win indicator** (`Win`), averaged over games ≈ win rate.

For **men**, it also merges in a **late-season ranking snapshot** from `MRankings.csv`:

- `AveRank`, `MedianRank`, `Quantile20`, `Quantile80` for each team, using the latest `RankingDayNum` for that season.

Result: for each `(Season, TeamID)` we have a dense feature vector summarizing how the team played that year.

### 2. Training rows from tournament games

Function: `_build_training_rows(which, seasons)`.

For each tournament game (from `*NCAATourneyCompactResults.csv`) in the selected seasons:

- Let `WTeamID` = winner, `LTeamID` = loser.
- Build **two rows**:
  - Row A: `Team1 = WTeamID`, `Team2 = LTeamID`, label `y = 1`.
  - Row B: `Team1 = LTeamID`, `Team2 = WTeamID`, label `y = 0`.
- For each row:
  - Fetch per-season feature vectors `f1` and `f2` for `Team1`, `Team2`.
  - Features are **differences**: `diff_feature = f1[feature] - f2[feature]` for all season features.
  - Add a **seed difference feature**:
    - From `*NCAATourneySeeds.csv`, compute numeric seeds (1–16) for both teams.
    - `diff_seed = seed(Team1) - seed(Team2)` (0 if seed is missing).

This yields a training matrix `X` of matchup feature differences and a label vector `y` where `1` means “Team1 wins”.

### 3. Time-based training and validation

Function: `train_time_series_gb`.

Key ideas:

- **Season-based splits** to avoid leakage:
  - Only use seasons up to `train_end_season` (default: 2025).
  - Split seasons into:
    - **Calibration season**: last usable season (default: 2025).
    - **Training seasons**: all usable seasons before the calibration season.
    - Restrict to the **last 5 training seasons**.
- **Walk-forward validation**:
  - For each validation season `vs` within the training seasons:
    - Train on earlier seasons `< vs`.
    - Validate on season `vs`.
    - Compute **log loss** for that fold.

### 4. Explicit 5-year recency weights

Instead of exponential decay, the model uses **fixed weights by season** for both CV and final training:

For a game from season `s` with `train_end_season = 2025`, we set:

- `offset = 2025 - s`
- Weights:
  - `offset == 0` (2025): **0.75**
  - `offset == 1` (2024): **0.18**
  - `offset == 2` (2023): **0.05**
  - `offset == 3` (2022): **0.01**
  - `offset == 4` (2021): **0.01**
  - Older seasons: **0.0** (ignored)

These weights are applied as `sample_weight` when fitting `HistGradientBoostingClassifier`.

### 5. Base model: HistGradientBoostingClassifier

The base classifier is `sklearn.ensemble.HistGradientBoostingClassifier`:

- Loss: `log_loss` (optimized directly for probabilistic accuracy).
- Trees: depth 4, several hundred boosting iterations.
- Regularization: L2 penalty and standard tree-based subsampling.

It outputs **uncalibrated** probabilities `p_raw = P(Team1 wins | features)` for each matchup.

### 6. Probability calibration

To get **calibrated probabilities**, the pipeline:

- Fits the base model on **pre-calibration seasons** (up to 2024).
- Computes predictions on the **calibration season** (default: 2025).
- Fits **`IsotonicRegression`** on `(p_raw, y)` pairs from 2025:
  - This learns a monotone function `g` such that `p_cal = g(p_raw)` better matches empirical frequencies.
- Uses `p_cal` for:
  - Log-loss reporting on 2025.
  - All downstream probabilities in 2026 predictions.

### 7. Generating 2026 predictions for all matchups

Function: `generate_global_predictions_csv(which, bundle, season, out_path)`.

Steps:

1. Determine all eligible teams for the given `season` from `*Teams.csv`.
2. Build per-team season features using `_season_feature_snapshot`.
3. For **every unordered pair** of teams `(t1, t2)`:
   - Build a “diff” feature vector `diff_*` by subtracting team features.
   - Run it through:
     - `base_model.predict_proba` to get `p_raw`.
     - `calibrator.transform` to get `p_cal`.
   - If `p_cal >= 0.5`, set `WTeamID = t1, LTeamID = t2`; otherwise swap.

The result is a CSV with one row per possible matchup, in submission format.

---

## Bracket generation and display

Bracket logic lives in `madness.py`.

### 1. Loading predictions

`Bracket.fill()` loads the predictions via:

- `utils.read_tourney_predictions(which='M' or 'W', folder='predictions')`
- This reads:
  - `predictions/MNCAATourneyPredictions.csv`
  - `predictions/WNCAATourneyPredictions.csv`

When filling the bracket, `get_winner(id1, id2, predictions)` chooses the winner by looking up which ID appears as `WTeamID` for that pair.

### 2. 2026 men’s bracket seeding

For the **men’s 2026 bracket**:

- If `data/MNCAATourneySeeds.csv` has no `Season == 2026` rows, `Bracket.seed()` calls `seed_2026_mens_espn()`.
- `seed_2026_mens_espn()` hard-codes the **ESPN first-round matchups** (Duke vs Siena, Arizona vs Long Island, etc.) and the **First Four** (NC State vs Texas, SMU vs Miami OH, Howard vs UMBC, Lehigh vs Prairie View).
- Team names from ESPN are resolved to `TeamID` using:
  - String normalization (case-insensitive, stripping punctuation).
  - A small alias table (`Ohio State` → `Ohio St`, `UConn` → `Connecticut`, `CA Baptist` → `Cal Baptist`, etc.).

The women’s 2026 bracket still uses the seeding logic from `WNCAATourneySeeds.csv` (with a synthetic fallback if 2026 seeds are missing).

### 3. Bracket printing

`madness.py` uses the `ShowBracket` class to render ASCII brackets for:

- 4 regions (EAST, WEST, SOUTH, MIDWEST), mapped internally as `W, X, Y, Z`.
- 6 rounds plus the championship.
- Optional “play-in” lines (First Four).

When run as a script, it:

- Ensures predictions exist in `predictions/`.
- Creates `Bracket(2026, 'M')` and `Bracket(2026, 'W')`.
- Seeds, fills, and prints both brackets.

---

## How to run it

From the `MMML` directory:

### 1. Generate 2026 predictions

This will train the model (last-5-years weighted, log-loss optimized, calibrated) and write predictions:

```bash
python3 make_predictions.py --season 2026 --train_end_season 2025 --out_dir predictions
```

This creates/overwrites:

- `predictions/MNCAATourneyPredictions.csv`
- `predictions/WNCAATourneyPredictions.csv`

### 2. View predicted 2026 brackets

To see the **men’s and women’s 2026 brackets**:

```bash
python3 madness.py
```
