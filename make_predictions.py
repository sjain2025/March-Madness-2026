import argparse
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from utils import read_data


@dataclass(frozen=True)
class ModelBundle:
    base_model: HistGradientBoostingClassifier
    calibrator: IsotonicRegression
    feature_cols: list
    train_seasons: list
    calib_season: int
    recency_decay: float
    season_weights: Dict[int, float]


def _season_feature_snapshot(which: str, season: int, recency_decay: float = 0.995) -> pd.DataFrame:
    """
    Build per-team season features from regular season detailed results.

    Features are primarily efficiency and rate stats:
    - offensive and defensive rating (points per 100 possessions)
    - turnover rate, offensive rebound rate, three-point attempt rate
    - win rate and average margin
    - opponent-strength summaries derived from opponents' efficiency profiles

    Late-season momentum is modeled by exponentially down-weighting earlier
    regular-season games using the provided recency_decay parameter.
    """
    df = read_data("RegularSeasonDetailedResults", which)
    df = df[df["Season"] == season].copy()
    if df.empty:
        return pd.DataFrame(columns=["TeamID", "Season"])

    # Winner perspective
    w = df.rename(
        columns={
            "WTeamID": "TeamID",
            "LTeamID": "OppID",
            "DayNum": "DayNum",
            "WScore": "TeamScore",
            "LScore": "OppScore",
            "WFGM": "TeamFGM",
            "WFGA": "TeamFGA",
            "WFGM3": "TeamFGM3",
            "WFGA3": "TeamFGA3",
            "WFTM": "TeamFTM",
            "WFTA": "TeamFTA",
            "WOR": "TeamOR",
            "WDR": "TeamDR",
            "WAst": "TeamAst",
            "WTO": "TeamTO",
            "WStl": "TeamStl",
            "WBlk": "TeamBlk",
            "WPF": "TeamPF",
            "LFGM": "OppFGM",
            "LFGA": "OppFGA",
            "LFGM3": "OppFGM3",
            "LFGA3": "OppFGA3",
            "LFTM": "OppFTM",
            "LFTA": "OppFTA",
            "LOR": "OppOR",
            "LDR": "OppDR",
            "LAst": "OppAst",
            "LTO": "OppTO",
            "LStl": "OppStl",
            "LBlk": "OppBlk",
            "LPF": "OppPF",
        }
    )
    w["Win"] = 1

    # Loser perspective (swap)
    l = df.rename(
        columns={
            "LTeamID": "TeamID",
            "WTeamID": "OppID",
            "DayNum": "DayNum",
            "LScore": "TeamScore",
            "WScore": "OppScore",
            "LFGM": "TeamFGM",
            "LFGA": "TeamFGA",
            "LFGM3": "TeamFGM3",
            "LFGA3": "TeamFGA3",
            "LFTM": "TeamFTM",
            "LFTA": "TeamFTA",
            "LOR": "TeamOR",
            "LDR": "TeamDR",
            "LAst": "TeamAst",
            "LTO": "TeamTO",
            "LStl": "TeamStl",
            "LBlk": "TeamBlk",
            "LPF": "TeamPF",
            "WFGM": "OppFGM",
            "WFGA": "OppFGA",
            "WFGM3": "OppFGM3",
            "WFGA3": "OppFGA3",
            "WFTM": "OppFTM",
            "WFTA": "OppFTA",
            "WOR": "OppOR",
            "WDR": "OppDR",
            "WAst": "OppAst",
            "WTO": "OppTO",
            "WStl": "OppStl",
            "WBlk": "OppBlk",
            "WPF": "OppPF",
        }
    )
    l["Win"] = 0

    games = pd.concat([w, l], ignore_index=True)
    games["Margin"] = games["TeamScore"] - games["OppScore"]

    # Possessions for team and opponent
    games["TeamPoss"] = (
        games["TeamFGA"] - games["TeamOR"] + games["TeamTO"] + 0.44 * games["TeamFTA"]
    )
    games["OppPoss"] = (
        games["OppFGA"] - games["OppOR"] + games["OppTO"] + 0.44 * games["OppFTA"]
    )

    # Efficiency and rate stats (handle zero-possession edge cases)
    games["OffRating"] = np.where(
        games["TeamPoss"] > 0, 100.0 * games["TeamScore"] / games["TeamPoss"], 0.0
    )
    games["DefRating"] = np.where(
        games["OppPoss"] > 0, 100.0 * games["OppScore"] / games["OppPoss"], 0.0
    )
    games["TOVRate"] = np.where(
        games["TeamPoss"] > 0, games["TeamTO"] / games["TeamPoss"], 0.0
    )
    # Offensive rebound rate: share of available offensive rebounds
    games["ORRate"] = np.where(
        (games["TeamOR"] + games["OppDR"]) > 0,
        games["TeamOR"] / (games["TeamOR"] + games["OppDR"]),
        0.0,
    )
    # Three-point attempt rate: fraction of FGA that are 3PA
    games["ThreePARate"] = np.where(
        games["TeamFGA"] > 0, games["TeamFGA3"] / games["TeamFGA"], 0.0
    )

    # Exponential recency weighting: most recent games get weight 1.0,
    # earlier games get weight recency_decay ** (days_ago).
    max_day = games["DayNum"].max()
    days_ago = max_day - games["DayNum"]
    games["_weight"] = np.power(recency_decay, days_ago.astype(float))

    # Use a blend of traditional box-score stats (per game) and
    # possession-based efficiency/rate metrics so that the new metrics
    # inform but do not dominate the representation.
    agg_cols = [
        # Possession-based metrics
        "OffRating",
        "DefRating",
        "TOVRate",
        "ORRate",
        "ThreePARate",
        # Traditional per-game box-score stats (pace-influenced)
        "TeamScore",
        "OppScore",
        "TeamFGM",
        "TeamFGA",
        "TeamFGM3",
        "TeamFGA3",
        "TeamFTM",
        "TeamFTA",
        "TeamOR",
        "TeamDR",
        "TeamAst",
        "TeamTO",
        "TeamStl",
        "TeamBlk",
        "TeamPF",
        "OppFGM",
        "OppFGA",
        "OppFGM3",
        "OppFGA3",
        "OppFTM",
        "OppFTA",
        "OppOR",
        "OppDR",
        "OppAst",
        "OppTO",
        "OppStl",
        "OppBlk",
        "OppPF",
        # Outcomes
        "Margin",
        "Win",
    ]
    # Compute weighted per-team averages to capture momentum
    weighted = games[agg_cols].multiply(games["_weight"], axis=0)
    sum_weight = games.groupby("TeamID")["_weight"].sum()
    sum_xw = weighted.groupby(games["TeamID"]).sum()
    feats = sum_xw.div(sum_weight, axis=0).reset_index()

    # Opponent strength: for each team, compute recency-weighted averages of
    # opponents' efficiency stats (strength of schedule style features).
    opp_base = feats[
        [
            "TeamID",
            "OffRating",
            "DefRating",
            "Margin",
            "Win",
        ]
    ].rename(
        columns={
            "TeamID": "OppID",
            "OffRating": "OppOffRating",
            "DefRating": "OppDefRating",
            "Margin": "OppMargin",
            "Win": "OppWin",
        }
    )
    games_opp = games.merge(opp_base, on="OppID", how="left")
    opp_agg_cols = ["OppOffRating", "OppDefRating", "OppMargin", "OppWin"]
    weighted_opp = games_opp[opp_agg_cols].multiply(games_opp["_weight"], axis=0)
    sum_xw_opp = weighted_opp.groupby(games_opp["TeamID"]).sum()
    opp_feats = sum_xw_opp.div(sum_weight, axis=0).reset_index()
    feats = feats.merge(opp_feats, on="TeamID", how="left")
    feats.insert(1, "Season", season)

    # Men only: add ranking snapshot near tourney time if available
    if which == "M":
        ranks = read_data("Rankings", which)
        r = ranks[ranks["Season"] == season].copy()
        if not r.empty:
            # Use the latest available RankingDayNum in that season per team
            r.sort_values(["TeamID", "RankingDayNum"], inplace=True)
            r_last = r.groupby("TeamID").tail(1)[["TeamID", "AveRank", "MedianRank", "Quantile20", "Quantile80"]]
            feats = feats.merge(r_last, on="TeamID", how="left")
    return feats


def _build_training_rows(
    which: str,
    seasons: list[int],
    recency_decay: float = 0.995,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build supervised training data from tourney games.
    Label: 1 if Team1 wins.
    Features: Team1 season features minus Team2 season features + seed difference (if present).
    """
    results = read_data("NCAATourneyCompactResults", which)
    results = results[results["Season"].isin(seasons)].copy()
    if results.empty:
        raise Exception("No tourney results found for requested seasons")

    seeds = read_data("NCAATourneySeeds", which)
    seeds = seeds[seeds["Season"].isin(seasons)].copy()
    seeds["SeedNum"] = seeds["Seed"].str[1:3].astype(int)
    seed_map = seeds.set_index(["Season", "TeamID"])["SeedNum"]

    # Precompute per-season team features (with recency-weighted momentum)
    season_feats = {s: _season_feature_snapshot(which, s, recency_decay=recency_decay) for s in seasons}

    rows = []
    y = []
    season_out = []
    for _, g in results.iterrows():
        season = int(g["Season"])
        w_id = int(g["WTeamID"])
        l_id = int(g["LTeamID"])

        # Two training examples per game (swap sides) improves symmetry
        for t1, t2, label in [(w_id, l_id, 1), (l_id, w_id, 0)]:
            f = season_feats[season]
            f1 = f[f["TeamID"] == t1]
            f2 = f[f["TeamID"] == t2]
            if f1.empty or f2.empty:
                continue

            f1 = f1.iloc[0].to_dict()
            f2 = f2.iloc[0].to_dict()

            # Build diff features
            row = {"Season": season, "Team1": t1, "Team2": t2}
            ignore = {"TeamID", "Season"}
            for k in f1.keys():
                if k in ignore:
                    continue
                if k in f2:
                    row[f"diff_{k}"] = float(f1[k]) - float(f2[k])

            s1 = seed_map.get((season, t1), np.nan)
            s2 = seed_map.get((season, t2), np.nan)
            row["diff_seed"] = (float(s1) - float(s2)) if (not np.isnan(s1) and not np.isnan(s2)) else 0.0

            rows.append(row)
            y.append(label)
            season_out.append(season)

    X = pd.DataFrame(rows)
    y = np.array(y, dtype=int)
    if len(X) == 0:
        raise Exception("No training rows could be built (missing features?)")

    feature_cols = [c for c in X.columns if c.startswith("diff_")]
    return X[feature_cols], y, np.array(season_out, dtype=int)


def train_time_series_gb(
    which: str,
    train_end_season: int,
    calib_season: Optional[int] = None,
    train_years: int = 7,
    recency_decay: float = 0.995,
) -> ModelBundle:
    """
    Season-based CV: trains on earlier seasons, validates on later seasons.
    Optimizes log loss and then calibrates probabilities on the final calibration season.
    """
    results = read_data("NCAATourneyCompactResults", which)
    all_seasons = sorted(results["Season"].unique().tolist())
    reg = read_data("RegularSeasonDetailedResults", which)
    reg_seasons = set(reg["Season"].unique().tolist())
    # Only use seasons where we have regular-season features available
    usable = [s for s in all_seasons if s <= train_end_season and s in reg_seasons]
    if len(usable) < 8:
        raise Exception("Not enough seasons to train (need at least ~8)")

    calib = calib_season if calib_season is not None else usable[-1]
    train_seasons_full = [s for s in usable if s < calib]
    # Restrict to last N seasons (weighted toward recent)
    if len(train_seasons_full) > train_years:
        train_seasons = train_seasons_full[-train_years:]
    else:
        train_seasons = train_seasons_full

    # Season-based validation scores (walk-forward) using fixed 7-season weights
    val_seasons = [s for s in train_seasons if s >= train_seasons[0] + 3]
    feature_cols = None
    fold_scores = []
    for vs in val_seasons:
        tr = [s for s in train_seasons if s < vs]
        if len(tr) < 3:
            continue
        X_tr, y_tr, s_tr = _build_training_rows(which, tr, recency_decay=recency_decay)
        X_va, y_va, _ = _build_training_rows(which, [vs], recency_decay=recency_decay)
        feature_cols = list(X_tr.columns)

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=4,
            max_iter=600,
            l2_regularization=1.0,
            random_state=7,
        )
        # Fixed 7-season weights (offsets 0..6, 0=most recent):
        # 0.4, 0.25, 0.15, 0.05, 0.05, 0.05, 0.05
        season_weight = {
            0: 0.40,
            1: 0.25,
            2: 0.15,
            3: 0.05,
            4: 0.05,
            5: 0.05,
            6: 0.05,
        }
        offsets_tr = train_end_season - s_tr
        w_tr = np.array([season_weight.get(int(d), 0.0) for d in offsets_tr], dtype=float)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        p = model.predict_proba(X_va)[:, 1]
        fold_scores.append((vs, float(log_loss(y_va, p, labels=[0, 1]))))

    if fold_scores:
        avg = sum(s for _, s in fold_scores) / len(fold_scores)
        print(f"[{which}] walk-forward logloss avg={avg:.4f} folds={len(fold_scores)} last={fold_scores[-1]}")

    # Train base model on all pre-calibration seasons
    X_train, y_train, s_train = _build_training_rows(which, train_seasons, recency_decay=recency_decay)
    feature_cols = list(X_train.columns)

    base_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_depth=4,
        max_iter=900,
        l2_regularization=1.0,
        random_state=7,
    )
    offsets_train = train_end_season - s_train
    w_train = np.array([season_weight.get(int(d), 0.0) for d in offsets_train], dtype=float)
    base_model.fit(X_train, y_train, sample_weight=w_train)

    # Calibrate on the held-out calibration season (time-safe)
    X_cal, y_cal, _ = _build_training_rows(which, [calib], recency_decay=recency_decay)
    p_cal = base_model.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p_cal, y_cal)

    cal_ll = float(log_loss(y_cal, calibrator.transform(p_cal), labels=[0, 1]))
    raw_ll = float(log_loss(y_cal, p_cal, labels=[0, 1]))
    print(f"[{which}] calibration season={calib} raw_logloss={raw_ll:.4f} calibrated_logloss={cal_ll:.4f}")

    return ModelBundle(
        base_model=base_model,
        calibrator=calibrator,
        feature_cols=feature_cols,
        train_seasons=train_seasons,
        calib_season=calib,
        recency_decay=recency_decay,
        season_weights=season_weight,
    )


def _eligible_team_ids(which: str, season: int) -> list[int]:
    teams = read_data("Teams", which)
    # Restrict to teams that are active Division I participants in the
    # requested season, based on FirstD1Season / LastD1Season.
    if "FirstD1Season" in teams.columns and "LastD1Season" in teams.columns:
        active = teams[
            (teams["FirstD1Season"] <= season) & (teams["LastD1Season"] >= season)
        ]
    else:
        active = teams
    return sorted(active["TeamID"].unique().tolist())


def _season_features_by_team(which: str, season: int, recency_decay: float = 0.995) -> pd.DataFrame:
    f = _season_feature_snapshot(which, season, recency_decay=recency_decay)
    if f.empty:
        reg = read_data("RegularSeasonDetailedResults", which)
        avail = sorted(reg["Season"].unique().tolist())
        fallback = max([s for s in avail if s <= season], default=None)
        if fallback is None:
            raise Exception(f"No regular season feature data found for {which} (requested season {season})")
        print(f"[{which}] WARNING: no regular-season data for {season}; using {fallback} features instead")
        f = _season_feature_snapshot(which, fallback, recency_decay=recency_decay)
    f = f.set_index("TeamID")
    return f


def _pairwise_feature_diff(feature_df: pd.DataFrame, feature_cols_raw: list[str], t1: int, t2: int) -> np.ndarray:
    # feature_df columns include raw per-team columns, but our model expects diff_*
    f1 = feature_df.loc[t1]
    f2 = feature_df.loc[t2]
    out = []
    for col in feature_cols_raw:
        raw = col.replace("diff_", "")
        out.append(float(f1.get(raw, 0.0)) - float(f2.get(raw, 0.0)))
    return np.array(out, dtype=float)


def generate_global_predictions_csv(which: str, bundle: ModelBundle, season: int, out_path: str) -> None:
    team_ids = _eligible_team_ids(which, season)
    feats = _season_features_by_team(which, season, recency_decay=bundle.recency_decay)

    # Ensure all teams exist in feature snapshot (fill missing teams with zeros)
    feats = feats.reindex(team_ids).fillna(0.0)

    rows = []
    X_rows = []
    pairs = []
    for t1, t2 in combinations(team_ids, 2):
        pairs.append((t1, t2))
        X_rows.append(_pairwise_feature_diff(feats, bundle.feature_cols, t1, t2))

    X = pd.DataFrame(X_rows, columns=bundle.feature_cols) if len(X_rows) else pd.DataFrame(columns=bundle.feature_cols)
    p = bundle.base_model.predict_proba(X)[:, 1] if len(X) else np.array([])
    p = bundle.calibrator.transform(p) if len(p) else p

    # Decide winner deterministically by p>=0.5 for the required WTeamID/LTeamID format
    for (t1, t2), prob in zip(pairs, p):
        if prob >= 0.5:
            winner, loser = t1, t2
        else:
            winner, loser = t2, t1
        rows.append((winner, loser))

    pd.DataFrame(rows, columns=["WTeamID", "LTeamID"]).to_csv(out_path, index=False)
    print(f"[{which}] wrote {out_path} rows={len(rows)} teams={len(team_ids)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026, help="Season to generate predictions for (default: 2026)")
    ap.add_argument("--train_end_season", type=int, default=2025, help="Last season allowed in training data")
    ap.add_argument("--calib_season", type=int, default=None, help="Held-out calibration season (default: train_end_season)")
    ap.add_argument("--out_dir", type=str, default="predictions", help="Output directory for prediction CSVs")
    args = ap.parse_args()

    m_bundle = train_time_series_gb("M", train_end_season=args.train_end_season, calib_season=args.calib_season)
    w_bundle = train_time_series_gb("W", train_end_season=args.train_end_season, calib_season=args.calib_season)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    generate_global_predictions_csv("M", m_bundle, args.season, f"{args.out_dir}/MNCAATourneyPredictions.csv")
    generate_global_predictions_csv("W", w_bundle, args.season, f"{args.out_dir}/WNCAATourneyPredictions.csv")


if __name__ == "__main__":
    main()

