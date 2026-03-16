import pandas as pd
from itertools import combinations

from utils import read_data


def compute_team_strengths(which: str) -> pd.Series:
    """Very basic strength metric: average scoring margin over all detailed regular-season games."""
    df = read_data("RegularSeasonDetailedResults", which)

    # Margin for winner and loser
    df["Wmargin"] = df["WScore"] - df["LScore"]
    df["Lmargin"] = -df["Wmargin"]

    w = df[["WTeamID", "Wmargin"]].rename(columns={"WTeamID": "TeamID", "Wmargin": "margin"})
    l = df[["LTeamID", "Lmargin"]].rename(columns={"LTeamID": "TeamID", "Lmargin": "margin"})
    all_margins = pd.concat([w, l], ignore_index=True)

    strengths = all_margins.groupby("TeamID")["margin"].mean()
    return strengths


def build_global_predictions(which: str, output_path: str) -> None:
    """Create predictions for all pairs of Division 1 teams for the given gender."""
    # Men's: MTeams has TeamID + metadata; Women's: WTeams has TeamID only.
    teams = read_data("Teams", which)
    team_ids = sorted(teams["TeamID"].unique())

    strengths = compute_team_strengths(which)

    rows = []
    for t1, t2 in combinations(team_ids, 2):
        s1 = strengths.get(t1, 0.0)
        s2 = strengths.get(t2, 0.0)

        if s1 > s2:
            winner, loser = t1, t2
        elif s2 > s1:
            winner, loser = t2, t1
        else:
            # Tie-break by team ID
            winner, loser = (t1, t2) if t1 < t2 else (t2, t1)

        rows.append((winner, loser))

    preds = pd.DataFrame(rows, columns=["WTeamID", "LTeamID"])
    preds.to_csv(output_path, index=False)


def main():
    build_global_predictions("M", "data/MNCAATourneyPredictions.csv")
    build_global_predictions("W", "data/WNCAATourneyPredictions.csv")


if __name__ == "__main__":
    main()

