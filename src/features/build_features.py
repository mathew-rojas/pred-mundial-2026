"""Build the final feature matrix from raw results + ELO ratings."""
import pandas as pd
import numpy as np
from src.features.elo import compute_elo


def _compute_team_form(df: pd.DataFrame, n: int = 5) -> tuple[pd.Series, pd.Series]:
    """Vectorized rolling form: avg points over last n matches per team (excludes current)."""
    home = df[["date", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "team", "scored", "conceded"]
    home["role"] = "home"
    home["match_idx"] = df.index

    away = df[["date", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "team", "scored", "conceded"]
    away["role"] = "away"
    away["match_idx"] = df.index

    long = pd.concat([home, away], ignore_index=True).sort_values(
        ["date", "match_idx"]
    )
    long["pts"] = np.where(
        long["scored"] > long["conceded"], 3,
        np.where(long["scored"] == long["conceded"], 1, 0),
    )
    # shift(1) excludes current match; rolling computes over previous n
    long["form"] = long.groupby("team")["pts"].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean()
    )

    home_form = (
        long[long["role"] == "home"]
        .set_index("match_idx")["form"]
        .reindex(df.index)
    )
    away_form = (
        long[long["role"] == "away"]
        .set_index("match_idx")["form"]
        .reindex(df.index)
    )
    return home_form, away_form


def build_features(results_path: str, elo_start: str = "1993-01-01") -> pd.DataFrame:
    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].fillna(False).astype(bool)

    # compute_elo filters to elo_start internally and returns final ratings
    df, _ = compute_elo(df, start_date=elo_start)

    home_form, away_form = _compute_team_form(df)
    df["home_form"] = home_form.values
    df["away_form"] = away_form.values

    df["elo_diff"] = df["home_elo_before"] - df["away_elo_before"]
    df["form_diff"] = df["home_form"] - df["away_form"]

    # Target: 0 = away win, 1 = draw, 2 = home win
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 2,
        np.where(df["home_score"] == df["away_score"], 1, 0),
    )

    return df


FEATURE_COLS = [
    "home_elo_before",
    "away_elo_before",
    "elo_diff",
    "home_form",
    "away_form",
    "form_diff",
    "neutral",
]
