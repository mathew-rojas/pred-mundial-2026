"""Build the final feature matrix from raw results + ELO ratings."""
from pathlib import Path
import pandas as pd
import numpy as np
from src.features.elo import compute_elo
from src.features.wc_team_profiles import load_wc22_profiles
from src.features.wc_rank import load_wc26_ranks

_PROJECT_RAW = str(Path(__file__).parent.parent.parent / "data" / "raw")


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


def build_features(results_path: str, elo_start: str = "1993-01-01", raw_dir: str | None = None) -> pd.DataFrame:
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

    # WC 2022 team attack/defense profiles (pre-match reputation features)
    _raw = raw_dir if raw_dir is not None else _PROJECT_RAW
    profiles = load_wc22_profiles(_raw)
    df = df.merge(
        profiles.rename(columns={"team": "home_team",
                                 "wc22_shots_pg": "home_wc22_shots",
                                 "wc22_sot_pg": "home_wc22_sot",
                                 "wc22_possession": "home_wc22_possession"}),
        on="home_team", how="left",
    )
    df = df.merge(
        profiles.rename(columns={"team": "away_team",
                                 "wc22_shots_pg": "away_wc22_shots",
                                 "wc22_sot_pg": "away_wc22_sot",
                                 "wc22_possession": "away_wc22_possession"}),
        on="away_team", how="left",
    )

    # FIFA rank from WC 2026 dataset (pre-match, lower = better)
    ranks = load_wc26_ranks(_raw)
    df["home_fifa_rank"] = df["home_team"].map(ranks)
    df["away_fifa_rank"] = df["away_team"].map(ranks)
    df["rank_diff"] = df["home_fifa_rank"] - df["away_fifa_rank"]

    return df


CORE_FEATURE_COLS = [
    "home_elo_before",
    "away_elo_before",
    "elo_diff",
    "home_form",
    "away_form",
    "form_diff",
    "neutral",
]

FEATURE_COLS = CORE_FEATURE_COLS + [
    "home_wc22_shots",
    "away_wc22_shots",
    "home_wc22_sot",
    "away_wc22_sot",
    "home_wc22_possession",
    "away_wc22_possession",
    "home_fifa_rank",
    "away_fifa_rank",
    "rank_diff",
]
