"""Compute ELO ratings for every team across the match history."""
import numpy as np
import pandas as pd

INITIAL_RATING = 1500
K_BASE = 20
TOURNAMENT_K = {
    "FIFA World Cup": 2.0,
    "UEFA Euro": 1.5,
    "Copa América": 1.5,
    "AFC Asian Cup": 1.3,
    "Africa Cup of Nations": 1.3,
    "Friendly": 0.5,
}


def _k_factor(tournament: str) -> float:
    for key, mult in TOURNAMENT_K.items():
        if key in tournament:
            return K_BASE * mult
    return K_BASE


def _expected(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def _outcome(home_score: int, away_score: int) -> tuple[float, float]:
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


def compute_elo(
    df: pd.DataFrame,
    start_date: str = "1993-01-01",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Compute ELO ratings over time.

    Returns (df_with_elo_columns, final_ratings_dict).
    Only rows from start_date onward are included in the output,
    but ratings are seeded from INITIAL_RATING at that cutoff.
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}

    home_before, away_before, home_after, away_after = [], [], [], []
    keep_mask = []

    cutoff = pd.Timestamp(start_date)

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh = ratings.get(h, INITIAL_RATING)
        ra = ratings.get(a, INITIAL_RATING)

        rh_adj = rh + 100 if not row.get("neutral", False) else rh

        exp_h = _expected(rh_adj, ra)
        act_h, act_a = _outcome(row["home_score"], row["away_score"])

        k = _k_factor(row["tournament"])
        new_rh = rh + k * (act_h - exp_h)
        new_ra = ra + k * (act_a - (1 - exp_h))

        after_cutoff = row["date"] >= cutoff
        keep_mask.append(after_cutoff)
        if after_cutoff:
            home_before.append(rh)
            away_before.append(ra)
            home_after.append(new_rh)
            away_after.append(new_ra)

        ratings[h] = new_rh
        ratings[a] = new_ra

    df_out = df[keep_mask].copy().reset_index(drop=True)
    df_out["home_elo_before"] = home_before
    df_out["away_elo_before"] = away_before
    df_out["home_elo_after"] = home_after
    df_out["away_elo_after"] = away_after

    return df_out, ratings
