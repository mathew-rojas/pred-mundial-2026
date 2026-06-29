"""
Incremental ELO update (after each matchday) and full model retrain (between phases).
"""
import json
from pathlib import Path

import pandas as pd
import joblib

from src.features.elo import TOURNAMENT_K, INITIAL_RATING, K_BASE

MODELS_DIR = Path("models")


# ── Incremental ELO update ─────────────────────────────────────────────────────
def _k_factor(tournament: str) -> float:
    for key, mult in TOURNAMENT_K.items():
        if key in tournament:
            return K_BASE * mult
    return K_BASE


def _expected(r_a: float, r_b: float) -> float:
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def update_elo(
    new_results: pd.DataFrame,
    elo_ratings: dict[str, float],
    home_adv: float = 100.0,
) -> dict[str, float]:
    """
    Apply ELO updates for new_results (DataFrame with home_team, away_team,
    home_score, away_score, tournament, neutral columns).
    Returns a NEW dict (does not mutate the input).
    """
    ratings = dict(elo_ratings)
    new_results = new_results.sort_values("date")

    for _, row in new_results.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh = ratings.get(h, INITIAL_RATING)
        ra = ratings.get(a, INITIAL_RATING)

        rh_adj = rh + home_adv if not row.get("neutral", True) else rh
        exp_h = _expected(rh_adj, ra)

        hs, as_ = int(row["home_score"]), int(row["away_score"])
        if hs > as_:
            act_h = 1.0
        elif hs == as_:
            act_h = 0.5
        else:
            act_h = 0.0

        k = _k_factor(str(row.get("tournament", "")))
        ratings[h] = rh + k * (act_h - exp_h)
        ratings[a] = ra + k * ((1 - act_h) - (1 - exp_h))

    return ratings


def save_elo(elo_ratings: dict[str, float], path: str | None = None) -> None:
    dest = Path(path) if path else MODELS_DIR / "elo_ratings.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(elo_ratings, f, indent=2)
    print(f"  ELO saved -> {dest}")


def load_elo(path: str | None = None) -> dict[str, float]:
    src = Path(path) if path else MODELS_DIR / "elo_ratings.json"
    with open(src) as f:
        return json.load(f)


# ── Full model retrain (between phases) ───────────────────────────────────────
def retrain_full(
    results_path: str = "data/raw/results.csv",
    cutoff_date: str | None = None,
    elo_start: str = "1993-01-01",
    ml_start: str = "2006-01-01",
    save: bool = True,
) -> tuple:
    """
    Full retrain of Poisson + XGBoost on all data up to cutoff_date.
    If cutoff_date is None, uses all available data.
    Returns (poisson_model, xgb_pipeline, elo_ratings).
    """
    import warnings
    warnings.filterwarnings("ignore")

    from src.features.build_features import build_features
    from src.models.poisson_model import PoissonModel
    from src.models import ml_model

    print(f"  Building features (cutoff: {cutoff_date or 'all data'})...")
    df = build_features(results_path, elo_start=elo_start)

    if cutoff_date:
        df = df[df["date"] < pd.Timestamp(cutoff_date)]

    df_ml = df[df["date"].dt.year >= int(ml_start[:4])].copy()

    print(f"  Poisson: {len(df):,} matches | XGBoost: {len(df_ml):,} matches")

    print("  Fitting Poisson...")
    poisson = PoissonModel(xi=0.003, wc_weight=2.0)
    poisson.fit(df)

    print("  Fitting XGBoost...")
    pipeline = ml_model.train(df_ml, n_splits=5, wc_weight=2.0)

    # Extract latest ELO ratings
    from pandas import concat
    elo_h = df[["home_team", "home_elo_after"]].rename(columns={"home_team": "team", "home_elo_after": "elo"})
    elo_a = df[["away_team", "away_elo_after"]].rename(columns={"away_team": "team", "away_elo_after": "elo"})
    elo_ratings = (
        concat([elo_h, elo_a])
        .drop_duplicates("team", keep="last")
        .set_index("team")["elo"]
        .to_dict()
    )

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        with open(MODELS_DIR / "poisson_params.json", "w") as f:
            json.dump(poisson.params_, f, indent=2)
        joblib.dump(pipeline, MODELS_DIR / "xgb_pipeline.joblib")
        save_elo(elo_ratings)
        print("  Models saved.")

    return poisson, pipeline, elo_ratings
