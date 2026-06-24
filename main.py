"""Main pipeline: download → features → train Poisson + XGBoost → save → demo."""
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent))

from src.data.download import download_all
from src.features.build_features import build_features
from src.models.poisson_model import PoissonModel
from src.models import ml_model
from src.models.ensemble import ensemble_probs

DATA_RAW = Path("data/raw/results.csv")
DATA_PROCESSED = Path("data/processed/features.csv")
MODELS_DIR = Path("models")

ELO_START = "1993-01-01"
ML_START = "2006-01-01"


def save_models(
    poisson: PoissonModel,
    pipeline,
    elo_ratings: dict,
) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "poisson_params.json", "w") as f:
        json.dump(poisson.params_, f, indent=2)
    with open(MODELS_DIR / "elo_ratings.json", "w") as f:
        json.dump(elo_ratings, f, indent=2)
    joblib.dump(pipeline, MODELS_DIR / "xgb_pipeline.joblib")
    print(f"  Models saved to {MODELS_DIR}/")


def load_models():
    """Load previously saved models without retraining."""
    with open(MODELS_DIR / "poisson_params.json") as f:
        poisson_params = json.load(f)
    with open(MODELS_DIR / "elo_ratings.json") as f:
        elo_ratings = json.load(f)
    pipeline = joblib.load(MODELS_DIR / "xgb_pipeline.joblib")

    poisson = PoissonModel()
    poisson.params_ = poisson_params
    poisson.teams_ = list(poisson_params["attack"].keys())
    return poisson, pipeline, elo_ratings


def predict_match(
    home: str,
    away: str,
    poisson: PoissonModel,
    pipeline,
    elo_ratings: dict,
    neutral: bool = False,
) -> None:
    from src.features.build_features import FEATURE_COLS
    import numpy as np

    home_elo = elo_ratings.get(home, 1500.0)
    away_elo = elo_ratings.get(away, 1500.0)
    # Use neutral form value (1.5 pts/game ≈ average) when team has no recent history
    home_form = 1.5
    away_form = 1.5

    xgb_probs = ml_model.predict_probs(pipeline, home_elo, away_elo, home_form, away_form, neutral)
    poisson_probs = poisson.predict_outcome_probs(home, away, neutral)
    exact = poisson.predict_exact_score(home, away, neutral)
    final = ensemble_probs(poisson_probs, xgb_probs)

    venue = "neutral" if neutral else "home advantage"
    print(f"\n  {home} vs {away} ({venue})")
    print(f"    Most likely score : {home} {exact[0]} - {exact[1]} {away}")
    print(f"    Outcome probs     : home {final['home_win']:.1%}  draw {final['draw']:.1%}  away {final['away_win']:.1%}")


_MODEL_FILES = [
    MODELS_DIR / "poisson_params.json",
    MODELS_DIR / "xgb_pipeline.joblib",
    MODELS_DIR / "elo_ratings.json",
]


def _models_exist() -> bool:
    return all(f.exists() for f in _MODEL_FILES)


def main():
    print("=== 1. Download data ===")
    new_data = download_all()

    if not new_data and _models_exist():
        print("\nData unchanged — loading saved models (skip retrain).")
        poisson_model, pipeline, elo_ratings = load_models()
        print("\n=== Sample predictions ===")
        for home, away, neutral in [
            ("Brazil", "Argentina", True),
            ("France", "England", False),
            ("Spain", "Germany", True),
        ]:
            predict_match(home, away, poisson_model, pipeline, elo_ratings, neutral)
        print("\nDone.")
        return poisson_model, pipeline, elo_ratings

    print("\n=== 2. Build features (ELO from 1993+) ===")
    df = build_features(str(DATA_RAW), elo_start=ELO_START)
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False)
    print(f"  {len(df)} matches with features saved to {DATA_PROCESSED}")

    df_train = df[df["date"].dt.year >= int(ML_START[:4])].copy()
    print(f"  ML training set: {len(df_train)} matches ({ML_START[:4]}+)")

    print("\n=== 3. Poisson model ===")
    poisson_model = PoissonModel(xi=0.003, wc_weight=2.0)
    poisson_model.fit(df_train)
    print(f"  Fitted on {len(poisson_model.teams_)} teams")
    print(f"  Home advantage: {poisson_model.params_['home_adv']:.3f}  rho: {poisson_model.params_['rho']:.3f}")

    print("\n=== 4. XGBoost model ===")
    pipeline = ml_model.train(df_train, wc_weight=2.0)

    # Recover final ELO ratings from the full feature dataframe
    # (last elo_after values per team)
    elo_home = df[["home_team", "home_elo_after"]].rename(columns={"home_team": "team", "home_elo_after": "elo"})
    elo_away = df[["away_team", "away_elo_after"]].rename(columns={"away_team": "team", "away_elo_after": "elo"})
    elo_ratings = (
        pd.concat([elo_home, elo_away])
        .sort_values("elo")
        .drop_duplicates("team", keep="last")
        .set_index("team")["elo"]
        .to_dict()
    )

    print("\n=== 5. Save models ===")
    save_models(poisson_model, pipeline, elo_ratings)

    print("\n=== 6. Sample predictions ===")
    for home, away, neutral in [
        ("Brazil", "Argentina", True),
        ("France", "England", False),
        ("Spain", "Germany", True),
    ]:
        predict_match(home, away, poisson_model, pipeline, elo_ratings, neutral)

    print("\nDone.")
    return poisson_model, pipeline, elo_ratings


if __name__ == "__main__":
    import pandas as pd
    main()
