"""XGBoost classifier for match outcome prediction (0=away win, 1=draw, 2=home win)."""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report

from src.features.build_features import FEATURE_COLS


def train(df: pd.DataFrame, n_splits: int = 5) -> Pipeline:
    """Train on all rows that have non-null features; return fitted pipeline."""
    df = df.dropna(subset=FEATURE_COLS).copy()
    df["neutral"] = df["neutral"].astype(int)

    X = df[FEATURE_COLS].values
    y = df["result"].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="accuracy")
    print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(X, y)
    return pipeline


def predict_probs(
    pipeline: Pipeline,
    home_elo: float,
    away_elo: float,
    home_form: float,
    away_form: float,
    neutral: bool = False,
) -> dict[str, float]:
    """Return P(away_win), P(draw), P(home_win) for a single match."""
    x = np.array([[
        home_elo,
        away_elo,
        home_elo - away_elo,
        home_form,
        away_form,
        home_form - away_form,
        int(neutral),
    ]])
    probs = pipeline.predict_proba(x)[0]
    return {"away_win": probs[0], "draw": probs[1], "home_win": probs[2]}
