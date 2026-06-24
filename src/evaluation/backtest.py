"""
Backtest both models on a held-out tournament.
Returns a DataFrame with one row per match containing predictions and actuals.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from src.models.poisson_model import PoissonModel
from src.models import ml_model
from src.models.ensemble import ensemble_probs
from src.features.build_features import FEATURE_COLS, CORE_FEATURE_COLS

_OUTCOME_IDX = {"away_win": 0, "draw": 1, "home_win": 2}


def _argmax(probs: dict) -> int:
    return _OUTCOME_IDX[max(probs, key=probs.get)]


def run_backtest(
    features_path: str,
    tournament: str = "FIFA World Cup",
    test_year: int = 2022,
    ml_start_year: int = 2006,
) -> tuple[pd.DataFrame, PoissonModel, object]:
    """
    Train on data strictly before the tournament starts, predict all matches.

    Returns (results_df, fitted_poisson, fitted_xgb_pipeline).
    """
    df = pd.read_csv(features_path, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(bool)

    test_mask = (df["tournament"] == tournament) & (df["date"].dt.year == test_year)
    df_test = df[test_mask].dropna(subset=CORE_FEATURE_COLS).copy()

    cutoff = df_test["date"].min()
    df_train_all = df[df["date"] < cutoff].copy()
    df_train_ml  = df_train_all[df_train_all["date"].dt.year >= ml_start_year].copy()

    print(f"  Cutoff           : {cutoff.date()}  (day tournament starts)")
    print(f"  Train (Poisson)  : {len(df_train_all):,} matches")
    print(f"  Train (XGBoost)  : {len(df_train_ml):,} matches")
    print(f"  Test             : {len(df_test)} matches  ({tournament} {test_year})")

    print("\n  Fitting Poisson ...")
    poisson = PoissonModel(xi=0.003, wc_weight=2.0)
    poisson.fit(df_train_all)

    print("  Fitting XGBoost ...")
    pipeline = ml_model.train(df_train_ml, n_splits=5, wc_weight=2.0)

    rows = []
    for _, m in df_test.iterrows():
        neutral = bool(m["neutral"])
        home, away = m["home_team"], m["away_team"]

        pp = poisson.predict_outcome_probs(home, away, neutral=neutral)
        exact = poisson.predict_exact_score(home, away, neutral=neutral)
        xp = ml_model.predict_probs(
            pipeline,
            m["home_elo_before"], m["away_elo_before"],
            m["home_form"],       m["away_form"],
            neutral=neutral,
            home_wc22_shots=m.get("home_wc22_shots", float("nan")),
            away_wc22_shots=m.get("away_wc22_shots", float("nan")),
            home_wc22_sot=m.get("home_wc22_sot", float("nan")),
            away_wc22_sot=m.get("away_wc22_sot", float("nan")),
            home_wc22_possession=m.get("home_wc22_possession", float("nan")),
            away_wc22_possession=m.get("away_wc22_possession", float("nan")),
            home_fifa_rank=m.get("home_fifa_rank", float("nan")),
            away_fifa_rank=m.get("away_fifa_rank", float("nan")),
            rank_diff=m.get("rank_diff", float("nan")),
        )
        ep = ensemble_probs(pp, xp)

        rows.append({
            "date":       m["date"],
            "home_team":  home,
            "away_team":  away,
            "home_score": int(m["home_score"]),
            "away_score": int(m["away_score"]),
            "actual_result": int(m["result"]),
            "elo_diff":   m["elo_diff"],
            # probabilities
            "p_home_poisson": pp["home_win"], "p_draw_poisson": pp["draw"], "p_away_poisson": pp["away_win"],
            "p_home_xgb":     xp["home_win"], "p_draw_xgb":     xp["draw"], "p_away_xgb":     xp["away_win"],
            "p_home_ens":     ep["home_win"], "p_draw_ens":     ep["draw"], "p_away_ens":     ep["away_win"],
            # argmax predictions
            "pred_poisson": _argmax(pp),
            "pred_xgb":     _argmax(xp),
            "pred_ens":     _argmax(ep),
            # exact score (Poisson only)
            "exact_home_poisson": exact[0],
            "exact_away_poisson": exact[1],
        })

    results = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return results, poisson, pipeline


def compute_metrics(results: pd.DataFrame, model: str) -> dict:
    """Compute summary metrics for one model: 'poisson' | 'xgb' | 'ens'."""
    y_true = results["actual_result"].values
    y_pred = results[f"pred_{model}"].values
    y_prob = results[[f"p_away_{model}", f"p_draw_{model}", f"p_home_{model}"]].values

    acc   = accuracy_score(y_true, y_pred)
    ll    = log_loss(y_true, y_prob, labels=[0, 1, 2])

    # Multiclass Brier score
    n = len(y_true)
    brier = sum(
        (y_prob[i, c] - (1 if c == y_true[i] else 0)) ** 2
        for i in range(n) for c in range(3)
    ) / n

    metrics = {"model": model, "accuracy": acc, "log_loss": ll, "brier": brier}

    if model == "poisson":
        exact_match = (
            (results["exact_home_poisson"] == results["home_score"]) &
            (results["exact_away_poisson"] == results["away_score"])
        )
        metrics["exact_score_acc"] = exact_match.mean()
        metrics["goal_mae"] = (
            (results["exact_home_poisson"] - results["home_score"]).abs().mean()
            + (results["exact_away_poisson"] - results["away_score"]).abs().mean()
        ) / 2

    return metrics
