"""
Save predictions before matches are played, then evaluate accuracy afterwards.
Predictions are stored in data/predictions/ as CSVs keyed by phase + matchday.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

PRED_DIR = Path("data/predictions")


def save_predictions(df: pd.DataFrame, phase: str, matchday: int) -> Path:
    """
    Persist a predictions DataFrame before the matches are played.
    phase: e.g. 'group_stage', 'round_of_32', 'quarterfinal'
    matchday: 1, 2, 3, ...
    """
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{phase}_md{matchday}.csv"
    df.to_csv(path, index=False)
    print(f"  Predictions saved -> {path}")
    return path


def load_predictions(phase: str, matchday: int) -> pd.DataFrame:
    path = PRED_DIR / f"{phase}_md{matchday}.csv"
    return pd.read_csv(path)


def evaluate(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Merge predictions with actual results and compute per-match evaluation.

    predictions: output of predict_matches() with home_team, away_team, pred_score,
                 p_home_win, p_draw, p_away_win
    actuals: DataFrame with home_team, away_team, home_score, away_score

    Returns merged DataFrame with correctness columns.
    """
    # Parse predicted score
    preds = predictions.copy()
    preds[['pred_home', 'pred_away']] = preds['pred_score'].str.split('-', expand=True).astype(int)

    # Actual result label
    act = actuals[['home_team', 'away_team', 'home_score', 'away_score']].copy()
    act['home_score'] = act['home_score'].astype(int)
    act['away_score'] = act['away_score'].astype(int)
    act['actual_result'] = np.where(
        act['home_score'] > act['away_score'], 'home_win',
        np.where(act['home_score'] == act['away_score'], 'draw', 'away_win')
    )

    merged = preds.merge(act, on=['home_team', 'away_team'], how='left')

    # Predicted result from ensemble probabilities (argmax) — best outcome predictor.
    prob_cols = {'home_win': 'p_home_win', 'draw': 'p_draw', 'away_win': 'p_away_win'}
    def _argmax_result(row):
        return max(prob_cols, key=lambda k: row[prob_cols[k]])
    merged['pred_result'] = merged.apply(_argmax_result, axis=1)

    merged['result_correct']      = merged['pred_result'] == merged['actual_result']
    merged['exact_score_correct'] = (
        (merged['pred_home'] == merged['home_score']) &
        (merged['pred_away'] == merged['away_score'])
    )
    merged['home_goal_err'] = (merged['pred_home'] - merged['home_score']).abs()
    merged['away_goal_err'] = (merged['pred_away'] - merged['away_score']).abs()

    return merged


def summary_metrics(eval_df: pd.DataFrame) -> dict:
    """Compute summary metrics from an evaluated DataFrame."""
    n = len(eval_df.dropna(subset=['actual_result']))
    if n == 0:
        return {}
    valid = eval_df.dropna(subset=['actual_result'])
    return {
        'n_matches':        n,
        'result_accuracy':  valid['result_correct'].mean(),
        'exact_score_pct':  valid['exact_score_correct'].mean(),
        'home_goal_mae':    valid['home_goal_err'].mean(),
        'away_goal_mae':    valid['away_goal_err'].mean(),
    }


def running_report(phase: str, max_matchday: int, actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Load all saved predictions up to max_matchday for a phase and evaluate them.
    Returns a DataFrame with one row per matchday showing cumulative metrics.
    """
    rows = []
    cumulative_eval = []

    for md in range(1, max_matchday + 1):
        path = PRED_DIR / f"{phase}_md{md}.csv"
        if not path.exists():
            continue
        preds = pd.read_csv(path)
        ev = evaluate(preds, actuals)
        cumulative_eval.append(ev)
        cum = pd.concat(cumulative_eval, ignore_index=True)
        m = summary_metrics(cum)
        m['matchday'] = md
        rows.append(m)

    return pd.DataFrame(rows)
