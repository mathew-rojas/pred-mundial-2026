"""Generate match predictions: predicted score + outcome probabilities."""
import pandas as pd
import numpy as np

from pathlib import Path

from src.models.poisson_model import PoissonModel, WC_GOAL_SCALE
from src.models import ml_model
from src.models.ensemble import ensemble_probs
from src.simulation.tournament import NAME_ALIASES as _ALIASES
from src.features.wc_team_profiles import load_wc22_profiles
from src.features.wc_rank import load_wc26_ranks

_PROJECT_RAW = str(Path(__file__).parent.parent.parent / "data" / "raw")


def _elo_name(team: str) -> str:
    return _ALIASES.get(team, team)


def predict_matches(
    matches: list[tuple[str, str]],
    poisson: PoissonModel,
    xgb_pipeline,
    elo_ratings: dict[str, float],
    neutral: bool = True,
    raw_dir: str | None = None,
) -> pd.DataFrame:
    """
    Predict a list of (home, away) matches.

    Returns a DataFrame with one row per match:
      home_team, away_team,
      pred_home, pred_away          (most likely exact score, Poisson)
      p_home_win, p_draw, p_away_win (ensemble probabilities)
      favourite                      (team with highest win probability)
    """
    _raw = raw_dir if raw_dir is not None else _PROJECT_RAW
    _wc22 = load_wc22_profiles(_raw).set_index('team')
    _ranks = load_wc26_ranks(_raw)

    def _wc22_feat(team: str, col: str) -> float:
        try:
            return float(_wc22.loc[team, col])
        except KeyError:
            return float('nan')

    rows = []
    for home, away in matches:
        # Poisson exact score — apply WC calibration scale
        exact = poisson.predict_exact_score(_elo_name(home), _elo_name(away), neutral=neutral, goal_scale=WC_GOAL_SCALE)
        pp = poisson.predict_outcome_probs(_elo_name(home), _elo_name(away), neutral=neutral, goal_scale=WC_GOAL_SCALE)

        # XGBoost (needs ELO + form + enriched features)
        h_elo = elo_ratings.get(_elo_name(home), 1500.0)
        a_elo = elo_ratings.get(_elo_name(away), 1500.0)
        h_rank = float(_ranks.get(home, float('nan')))
        a_rank = float(_ranks.get(away, float('nan')))
        xp = ml_model.predict_probs(
            xgb_pipeline, h_elo, a_elo,
            home_form=1.5, away_form=1.5,  # neutral form for unknown upcoming matches
            neutral=neutral,
            home_wc22_shots=_wc22_feat(home, 'wc22_shots_pg'),
            away_wc22_shots=_wc22_feat(away, 'wc22_shots_pg'),
            home_wc22_sot=_wc22_feat(home, 'wc22_sot_pg'),
            away_wc22_sot=_wc22_feat(away, 'wc22_sot_pg'),
            home_wc22_possession=_wc22_feat(home, 'wc22_possession'),
            away_wc22_possession=_wc22_feat(away, 'wc22_possession'),
            home_fifa_rank=h_rank,
            away_fifa_rank=a_rank,
            rank_diff=h_rank - a_rank,
        )

        ep = ensemble_probs(pp, xp)

        # Favourite
        if ep['home_win'] >= ep['away_win'] and ep['home_win'] >= ep['draw']:
            favourite = home
        elif ep['away_win'] >= ep['home_win'] and ep['away_win'] >= ep['draw']:
            favourite = away
        else:
            favourite = 'Draw'

        rows.append({
            'home_team':   home,
            'away_team':   away,
            'pred_score':  f"{exact[0]}-{exact[1]}",
            'p_home_win':  ep['home_win'],
            'p_draw':      ep['draw'],
            'p_away_win':  ep['away_win'],
            'favourite':   favourite,
            'h_elo':       round(h_elo, 1),
            'a_elo':       round(a_elo, 1),
        })

    df = pd.DataFrame(rows)
    return df


def format_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready copy with formatted probability columns."""
    out = df.copy()
    out['p_home_win'] = out['p_home_win'].map('{:.1%}'.format)
    out['p_draw']     = out['p_draw'].map('{:.1%}'.format)
    out['p_away_win'] = out['p_away_win'].map('{:.1%}'.format)
    return out
