"""Generate match predictions: predicted score + outcome probabilities."""
import pandas as pd
import numpy as np

from src.models.poisson_model import PoissonModel
from src.models import ml_model
from src.models.ensemble import ensemble_probs
from src.simulation.tournament import NAME_ALIASES as _ALIASES


def _elo_name(team: str) -> str:
    return _ALIASES.get(team, team)


def predict_matches(
    matches: list[tuple[str, str]],
    poisson: PoissonModel,
    xgb_pipeline,
    elo_ratings: dict[str, float],
    neutral: bool = True,
) -> pd.DataFrame:
    """
    Predict a list of (home, away) matches.

    Returns a DataFrame with one row per match:
      home_team, away_team,
      pred_home, pred_away          (most likely exact score, Poisson)
      p_home_win, p_draw, p_away_win (ensemble probabilities)
      favourite                      (team with highest win probability)
    """
    rows = []
    for home, away in matches:
        # Poisson exact score
        exact = poisson.predict_exact_score(_elo_name(home), _elo_name(away), neutral=neutral)
        pp = poisson.predict_outcome_probs(_elo_name(home), _elo_name(away), neutral=neutral)

        # XGBoost (needs ELO + form features)
        h_elo = elo_ratings.get(_elo_name(home), 1500.0)
        a_elo = elo_ratings.get(_elo_name(away), 1500.0)
        xp = ml_model.predict_probs(
            xgb_pipeline, h_elo, a_elo,
            home_form=1.5, away_form=1.5,  # neutral form for unknown upcoming matches
            neutral=neutral,
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
