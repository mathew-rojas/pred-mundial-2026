"""Weighted ensemble of Poisson and XGBoost outcome probabilities."""


def ensemble_probs(
    poisson_probs: dict[str, float],
    xgb_probs: dict[str, float],
    w_poisson: float = 0.6,
) -> dict[str, float]:
    """
    Combine Poisson and XGBoost outcome probabilities.
    Keys: 'home_win', 'draw', 'away_win'.
    """
    w_xgb = 1 - w_poisson
    return {
        k: w_poisson * poisson_probs[k] + w_xgb * xgb_probs[k]
        for k in ("home_win", "draw", "away_win")
    }
