"""
Dixon-Coles Poisson model for match score prediction.
Vectorized log-likelihood for fast optimization.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

# WC matches score ~36% more goals (home) / ~10% more (away) than the model
# predicts from general international data. Apply when predicting WC matches.
WC_GOAL_SCALE: tuple[float, float] = (1.36, 1.10)  # (home_scale, away_scale)


class PoissonModel:
    def __init__(self, xi: float = 0.003, wc_weight: float = 2.0):
        self.xi = xi
        self.wc_weight = wc_weight
        self.params_: dict = {}
        self.teams_: list[str] = []

    def _time_weight(self, dates: pd.Series) -> np.ndarray:
        days_ago = (dates.max() - dates).dt.days.values
        return np.exp(-self.xi * days_ago)

    def fit(self, df: pd.DataFrame) -> "PoissonModel":
        df = df.copy()
        self.teams_ = sorted(set(df["home_team"]) | set(df["away_team"]))
        n = len(self.teams_)
        idx = {t: i for i, t in enumerate(self.teams_)}

        time_w = self._time_weight(df["date"])
        wc_mask = df["tournament"].str.contains("FIFA World Cup", na=False).values
        weights = time_w * np.where(wc_mask, self.wc_weight, 1.0)

        # Pre-compute arrays once — used inside optimizer on every call
        home_idx = np.array([idx[t] for t in df["home_team"]])
        away_idx = np.array([idx[t] for t in df["away_team"]])
        home_goals = df["home_score"].values.astype(int)
        away_goals = df["away_score"].values.astype(int)
        neutral_arr = df["neutral"].astype(int).values
        log_fact_h = gammaln(home_goals + 1)
        log_fact_a = gammaln(away_goals + 1)

        # Masks for Dixon-Coles tau correction (low-score adjustment)
        m00 = (home_goals == 0) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m10 = (home_goals == 1) & (away_goals == 0)
        m11 = (home_goals == 1) & (away_goals == 1)

        def neg_log_likelihood(params: np.ndarray) -> float:
            attack = params[:n]
            defence = params[n : 2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]

            lam = np.exp(
                attack[home_idx] - defence[away_idx] + home_adv * (1 - neutral_arr)
            )
            mu = np.exp(attack[away_idx] - defence[home_idx])

            ll = weights * (
                home_goals * np.log(lam) - lam - log_fact_h
                + away_goals * np.log(mu) - mu - log_fact_a
            )

            tau = np.ones(len(df))
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho

            ll += weights * np.log(np.maximum(tau, 1e-10))
            return -ll.sum()

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.3
        bounds = [(-3, 3)] * (2 * n) + [(0, 1)] + [(-1, 1)]

        res = minimize(
            neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-6},
        )
        if not res.success:
            print(f"  Warning: Poisson optimizer did not fully converge: {res.message}")

        params = res.x
        self.params_ = {
            "attack": dict(zip(self.teams_, params[:n])),
            "defence": dict(zip(self.teams_, params[n : 2 * n])),
            "home_adv": float(params[2 * n]),
            "rho": float(params[2 * n + 1]),
        }
        return self

    def _lam_mu(
        self, home: str, away: str, neutral: bool, goal_scale: tuple[float, float] | None
    ) -> tuple[float, float]:
        att = self.params_["attack"]
        dfe = self.params_["defence"]
        home_adv = 0.0 if neutral else self.params_["home_adv"]
        lam = np.exp(att.get(home, 0.0) - dfe.get(away, 0.0) + home_adv)
        mu = np.exp(att.get(away, 0.0) - dfe.get(home, 0.0))
        if goal_scale is not None:
            lam *= goal_scale[0]
            mu *= goal_scale[1]
        return lam, mu

    def predict_score_matrix(
        self,
        home: str,
        away: str,
        neutral: bool = False,
        max_goals: int = 8,
        goal_scale: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Joint probability matrix P(home_goals=i, away_goals=j)."""
        lam, mu = self._lam_mu(home, away, neutral, goal_scale)

        goals = np.arange(max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))

        rho = self.params_["rho"]
        for (i, j), tau_fn in [
            ((0, 0), lambda: 1 - lam * mu * rho),
            ((0, 1), lambda: 1 + lam * rho),
            ((1, 0), lambda: 1 + mu * rho),
            ((1, 1), lambda: 1 - rho),
        ]:
            matrix[i, j] *= tau_fn()

        return matrix / matrix.sum()

    def predict_exact_score(
        self,
        home: str,
        away: str,
        neutral: bool = False,
        goal_scale: tuple[float, float] | None = None,
    ) -> tuple[int, int]:
        """Most likely exact score via rounded expected goals (home_goals, away_goals)."""
        lam, mu = self._lam_mu(home, away, neutral, goal_scale)
        return int(round(lam)), int(round(mu))

    def predict_outcome_probs(
        self,
        home: str,
        away: str,
        neutral: bool = False,
        goal_scale: tuple[float, float] | None = None,
    ) -> dict[str, float]:
        m = self.predict_score_matrix(home, away, neutral, goal_scale=goal_scale)
        return {
            "home_win": float(np.tril(m, -1).sum()),
            "draw": float(np.trace(m)),
            "away_win": float(np.triu(m, 1).sum()),
        }
