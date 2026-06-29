# pred-mundial-2026

FIFA World Cup 2026 match predictor. Predicts exact scores and win/draw/loss probabilities using a Dixon-Coles Poisson model and XGBoost classifier combined as a weighted ensemble.

Data sources:
- [martj42/international_results](https://github.com/martj42/international_results) — 49k international matches, 1872–2026
- Kaggle WC 2022 complete dataset — per-match stats (shots, SOT, possession) for all 64 WC 2022 matches
- Kaggle WC 2026 complete dataset — FIFA rankings and match-level stats

## Results

Backtest on WC 2026 (live, post-hoc evaluation with current models):

| | N | Result accuracy | Exact score |
|---|---|---|---|
| Matchday 1 (Jun 11–17) | 24 | 58.3% | 29.2% |
| Matchday 2 (Jun 18–22) | 20 | 80.0% | 15.0% |
| Matchday 3 (Jun 23–27) | 4 | 75.0% | 0.0% |
| **Total** | **48** | **68.8%** | **20.8%** |

Backtest on WC 2022 (true held-out test — models trained before tournament):

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Poisson | 46.9% | 1.121 | 0.656 |
| XGBoost | **50.0%** | 1.162 | — |
| **Ensemble** | 46.9% | **1.107** | — |

## Setup

```bash
pip install -r requirements.txt
python main.py   # checks GitHub for new data → build features → train → save
```

`main.py` auto-detects new data on GitHub (via HTTP HEAD size check) and skips the retrain if nothing changed and saved models already exist. Force a full refresh with:

```bash
python -c "from src.data.download import download_all; download_all(force=True)"
python main.py
```

Place the following Kaggle CSVs in `data/raw/` before running:
- `world_cup_2022_stats.csv`
- `world_cup_2026_matches.csv`

`main.py` downloads the raw CSVs, builds the feature matrix (16 features), trains both models, and saves them to `models/`.

## Usage

### Predict a match

```python
import json, joblib
from src.models.poisson_model import PoissonModel
from src.prediction.predict import predict_matches, format_predictions
from src.prediction.update import load_elo

with open("models/poisson_params.json") as f: params = json.load(f)
poisson = PoissonModel(); poisson.params_ = params; poisson.teams_ = list(params["attack"].keys())
xgb = joblib.load("models/xgb_pipeline.joblib")
elo = load_elo()

pred = predict_matches([("France", "Argentina")], poisson, xgb, elo, neutral=True)
print(format_predictions(pred).to_string(index=False))
```

### Update ELO after a matchday (no full retrain)

```python
import pandas as pd
from src.prediction.update import update_elo, save_elo, load_elo

df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
played = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)].dropna(subset=["home_score"])
save_elo(update_elo(played, load_elo()))
```

### Retrain between phases

```python
from src.prediction.update import retrain_full
retrain_full()   # ~2 min, saves all models
```

Then re-execute notebook `04_live_wc2026.ipynb` — it auto-detects the current tournament phase (group stage → octavos → ronda 16 → cuartos → semis → final) and predicts the next round automatically, walking the seeded bracket from actual results in `results.csv`.

## Notebooks

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Dataset exploration: goals, ELO evolution, WC history, feature correlations |
| `02_backtest_wc2022.ipynb` | Held-out evaluation on WC 2022 (64 matches) |
| `03_simulation_2026.ipynb` | Monte Carlo simulation — win probabilities for all 48 teams |
| `04_live_wc2026.ipynb` | Live workflow (auto-adapts phase to phase): standings/bracket · full backtest · next-phase predictions through the Final |

## Architecture

```
src/
├── data/download.py              # Downloads CSVs from GitHub
├── features/
│   ├── elo.py                    # ELO ratings (K-factor weighted by tournament)
│   ├── build_features.py         # Feature matrix: ELO + rolling form + WC enrichment
│   ├── wc_team_profiles.py       # WC 2022 team attack/defense profiles (shots, SOT, possession)
│   └── wc_rank.py                # FIFA rank lookup from WC 2026 dataset
├── models/
│   ├── poisson_model.py          # Dixon-Coles model — vectorized log-likelihood + WC calibration
│   ├── ml_model.py               # XGBoost 3-class classifier (16 features)
│   └── ensemble.py               # Weighted combination (Poisson 60%, XGBoost 40%)
├── evaluation/backtest.py        # Held-out tournament evaluation
├── prediction/
│   ├── predict.py                # predict_matches() — score + ensemble probs
│   ├── update.py                 # Incremental ELO update + full retrain
│   └── tracker.py                # Save predictions, evaluate vs actuals
└── simulation/tournament.py      # Monte Carlo simulation (~34s for 10k iterations)
```

**ELO:** computed from 1993+ (stable post-USSR/Yugoslavia team landscape). Tournament K-factor multipliers: World Cup ×2, major confederations ×1.3–1.5, friendlies ×0.5.

**Poisson:** fits attack/defence strength per team via L-BFGS-B with exponential time-decay weights (`xi=0.003`). World Cup matches receive an additional `wc_weight=2.0` multiplier so WC 2022/2026 results outweigh regular internationals from the same period. Outputs a 9×9 joint score probability matrix. `WC_GOAL_SCALE = (1.36, 1.10)` is applied to all WC predictions to correct for the model's systematic underestimation of WC-level goal rates. Exact scores use `round(λ)` rather than argmax to avoid mode bias.

**XGBoost (16 features):** trained with combined sample weights — exponential time-decay (`exp(-0.001 × days_ago)`) multiplied by a 2× WC tournament bonus — so recent World Cup matches have the highest influence on the fitted classifier.

| Feature | Source |
|---|---|
| `home/away_elo_before`, `elo_diff` | ELO ratings |
| `home/away_form`, `form_diff` | Rolling avg points, last 5 matches |
| `neutral` | Venue flag |
| `home/away_wc22_shots`, `home/away_wc22_sot`, `home/away_wc22_possession` | WC 2022 team profiles |
| `home/away_fifa_rank`, `rank_diff` | FIFA ranking (WC 2026 dataset) |

New features are sparse (WC 2022 profiles cover 32 teams, FIFA rank covers WC 2026 teams). XGBoost handles NaN natively — only the 7 core ELO/form features are required for training rows.

**Monte Carlo:** simulates the remaining group stage + full knockout bracket 10,000 times in ~34s using cached score matrices and dict-based standings (no pandas in the hot loop).
