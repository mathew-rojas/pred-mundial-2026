# pred-mundial-2026

FIFA World Cup 2026 match predictor. Predicts exact scores and win/draw/loss probabilities using a Dixon-Coles Poisson model and XGBoost classifier combined as a weighted ensemble.

Data source: [martj42/international_results](https://github.com/martj42/international_results) — 49k international matches, 1872–2026.

## Results

Backtest on WC 2026 (live, post-hoc evaluation with current models):

| | N | Result accuracy | Exact score |
|---|---|---|---|
| Matchday 1 | 24 | 50.0% | 0.0% |
| Matchday 2 | 20 | 75.0% | 15.0% |
| **Total** | **44** | **61.4%** | **6.8%** |

Backtest on WC 2022 (true held-out test — models trained before tournament):

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Poisson | 46.9% | 1.121 | 0.656 |
| XGBoost | 43.8% | 1.085 | 0.642 |
| **Ensemble** | 45.3% | **1.081** | **0.637** |

## Setup

```bash
pip install -r requirements.txt
python main.py   # download data → build features → train models → save
```

`main.py` downloads the raw CSVs, builds the feature matrix, trains both models, and saves them to `models/`.

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

## Notebooks

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Dataset exploration: goals, ELO evolution, WC history |
| `02_backtest_wc2022.ipynb` | Held-out evaluation on WC 2022 (64 matches) |
| `03_simulation_2026.ipynb` | Monte Carlo simulation — win probabilities for all 48 teams |
| `04_live_wc2026.ipynb` | Live workflow: standings · backtest · upcoming match predictions |

## Architecture

```
src/
├── data/download.py          # Downloads CSVs from GitHub
├── features/
│   ├── elo.py                # ELO ratings (K-factor weighted by tournament)
│   └── build_features.py     # Feature matrix: ELO + rolling form (vectorized)
├── models/
│   ├── poisson_model.py      # Dixon-Coles model — vectorized log-likelihood
│   ├── ml_model.py           # XGBoost 3-class classifier
│   └── ensemble.py           # Weighted combination (Poisson 60%, XGBoost 40%)
├── evaluation/backtest.py    # Held-out tournament evaluation
├── prediction/
│   ├── predict.py            # predict_matches() — score + ensemble probs
│   ├── update.py             # Incremental ELO update + full retrain
│   └── tracker.py            # Save predictions, evaluate vs actuals
└── simulation/tournament.py  # Monte Carlo simulation (~34s for 10k iterations)
```

**ELO:** computed from 1993+ (stable post-USSR/Yugoslavia team landscape). Tournament K-factor multipliers: World Cup ×2, major confederations ×1.3–1.5, friendlies ×0.5.

**Poisson:** fits attack/defence strength per team via L-BFGS-B with exponential time-decay weights. Outputs a 9×9 joint score probability matrix.

**Monte Carlo:** simulates the remaining group stage + full knockout bracket 10,000 times in ~34s using cached score matrices and dict-based standings (no pandas in the hot loop).
