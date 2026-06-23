# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FIFA World Cup 2026 match predictor. Data source: `martj42/international_results` (49k international matches, 1872–2026). Predicts exact scores and win/draw/loss probabilities using a Dixon-Coles Poisson model and XGBoost classifier, combined as a weighted ensemble.

## Key commands

```bash
# Full pipeline: download data → build features → train both models → save
python main.py

# Re-download raw CSVs (force refresh)
python -c "from src.data.download import download_all; download_all(force=True)"

# Execute a notebook non-interactively
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_live_wc2026.ipynb

# Backtest on WC 2022 (trains on pre-tournament data, evaluates 64 matches)
python -c "
import sys; sys.path.insert(0, '.')
from src.evaluation.backtest import run_backtest, compute_metrics
results, _, _ = run_backtest('data/processed/features.csv', test_year=2022)
for m in ('poisson','xgb','ens'): print(compute_metrics(results, m))
"

# Retrain all models with latest data (run between tournament phases)
python -c "
import sys; sys.path.insert(0, '.')
from src.prediction.update import retrain_full
retrain_full()
"

# Incremental ELO update after a matchday (no full retrain)
# Pass all WC 2026 played matches — update_elo handles ordering and K-factors
python -c "
import sys, pandas as pd; sys.path.insert(0, '.')
from src.prediction.update import update_elo, save_elo, load_elo
df = pd.read_csv('data/raw/results.csv', parse_dates=['date'])
played = df[(df['tournament']=='FIFA World Cup') & (df['date'].dt.year==2026)].dropna(subset=['home_score'])
elo = update_elo(played, load_elo()); save_elo(elo)
"
```

## Architecture

### Data flow
```
data/raw/results.csv  (downloaded from GitHub)
  → src/features/build_features.py::build_features()
      - filters to 1993+ before ELO computation
      - computes ELO via src/features/elo.py::compute_elo()
      - computes rolling form (last 5 matches, vectorized groupby)
  → data/processed/features.csv  (30k rows, 18 cols)
      - trains Poisson on all data, XGBoost on 2006+
  → models/  (poisson_params.json, xgb_pipeline.joblib, elo_ratings.json)
```

### Models

**Poisson** (`src/models/poisson_model.py`): Dixon-Coles model. Fits per-team attack/defence via L-BFGS-B with exponential time-decay weights (`xi=0.002`). Inner log-likelihood is fully vectorised (numpy array indexing, no Python loop). Outputs a `(9×9)` score probability matrix → `predict_exact_score()` returns argmax, `predict_outcome_probs()` returns win/draw/loss from triangle sums.

**XGBoost** (`src/models/ml_model.py`): 3-class classifier (0=away win, 1=draw, 2=home win). Features: `home_elo_before`, `away_elo_before`, `elo_diff`, `home_form`, `away_form`, `form_diff`, `neutral` (7 total, defined in `FEATURE_COLS`). Uses `TimeSeriesSplit` CV. Predict via `predict_probs()`.

**Ensemble** (`src/models/ensemble.py`): weighted average of both models' outcome probabilities. Default weights: Poisson 0.6, XGBoost 0.4.

### Live prediction workflow (WC 2026)

The tournament is in progress. The workflow is:
1. **After each matchday** — update ELO incrementally (`src/prediction/update.py::update_elo`)
2. **Between phases** — full retrain (`retrain_full()`) then regenerate all notebooks
3. **Predict pending matches** — `src/prediction/predict.py::predict_matches()` → score + ensemble probs
4. **Track accuracy** — `src/prediction/tracker.py`: `save_predictions()` before matches, `evaluate()` after

### Monte Carlo simulation (`src/simulation/tournament.py`)

Simulates the remaining WC 2026 group stage + knockout bracket 10k times. Performance-critical path:
- `_MATRIX_CACHE` caches Poisson score matrices per team pair (warm once, reuse across 10k iters)
- `prepare_group_stage()` precomputes base standings from played matches (called once)
- `simulate_group_stage_fast()` uses plain dicts, no pandas in the hot loop (~3ms/iter)
- `simulate_knockout()` resolves drawn knockout matches via ELO-weighted penalty coin flip

Group structure is hardcoded in `GROUPS` dict (12 groups × 4 teams). `NAME_ALIASES` maps WC dataset team names → ELO/Poisson model names (e.g. `"Cape Verde"` → `"Cape Verde Islands"`).

### Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Dataset exploration: results distribution, goals, ELO evolution, WC history |
| `02_backtest_wc2022.ipynb` | Held-out evaluation on WC 2022 (64 matches) |
| `03_simulation_2026.ipynb` | Monte Carlo win probabilities for all 48 teams |
| `04_live_wc2026.ipynb` | **Live workflow**: 1) clasificaciones, 2) backtest completo (J1+J2) con tabla partido a partido, 3) próximos partidos con top-5 scores + P(local/empate/visita). Predicciones guardadas en `data/predictions/group_stage_md0.csv`. |

### ELO design decisions
- Computed from 1993+ (post-USSR/Yugoslavia dissolution — stable team landscape)
- Tournament K-factor multipliers: World Cup ×2, major confederations ×1.3–1.5, friendlies ×0.5
- `compute_elo()` returns `(df_with_elo_cols, final_ratings_dict)` — not stored as function attribute
- `elo_ratings.json` always reflects the latest state including in-progress WC 2026 matches

## Important caveats
- Poisson optimizer (`L-BFGS-B`, `maxiter=1000`) may not fully converge for 319+ teams — the convergence warning is expected and the solution is close enough.
- `home_form=1.5` (league average) is used for upcoming matches that have no form history in the feature set. This is intentional for forward predictions.
- All WC 2026 group-stage matches are treated as `neutral=True`.
