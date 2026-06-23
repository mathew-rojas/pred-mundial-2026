import sys, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd
from src.models.poisson_model import PoissonModel
from src.simulation.tournament import run_monte_carlo

with open("models/poisson_params.json") as f:
    params = json.load(f)
with open("models/elo_ratings.json") as f:
    elo = json.load(f)

poisson = PoissonModel()
poisson.params_ = params
poisson.teams_ = list(params["attack"].keys())

mc = run_monte_carlo(
    "data/raw/results.csv",
    poisson_model=poisson,
    elo_ratings=elo,
    n_simulations=10000,
    seed=42,
)

print("\nTOP 10 — PROBABILIDAD DE GANAR EL MUNDIAL 2026")
print(f"{'Equipo':<22} {'Campeon':>8} {'Final':>7} {'Semis':>7} {'Cuartos':>8}")
print("-" * 56)
for _, row in mc.head(10).iterrows():
    print(
        f"{row['team']:<22} {row['p_winner']:>7.1%} {row['p_final']:>7.1%}"
        f" {row['p_sf']:>7.1%} {row['p_qf']:>8.1%}"
    )
