"""
FIFA World Cup 2026 tournament simulator.
Supports simulating from the current state (mid-tournament).
"""
import numpy as np
import pandas as pd
from itertools import combinations

# ── Group structure ────────────────────────────────────────────────────────────
GROUPS: dict[str, list[str]] = {
    'A': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'B': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'C': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'D': ['Canada', 'Bosnia and Herzegovina', 'Switzerland', 'Qatar'],
    'E': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'F': ['Spain', 'Saudi Arabia', 'Uruguay', 'Cape Verde'],
    'G': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'],
    'H': ['England', 'Ghana', 'Panama', 'Croatia'],
    'I': ['Germany', 'Ivory Coast', 'Ecuador', 'Curaçao'],
    'J': ['Mexico', 'South Korea', 'Czech Republic', 'South Africa'],
    'K': ['France', 'Senegal', 'Iraq', 'Norway'],
    'L': ['Netherlands', 'Sweden', 'Japan', 'Tunisia'],
}

# Team name aliases: name in WC dataset → name in ELO ratings / Poisson model
NAME_ALIASES: dict[str, str] = {
    'Curaçao': 'Curaçao',
    'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'DR Congo': 'DR Congo',
    'Ivory Coast': "Ivory Coast",
    'Cape Verde': 'Cape Verde Islands',
}


def _elo_name(team: str) -> str:
    return NAME_ALIASES.get(team, team)


# ── Standings helpers ──────────────────────────────────────────────────────────
def _match_points(home_score: int, away_score: int) -> tuple[int, int]:
    if home_score > away_score:
        return 3, 0
    if home_score == away_score:
        return 1, 1
    return 0, 3


def compute_standings(played: pd.DataFrame, group_teams: list[str]) -> pd.DataFrame:
    """
    Given completed matches, return standings DataFrame for one group.
    Columns: team, pts, gf, ga, gd, w, d, l
    """
    stats = {t: {'pts': 0, 'gf': 0, 'ga': 0, 'w': 0, 'd': 0, 'l': 0} for t in group_teams}

    for _, row in played.iterrows():
        h, a = row['home_team'], row['away_team']
        if h not in stats or a not in stats:
            continue
        hs, as_ = int(row['home_score']), int(row['away_score'])
        hp, ap = _match_points(hs, as_)
        stats[h]['pts'] += hp;  stats[a]['pts'] += ap
        stats[h]['gf']  += hs;  stats[a]['gf']  += as_
        stats[h]['ga']  += as_; stats[a]['ga']  += hs
        if hp == 3:
            stats[h]['w'] += 1; stats[a]['l'] += 1
        elif hp == 1:
            stats[h]['d'] += 1; stats[a]['d'] += 1
        else:
            stats[a]['w'] += 1; stats[h]['l'] += 1

    df = pd.DataFrame(stats).T.reset_index().rename(columns={'index': 'team'})
    df['gd'] = df['gf'] - df['ga']
    return df.sort_values(['pts', 'gd', 'gf'], ascending=False).reset_index(drop=True)


def _rank_third_place(all_standings: dict[str, pd.DataFrame]) -> list[str]:
    """Return ordered list of all 3rd-place teams, best first."""
    thirds = []
    for g, st in all_standings.items():
        thirds.append(st.iloc[2].to_dict() | {'group': g})
    thirds_df = pd.DataFrame(thirds).sort_values(
        ['pts', 'gd', 'gf'], ascending=False
    )
    return thirds_df['team'].tolist()


# ── Score matrix cache (avoids recomputing identical matchups) ─────────────────
_MATRIX_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _get_matrix(home: str, away: str, poisson_model) -> np.ndarray:
    key = (_elo_name(home), _elo_name(away))
    if key not in _MATRIX_CACHE:
        mat = poisson_model.predict_score_matrix(key[0], key[1], neutral=True)
        flat = mat.flatten()
        _MATRIX_CACHE[key] = flat / flat.sum()
    return _MATRIX_CACHE[key]


def clear_matrix_cache() -> None:
    _MATRIX_CACHE.clear()


# ── Match simulation ───────────────────────────────────────────────────────────
def simulate_match(
    home: str,
    away: str,
    poisson_model,
    elo_ratings: dict[str, float],
    allow_draw: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, str | None]:
    """
    Simulate a single match. Returns (home_goals, away_goals, winner_or_None).
    If allow_draw=False (knockout), resolves ties via penalty ELO lottery.
    """
    if rng is None:
        rng = np.random.default_rng()

    flat = _get_matrix(home, away, poisson_model)
    n = int(np.sqrt(len(flat)))
    idx = rng.choice(len(flat), p=flat)
    h_goals = int(idx // n)
    a_goals = int(idx % n)

    if h_goals > a_goals:
        return h_goals, a_goals, home
    if a_goals > h_goals:
        return h_goals, a_goals, away

    if allow_draw:
        return h_goals, a_goals, None

    # Penalty shootout: winner proportional to ELO
    rh = elo_ratings.get(_elo_name(home), 1500.0)
    ra = elo_ratings.get(_elo_name(away), 1500.0)
    winner = home if rng.random() < rh / (rh + ra) else away
    return h_goals, a_goals, winner


# ── Fast dict-based standings (no pandas in hot path) ─────────────────────────
_Stats = dict  # {pts, gf, ga, w, d, l}


def _empty_stats() -> _Stats:
    return {'pts': 0, 'gf': 0, 'ga': 0, 'w': 0, 'd': 0, 'l': 0}


def _apply_result(
    stats: dict[str, _Stats], home: str, away: str, hs: int, as_: int
) -> None:
    hp, ap = _match_points(hs, as_)
    stats[home]['pts'] += hp;  stats[away]['pts'] += ap
    stats[home]['gf']  += hs;  stats[away]['gf']  += as_
    stats[home]['ga']  += as_; stats[away]['ga']  += hs
    if hp == 3:
        stats[home]['w'] += 1; stats[away]['l'] += 1
    elif hp == 1:
        stats[home]['d'] += 1; stats[away]['d'] += 1
    else:
        stats[away]['w'] += 1; stats[home]['l'] += 1


def _sort_group(stats: dict[str, _Stats]) -> list[str]:
    return sorted(
        stats,
        key=lambda t: (stats[t]['pts'], stats[t]['gf'] - stats[t]['ga'], stats[t]['gf']),
        reverse=True,
    )


# ── Group stage simulation ─────────────────────────────────────────────────────
def prepare_group_stage(
    played: pd.DataFrame,
) -> dict[str, dict[str, _Stats]]:
    """
    Precompute standings from already-played matches.
    Returns base_stats per group — reused across all Monte Carlo iterations.
    """
    base: dict[str, dict[str, _Stats]] = {
        g: {t: _empty_stats() for t in teams}
        for g, teams in GROUPS.items()
    }
    # team → group lookup
    team_group = {t: g for g, teams in GROUPS.items() for t in teams}

    for _, row in played.iterrows():
        h, a = row['home_team'], row['away_team']
        g = team_group.get(h)
        if g is None:
            continue
        _apply_result(base[g], h, a, int(row['home_score']), int(row['away_score']))

    return base


def simulate_group_stage_fast(
    base_stats: dict[str, dict[str, _Stats]],
    pending_list: list[tuple[str, str]],
    poisson_model,
    elo_ratings: dict[str, float],
    rng: np.random.Generator,
) -> tuple[dict[str, list[str]], dict[str, dict[str, _Stats]]]:
    """
    Fast group stage simulation using pre-built base stats (no pandas in loop).
    Returns ({group: [1st,2nd,3rd,4th]}, final_stats_per_group).
    Returning stats avoids a second simulation pass for third-place ranking.
    """
    stats: dict[str, dict[str, _Stats]] = {
        g: {t: dict(v) for t, v in group.items()}
        for g, group in base_stats.items()
    }
    team_group = {t: g for g, teams in GROUPS.items() for t in teams}

    for home, away in pending_list:
        h_goals, a_goals, _ = simulate_match(
            home, away, poisson_model, elo_ratings, allow_draw=True, rng=rng,
        )
        g = team_group[home]
        _apply_result(stats[g], home, away, h_goals, a_goals)

    ranked = {g: _sort_group(stats[g]) for g in GROUPS}
    return ranked, stats


def determine_qualifiers_fast(
    ranked: dict[str, list[str]],
    base_stats: dict[str, dict[str, _Stats]],
    pending_list: list[tuple[str, str]],
    poisson_model,
    elo_ratings: dict[str, float],
    rng: np.random.Generator,
) -> tuple[list[str], list[str], list[str]]:
    """Fast version — ranked already computed by simulate_group_stage_fast."""
    winners    = [ranked[g][0] for g in sorted(ranked)]
    runners_up = [ranked[g][1] for g in sorted(ranked)]
    thirds_info = [
        (ranked[g][2], g) for g in sorted(ranked)
    ]
    # Sort thirds by pts, gd, gf using the stats from the simulation
    # Need access to stats — will be passed via a closure or returned alongside
    # For simplicity, return thirds sorted by group name (caller handles this)
    thirds_teams = [t for t, _ in thirds_info]
    return winners, runners_up, thirds_teams


def determine_qualifiers(
    standings: dict[str, pd.DataFrame],
) -> tuple[list[str], list[str], list[str]]:
    """Returns (winners, runners_up, best_8_third_place). DataFrame version for EDA."""
    winners     = [standings[g].iloc[0]['team'] for g in sorted(standings)]
    runners_up  = [standings[g].iloc[1]['team'] for g in sorted(standings)]
    all_thirds  = _rank_third_place(standings)
    return winners, runners_up, all_thirds[:8]


# ── Knockout bracket ───────────────────────────────────────────────────────────
def _build_bracket(
    winners: list[str],
    runners_up: list[str],
    thirds: list[str],
) -> list[tuple[str, str]]:
    """
    Build Round-of-32 bracket.
    Seeds 1-12: group winners (A→L), 13-24: runners-up, 25-32: best 3rd-place.
    Standard seed pairing: 1v32, 2v31, ..., 16v17.
    """
    seeds = winners + runners_up + thirds  # length 32
    pairs = [(seeds[i], seeds[31 - i]) for i in range(16)]
    return pairs


def simulate_knockout(
    r32_pairs: list[tuple[str, str]],
    poisson_model,
    elo_ratings: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """
    Simulate full knockout bracket from Round of 32.
    Round keys: r32 (16 winners) → r16 (8) → qf (4) → sf (2 finalists) → winner (1).
    """
    results: dict[str, list[str]] = {
        'r32': [], 'r16': [], 'qf': [], 'sf': [], 'winner': [],
    }

    current_round = list(r32_pairs)

    for round_name in ('r32', 'r16', 'qf', 'sf'):
        next_round_teams = []
        for home, away in current_round:
            _, _, winner = simulate_match(
                home, away, poisson_model, elo_ratings,
                allow_draw=False, rng=rng,
            )
            results[round_name].append(winner)
            next_round_teams.append(winner)

        current_round = [
            (next_round_teams[i], next_round_teams[i + 1])
            for i in range(0, len(next_round_teams), 2)
        ]

    # current_round now has exactly 1 pair — the two finalists
    h, a = current_round[0]
    _, _, champion = simulate_match(
        h, a, poisson_model, elo_ratings, allow_draw=False, rng=rng,
    )
    results['winner'].append(champion)

    return results


# ── Full Monte Carlo simulation ────────────────────────────────────────────────
def run_monte_carlo(
    results_path: str,
    poisson_model,
    elo_ratings: dict[str, float],
    n_simulations: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run n_simulations of the remaining 2026 World Cup.
    Returns DataFrame with one row per team and columns for each stage probability.
    """
    df = pd.read_csv(results_path, parse_dates=['date'])
    wc26 = df[(df['tournament'] == 'FIFA World Cup') & (df['date'].dt.year == 2026)].copy()

    played  = wc26.dropna(subset=['home_score', 'away_score'])
    pending = wc26[wc26['home_score'].isna()].copy()

    print(f"  Played:  {len(played)} matches")
    print(f"  Pending: {len(pending)} matches")

    # Precompute once: base standings from played matches
    base_stats = prepare_group_stage(played)

    # Pending matches as list of (home, away) — order preserved
    pending_list = list(zip(pending['home_team'], pending['away_team']))

    # Warm up the matrix cache for all pending pairs
    print("  Warming matrix cache...")
    for home, away in pending_list:
        _get_matrix(home, away, poisson_model)
    print(f"  Cache size: {len(_MATRIX_CACHE)} matrices")

    all_teams = [t for teams in GROUPS.values() for t in teams]
    team_group = {t: g for g, teams in GROUPS.items() for t in teams}

    counts = {
        stage: {t: 0 for t in all_teams}
        for stage in ('qualified', 'r16', 'qf', 'sf', 'final', 'winner')
    }

    rng = np.random.default_rng(seed)

    for i in range(n_simulations):
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_simulations}...")

        # Single-pass group stage simulation — returns ranked order AND final stats
        ranked, final_stats = simulate_group_stage_fast(
            base_stats, pending_list, poisson_model, elo_ratings, rng
        )

        winners    = [ranked[g][0] for g in sorted(ranked)]
        runners_up = [ranked[g][1] for g in sorted(ranked)]

        # Best 8 third-place ranked by pts/gd/gf using the same simulation's stats
        thirds_ranked = [ranked[g][2] for g in sorted(ranked)]
        thirds = sorted(
            thirds_ranked,
            key=lambda t: (
                final_stats[team_group[t]][t]['pts'],
                final_stats[team_group[t]][t]['gf'] - final_stats[team_group[t]][t]['ga'],
                final_stats[team_group[t]][t]['gf'],
            ),
            reverse=True,
        )[:8]

        qualified = winners + runners_up + thirds
        for t in qualified:
            counts['qualified'][t] += 1

        r32_pairs = _build_bracket(winners, runners_up, thirds)
        ko = simulate_knockout(r32_pairs, poisson_model, elo_ratings, rng)

        for t in ko['r32']:    counts['r16'][t]    += 1
        for t in ko['r16']:    counts['qf'][t]     += 1
        for t in ko['qf']:     counts['sf'][t]     += 1
        for t in ko['sf']:     counts['final'][t]  += 1
        for t in ko['winner']: counts['winner'][t] += 1

    rows = []
    for team in all_teams:
        rows.append({
            'team':      team,
            'group':     team_group[team],
            'p_qualify': counts['qualified'][team] / n_simulations,
            'p_r16':     counts['r16'][team]       / n_simulations,
            'p_qf':      counts['qf'][team]        / n_simulations,
            'p_sf':      counts['sf'][team]        / n_simulations,
            'p_final':   counts['final'][team]     / n_simulations,
            'p_winner':  counts['winner'][team]    / n_simulations,
        })

    return pd.DataFrame(rows).sort_values('p_winner', ascending=False).reset_index(drop=True)
