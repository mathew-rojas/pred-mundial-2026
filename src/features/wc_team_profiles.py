"""Per-team WC 2022 performance profiles used as pre-match features for WC 2026."""
import pandas as pd

_NAME_MAP = {'Korea Republic': 'South Korea'}


def load_wc22_profiles(raw_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Returns a DataFrame indexed by team with columns:
      wc22_shots_pg, wc22_sot_pg, wc22_possession
    Only 32 teams have data; merge with left-join to get NaN for the rest.
    """
    df = pd.read_csv(f'{raw_dir}/world_cup_2022_stats.csv')

    def _parse_pct(val) -> float:
        return float(str(val).replace('%', '').strip())

    def _norm(name: str) -> str:
        name = name.strip().title()
        return _NAME_MAP.get(name, name)

    records = []
    for _, row in df.iterrows():
        for team_col, shots_col, sot_col, poss_col in [
            ('team1', 'total attempts team1', 'on target attempts team1', 'possession team1'),
            ('team2', 'total attempts team2', 'on target attempts team2', 'possession team2'),
        ]:
            records.append({
                'team':       _norm(row[team_col]),
                'shots':      float(row[shots_col]),
                'sot':        float(row[sot_col]),
                'possession': _parse_pct(row[poss_col]),
            })

    long = pd.DataFrame(records)
    return (
        long.groupby('team')
        .agg(
            wc22_shots_pg=('shots', 'mean'),
            wc22_sot_pg=('sot', 'mean'),
            wc22_possession=('possession', 'mean'),
        )
        .reset_index()
    )
