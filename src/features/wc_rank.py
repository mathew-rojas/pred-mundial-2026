"""FIFA ranking lookup from WC 2026 dataset, used as pre-match features."""
import pandas as pd

_NAME_MAP = {'USA': 'United States'}


def load_wc26_ranks(raw_dir: str = 'data/raw') -> dict[str, int]:
    """
    Returns {team_name: fifa_rank} from the first appearance of each team.
    Lower rank = better (rank 1 is best).
    """
    df = pd.read_csv(f'{raw_dir}/world_cup_2026_matches.csv')

    def _norm(name: str) -> str:
        name = str(name).strip()
        return _NAME_MAP.get(name, name)

    ranks: dict[str, int] = {}
    for _, row in df.iterrows():
        t1 = _norm(row['Team_1'])
        t2 = _norm(row['Team_2'])
        if t1 not in ranks:
            ranks[t1] = int(row['Team_1_FIFA_Rank'])
        if t2 not in ranks:
            ranks[t2] = int(row['Team_2_FIFA_Rank'])

    return ranks
