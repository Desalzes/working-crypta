import os
from pathlib import Path
import pandas as pd

data_dir = Path("data/historical")

def combine_data_by_timeframe():
    for pair in [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]:
        pair_dir = data_dir / pair

        # Combine 1-day files
        ada_1d_files = sorted([f for f in pair_dir.glob(f"{pair}-1d-*.csv")])
        ada_1d_df = pd.concat([pd.read_csv(f) for f in ada_1d_files])
        ada_1d_df.to_csv(pair_dir / f"{pair}-1d.csv", index=False)

        # Combine 1-minute files
        ada_1m_files = sorted([f for f in pair_dir.glob(f"{pair}-1m-*.csv")])
        ada_1m_df = pd.concat([pd.read_csv(f) for f in ada_1m_files])
        ada_1m_df.to_csv(pair_dir / f"{pair}-1m.csv", index=False)

        # Combine 4-hour files
        ada_4h_files = sorted([f for f in pair_dir.glob(f"{pair}-4h-*.csv")])
        ada_4h_df = pd.concat([pd.read_csv(f) for f in ada_4h_files])
        ada_4h_df.to_csv(pair_dir / f"{pair}-4h.csv", index=False)

        # Combine 5-minute files
        ada_5m_files = sorted([f for f in pair_dir.glob(f"{pair}-5m-*.csv")])
        ada_5m_df = pd.concat([pd.read_csv(f) for f in ada_5m_files])
        ada_5m_df.to_csv(pair_dir / f"{pair}-5m.csv", index=False)

if __name__ == "__main__":
    combine_data_by_timeframe()