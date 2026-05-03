"""
Generate and save synthetic 3D printer fleet dataset.

Usage
-----
    python scripts/generate_data.py
    python scripts/generate_data.py --n-jobs 50 --n-layers 500 --output data/fleet.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import PrintConfig, PrintFleetGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic FDM printer fleet data.")
    p.add_argument("--n-jobs",   type=int, default=30)
    p.add_argument("--n-layers", type=int, default=400)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output",   type=str, default="data/print_fleet.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PrintConfig(n_layers=args.n_layers)
    df = PrintFleetGenerator(n_jobs=args.n_jobs, config=cfg, random_seed=args.seed).generate()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logger.info("Saved %d records (%d jobs) to %s", len(df), args.n_jobs, out_path)
    logger.info("Failure mode distribution:\n%s", df["failure_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
