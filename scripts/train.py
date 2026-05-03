"""
Train the print failure classifier and save the model artifact.

Usage
-----
    python scripts/train.py
    python scripts/train.py --data data/print_fleet.csv --output models/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import PrintConfig, PrintFleetGenerator
from monitor import PrintMonitor

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and save the 3D print failure monitor.")
    p.add_argument("--data",     type=str, default=None)
    p.add_argument("--n-jobs",   type=int, default=30)
    p.add_argument("--output",   type=str, default="models/")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.data:
        logger.info("Loading data from %s", args.data)
        df = pd.read_csv(args.data)
    else:
        logger.info("Generating fleet data for %d print jobs...", args.n_jobs)
        df = PrintFleetGenerator(n_jobs=args.n_jobs).generate()

    monitor = PrintMonitor()
    results = monitor.fit(df)

    print("\n=== Failure Mode Classification Report ===")
    print(results.classification_report)
    print(f"Test accuracy: {results.test_accuracy:.4f}")
    print("\nTop feature importances:")
    print(results.feature_importances.tail(8).to_string())

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(monitor, out_dir / "print_monitor.pkl")
    results.feature_importances.to_csv(out_dir / "feature_importances.csv")
    logger.info("Model saved to %s", out_dir)


if __name__ == "__main__":
    main()
