"""CLI script to generate synthetic demonstration datasets and sample CSVs."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.train_model import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-modal EV motor sensor feature dataset."
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=240,
        help="Number of windows to generate per fault class (default: 240).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    df = generate_synthetic_dataset(
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        save_csv=True,
    )
    print(f"[OK] Generated synthetic dataset with {len(df)} total windows across 3 classes.")
    print(df["fault_class"].value_counts().to_string())


if __name__ == "__main__":
    main()

