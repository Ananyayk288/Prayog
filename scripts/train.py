"""CLI script to train and evaluate the Random Forest & Isolation Forest models."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.preprocessing import load_csv_dataset
from models.train_model import train_and_evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EV Motor Fault Detection models from CSV or synthetic generator."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to a labeled CSV dataset (if omitted, uses/generates synthetic dataset).",
    )
    parser.add_argument(
        "--source-label",
        type=str,
        default="SYNTHETIC_DEMONSTRATION",
        help="Label describing dataset origin (e.g. SYNTHETIC_DEMONSTRATION or RECORDED_HARDWARE).",
    )
    args = parser.parse_args()

    if args.csv:
        df = load_csv_dataset(Path(args.csv), require_target=True)
        source_label = args.source_label if args.source_label != "SYNTHETIC_DEMONSTRATION" else "CSV_DATASET"
    else:
        df = None
        source_label = args.source_label

    metrics = train_and_evaluate_models(df=df, data_source_label=source_label)
    print("=" * 72)
    print("EV MOTOR FAULT DETECTION — MODEL TRAINING SUMMARY")
    print("=" * 72)
    print(f"Data Source        : {metrics['data_source']}")
    print(f"Disclaimer         : {metrics['validation_disclaimer']}")
    print(f"Total Windows      : {metrics['total_samples']} (Train: {metrics['train_samples']}, Test: {metrics['test_samples']})")
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Precision (Weight) : {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weight)    : {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (Weight)  : {metrics['f1_weighted']:.4f}")
    print("-" * 72)
    print("Classification Report:\n")
    print(metrics["classification_report_text"])


if __name__ == "__main__":
    main()

