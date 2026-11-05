"""Dataset preprocessing, validation, and scaling for EV Motor Fault Detection."""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, FAULT_CLASSES


def validate_and_clean_dataset(
    df: pd.DataFrame,
    require_target: bool = True,
    target_col: str = "fault_class",
) -> pd.DataFrame:
    """Validate feature columns, handle missing/infinite values, and filter valid fault classes."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input dataset must be a non-empty pandas DataFrame.")

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required feature columns: {missing_cols}")

    cleaned = df.copy()
    # Replace Inf with NaN, then impute with column median (or 0.0 if all NaN)
    for col in FEATURE_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned[col] = cleaned[col].replace([np.inf, -np.inf], np.nan)
        median_val = cleaned[col].median()
        fill_val = float(median_val) if pd.notna(median_val) else 0.0
        cleaned[col] = cleaned[col].fillna(fill_val)

    if require_target:
        if target_col not in cleaned.columns:
            raise ValueError(f"Target column '{target_col}' is missing from dataset.")
        cleaned[target_col] = cleaned[target_col].astype(str).str.strip().str.upper()
        valid_mask = cleaned[target_col].isin(FAULT_CLASSES)
        if not valid_mask.all():
            invalid_vals = cleaned.loc[~valid_mask, target_col].unique().tolist()
            raise ValueError(f"Invalid target class labels found: {invalid_vals}. Expected {FAULT_CLASSES}")

    return cleaned


def prepare_features_and_labels(
    df: pd.DataFrame,
    target_col: str = "fault_class",
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Extract ordered feature matrix X, string labels y, and fit/apply StandardScaler."""
    cleaned = validate_and_clean_dataset(df, require_target=True, target_col=target_col)
    X_raw = cleaned[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = cleaned[target_col].to_numpy(dtype=str)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    elif fit_scaler:
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw)

    return X_scaled, y, scaler


def load_csv_dataset(csv_path: Path, require_target: bool = True) -> pd.DataFrame:
    """Load and validate a CSV dataset from disk (either recorded hardware CSV or synthetic CSV)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    return validate_and_clean_dataset(df, require_target=require_target)

