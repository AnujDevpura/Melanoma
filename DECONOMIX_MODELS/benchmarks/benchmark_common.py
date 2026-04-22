from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_prepared(prepared_dir: Path, split: str = "test") -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    x = np.load(prepared_dir / f"X_{split}.npy")
    y = np.load(prepared_dir / f"y_{split}.npy")
    cell_types = read_lines(prepared_dir / "cell_types.txt")
    genes = read_lines(prepared_dir / "selected_genes.txt")
    return x, y, cell_types, genes


def load_train_test(prepared_dir: Path, split: str = "test") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    x_train = np.load(prepared_dir / "X_train.npy")
    y_train = np.load(prepared_dir / "y_train.npy")
    x_eval, y_eval, cell_types, genes = load_prepared(prepared_dir, split=split)
    return x_train, y_train, x_eval, y_eval, cell_types, genes


def project_to_simplex(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y = np.clip(y, 0.0, None)
    denom = y.sum(axis=1, keepdims=True)
    denom = np.where(denom < eps, eps, denom)
    return y / denom


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, cell_types: List[str]) -> pd.DataFrame:
    y_pred = project_to_simplex(y_pred)
    spearman = []
    mae = []
    for idx in range(y_true.shape[1]):
        try:
            rho, _ = spearmanr(y_true[:, idx], y_pred[:, idx])
            if np.isnan(rho):
                rho = 0.0
        except Exception:
            rho = 0.0
        spearman.append(float(rho))
        mae.append(float(np.mean(np.abs(y_true[:, idx] - y_pred[:, idx]))))

    metrics = pd.DataFrame(
        {
            "cell_type": cell_types,
            "spearman": spearman,
            "mae": mae,
            "avg_prop": y_true.mean(axis=0),
        }
    )
    average = pd.DataFrame(
        {
            "cell_type": ["AVERAGE"],
            "spearman": [float(np.mean(spearman))],
            "mae": [float(np.mean(mae))],
            "avg_prop": [float(y_true.mean())],
        }
    )
    return pd.concat([metrics, average], ignore_index=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
