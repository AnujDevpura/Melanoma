from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from tqdm.auto import tqdm

from benchmark_common import ensure_dir, evaluate_predictions, load_train_test, project_to_simplex


def learn_signature(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    signature, *_ = np.linalg.lstsq(y_train, x_train, rcond=None)
    signature = np.clip(signature, 0.0, None)
    denom = np.maximum(signature.sum(axis=1, keepdims=True), 1e-12)
    return signature / denom


def predict_nnls(signature: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    signature_t = signature.T
    pred = np.zeros((x_eval.shape[0], signature.shape[0]), dtype=np.float64)
    for i in tqdm(range(x_eval.shape[0]), desc="NNLS", leave=False):
        weights, _ = nnls(signature_t, x_eval[i].astype(np.float64, copy=False))
        pred[i] = weights
    return project_to_simplex(pred)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NNLS baseline on prepared benchmark data")
    parser.add_argument("--prepared-dir", default="DECONOMIX_MODELS/Data/prepared")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out-dir", default="DECONOMIX_MODELS/results/benchmarks/nnls")
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    x_train, y_train, x_eval, y_eval, cell_types, genes = load_train_test(prepared_dir, split=args.split)

    signature = learn_signature(x_train, y_train)
    pred = predict_nnls(signature, x_eval)

    pd.DataFrame(signature, index=cell_types, columns=genes).to_csv(out_dir / "signature.csv")
    pd.DataFrame(pred, columns=cell_types).to_csv(out_dir / f"pred_props_{args.split}.csv", index=False)
    metrics = evaluate_predictions(y_eval, pred, cell_types)
    metrics.to_csv(out_dir / f"performance_{args.split}.csv", index=False)

    average = metrics.loc[metrics["cell_type"] == "AVERAGE"].iloc[0]
    print(f"NNLS {args.split}: avg_spearman={average['spearman']:.3f} mae={average['mae']:.4f}")
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
