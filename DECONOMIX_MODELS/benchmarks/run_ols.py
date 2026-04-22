from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from benchmark_common import ensure_dir, evaluate_predictions, load_train_test, project_to_simplex


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OLS baseline on prepared benchmark data")
    parser.add_argument("--prepared-dir", default="DECONOMIX_MODELS/Data/prepared")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out-dir", default="DECONOMIX_MODELS/results/benchmarks/ols")
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    x_train, y_train, x_eval, y_eval, cell_types, _ = load_train_test(prepared_dir, split=args.split)

    model = LinearRegression()
    model.fit(x_train, y_train)
    pred = project_to_simplex(model.predict(x_eval))

    pd.DataFrame(pred, columns=cell_types).to_csv(out_dir / f"pred_props_{args.split}.csv", index=False)
    metrics = evaluate_predictions(y_eval, pred, cell_types)
    metrics.to_csv(out_dir / f"performance_{args.split}.csv", index=False)

    average = metrics.loc[metrics["cell_type"] == "AVERAGE"].iloc[0]
    print(f"OLS {args.split}: avg_spearman={average['spearman']:.3f} mae={average['mae']:.4f}")
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
