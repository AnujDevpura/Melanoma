from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmark_common import ensure_dir, load_prepared


def main() -> int:
    parser = argparse.ArgumentParser(description="Export prepared benchmark split to CSV for R baselines")
    parser.add_argument("--prepared-dir", default="DECONOMIX_MODELS/Data/prepared")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="DECONOMIX_MODELS/results/benchmark_inputs")
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    x, y, cell_types, genes = load_prepared(prepared_dir, split=args.split)

    pd.DataFrame(x, columns=genes).to_csv(out_dir / f"bulk_{args.split}_expr.csv", index=False)
    pd.DataFrame(y, columns=cell_types).to_csv(out_dir / f"truth_{args.split}_props.csv", index=False)
    (out_dir / "cell_types.txt").write_text("\n".join(cell_types) + "\n", encoding="utf-8")
    (out_dir / "selected_genes.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")

    print(f"Saved: {out_dir / f'bulk_{args.split}_expr.csv'}")
    print(f"Saved: {out_dir / f'truth_{args.split}_props.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
