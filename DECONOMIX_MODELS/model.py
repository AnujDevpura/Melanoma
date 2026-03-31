import pandas as pd

props = pd.read_csv("TCGA_SKCM_cell_proportions.csv", index_col=0)

print(props.head())
print("\nDo proportions sum to ~1?")
print(props.sum(axis=1).head())
