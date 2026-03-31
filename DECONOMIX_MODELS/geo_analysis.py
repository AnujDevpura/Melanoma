import pandas as pd
import scanpy as sc

# Load the cell type annotations
cell_types = pd.read_csv('Data/GSM4455935.txt', sep='\t')

print("=" * 60)
print("CELL TYPE FILE ANALYSIS")
print("=" * 60)

# Basic info
print(f"\nTotal cells: {len(cell_types)}")
print(f"\nColumns: {cell_types.columns.tolist()}")

# Cell type distribution
print("\n" + "=" * 60)
print("CELL TYPE DISTRIBUTION")
print("=" * 60)
print(cell_types['cell_type'].value_counts())

# Check for unique samples
if 'sample' in cell_types.columns:
    print(f"\nSamples included: {cell_types['sample'].unique()}")

# Preview data
print("\n" + "=" * 60)
print("FIRST FEW ROWS")
print("=" * 60)
print(cell_types.head(10))

# Load the .h5 file
print("\n" + "=" * 60)
print("LOADING EXPRESSION DATA")
print("=" * 60)
adata = sc.read_10x_h5('GSM4455935.h5')

print(f"\nExpression data shape: {adata.shape}")
print(f"Cells in .h5: {adata.n_obs}")
print(f"Genes in .h5: {adata.n_vars}")

# CRITICAL: Match cell IDs
print("\n" + "=" * 60)
print("MATCHING CELLS BETWEEN FILES")
print("=" * 60)

# Get cell IDs from both
h5_cells = set(adata.obs_names)
txt_cells = set(cell_types.index) if cell_types.index.name else set(cell_types.iloc[:, 0])

# Check overlap
common_cells = h5_cells.intersection(txt_cells)
print(f"Cells in .h5: {len(h5_cells)}")
print(f"Cells in .txt: {len(txt_cells)}")
print(f"Common cells: {len(common_cells)}")

if len(common_cells) > 0:
    print("\n✅ FILES MATCH! Ready to merge!")
else:
    print("\n⚠️ Cell IDs don't match - need to troubleshoot")
    print(f"Sample .h5 cell ID: {list(h5_cells)[:3]}")
    print(f"Sample .txt cell ID: {list(txt_cells)[:3]}")
