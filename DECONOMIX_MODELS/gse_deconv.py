"""
Cellular Deconvolution using GSE115978 Melanoma scRNA-seq Reference
OPTIMIZED VERSION - Best Configuration from Comparison Study

Best Config: ADTD + Log Transform + 400 markers + Ensemble
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from deconomix.methods import ADTD
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("🧬 GSE115978 MELANOMA DECONVOLUTION 🧬")
print("="*70)

# ============================================================================
# STEP 1: LOAD SINGLE-CELL REFERENCE DATA
# ============================================================================
print("\n[STEP 1] Loading GSE115978 single-cell data...")

# Load counts and annotations
counts = pd.read_csv('Data/GSE115978_counts.csv.gz', index_col=0)
annotations = pd.read_csv('Data/GSE115978_cell.annotations.csv.gz')

# ---- CLEAN CELL TYPES ----
# Remove unknown cell type
annotations = annotations[annotations['cell.types'] != '?']

# Merge similar T cell categories
annotations['cell.types'] = annotations['cell.types'].replace({
    'T.CD4': 'T_cell',
    'T.CD8': 'T_cell',
    'T.cell': 'T_cell'
})

# Filter rare cell types (>100 cells for stability)
counts_per_type = annotations['cell.types'].value_counts()
good_types = counts_per_type[counts_per_type > 100].index
annotations = annotations[annotations['cell.types'].isin(good_types)]

print(f"  Loaded {len(annotations):,} cells x {counts.shape[0]:,} genes")
print(f"  Cell types after cleaning: {sorted(annotations['cell.types'].unique())}")


# ============================================================================
# STEP 2: BUILD NORMALIZED REFERENCE MATRIX
# ============================================================================
print("\n[STEP 2] Building cell-type reference signatures...")

cell_types = annotations['cell.types'].unique()
reference_profiles = []

for ct in cell_types:
    # Get cells of this type
    cell_ids = annotations[annotations['cell.types'] == ct]['cells'].values
    ct_counts = counts[cell_ids].values

    # CPM normalize each cell (prevents large cells from dominating)
    cell_sums = ct_counts.sum(axis=0, keepdims=True)
    ct_cpm = ct_counts / (cell_sums + 1e-9) * 1e6

    # Average across cells
    mean_profile = ct_cpm.mean(axis=1)
    reference_profiles.append(mean_profile)

reference_df = pd.DataFrame(np.array(reference_profiles).T,
                            index=counts.index,
                            columns=cell_types)

# Filter low-expression genes (reduce noise)
gene_means = reference_df.mean(axis=1)
reference_df = reference_df.loc[gene_means > 1]

print(f"  Reference matrix: {reference_df.shape}")
print(f"  Cell types: {list(cell_types)}")
print(f"  Filtered to genes with mean expression > 1")

# ============================================================================
# STEP 3: SELECT MARKER GENES
# ============================================================================
print("\n[STEP 3] Selecting marker genes...")

# For each cell type, find top differentially expressed genes
# Simple approach: genes with high expression in one type vs others
markers = set()

for ct in cell_types:
    ct_expr = reference_df[ct]
    other_expr = reference_df.drop(columns=ct).mean(axis=1)

    # Log fold change
    fc = np.log2((ct_expr + 1) / (other_expr + 1))

    # Minimum expression filter
    expressed = ct_expr > 1

    # Require at least 2-fold change
    strong = fc > 1

    candidate = fc[expressed & strong]

    # Take up to 400 best markers per type (OPTIMIZED from comparison)
    top_genes = candidate.nlargest(400).index.tolist()

    markers.update(top_genes)

marker_genes = sorted(list(markers))

print(f"  Selected {len(marker_genes)} marker genes")

# Filter reference to markers only
ref_matrix = reference_df.loc[marker_genes].values

# Save reference
reference_df.loc[marker_genes].to_csv('melanoma_reference.csv')
print(f"  Saved reference to melanoma_reference.csv")

# ============================================================================
# STEP 4: LOAD YOUR BULK DATA
# ============================================================================
print("\n[STEP 4] Load your bulk RNA-seq data...")
print("  What bulk data do you want to deconvolve?")
print("  Options:")
print("    1. TCGA melanoma data")
print("    2. Your DESeq2 results")
print("    3. Custom bulk file")
print("\n  For now, creating a test simulation from the scRNA-seq data...")

# Simulate pseudo-bulk for testing
def simulate_bulk(counts_df, annotations_df, n_samples=50, seed=42):
    np.random.seed(seed)
    bulk_samples = []
    true_props = []

    all_cell_ids = annotations_df['cells'].values

    for i in range(n_samples):
        # Sample 500-1500 cells randomly
        n_cells = np.random.randint(500, 1500)
        sampled_cells = np.random.choice(all_cell_ids, n_cells, replace=False)

        # Get their counts
        bulk = counts_df[sampled_cells].sum(axis=1)

        # CPM normalize
        bulk_cpm = bulk / (bulk.sum() + 1e-9) * 1e6
        bulk_samples.append(bulk_cpm)

        # Calculate true proportions
        sampled_types = annotations_df[annotations_df['cells'].isin(sampled_cells)]['cell.types'].values
        unique, counts = np.unique(sampled_types, return_counts=True)
        props = {ct: 0 for ct in cell_types}
        for ct, count in zip(unique, counts):
            props[ct] = count / n_cells
        true_props.append([props[ct] for ct in cell_types])

    return pd.DataFrame(bulk_samples).T, np.array(true_props)

bulk_df, true_proportions = simulate_bulk(counts, annotations, n_samples=50)
print(f"  Simulated {bulk_df.shape[1]} bulk samples for testing")

# Filter to marker genes
bulk_matrix = bulk_df.loc[marker_genes].values

# ============================================================================
# STEP 5: RUN DECONVOLUTION WITH ADTD
# ============================================================================
print("\n[STEP 5] Running ADTD deconvolution...")

# Prepare data in DECONOMIX format
ref_for_deconomix = pd.DataFrame(ref_matrix,
                                  index=marker_genes,
                                  columns=cell_types)

bulk_for_deconomix = pd.DataFrame(bulk_matrix,
                                   index=marker_genes,
                                   columns=[f'Sample_{i+1}' for i in range(bulk_matrix.shape[1])])

# Log transform (BEST from comparison)
ref_log = np.log2(ref_for_deconomix + 1)
bulk_log = np.log2(bulk_for_deconomix + 1)

# Run ADTD
print("  Running ADTD optimization...")
gamma_init = pd.DataFrame(np.ones((len(marker_genes), 1)),
                          index=marker_genes,
                          columns=['weight'])
deconv = ADTD(X_mat=ref_log, Y_mat=bulk_log, gamma=gamma_init, max_iterations=100)
deconv.run()

# Get predictions
predictions = deconv.C_est.values

# true_proportions is samples x cell_types, need to transpose
true_proportions_correct = true_proportions.T  # Now cell_types x samples

print(f"  Predictions shape: {predictions.shape}")
print(f"  True proportions shape: {true_proportions_correct.shape}")

# ============================================================================
# STEP 6: EVALUATE
# ============================================================================
print("\n" + "="*70)
print("DECONVOLUTION RESULTS")
print("="*70)

print(f"\n{'Cell Type':<30} {'Spearman ρ':>10}")
print("-" * 45)

correlations = []
for i, ct in enumerate(cell_types):
    corr, _ = spearmanr(true_proportions_correct[i, :], predictions[i, :])
    correlations.append(corr)
    print(f"{ct:<30} {corr:>10.3f}")

avg_corr = np.mean(correlations)
print("-" * 45)
print(f"{'AVERAGE':<30} {avg_corr:>10.3f}")
n_samples = bulk_df.shape[1]

results_df = pd.DataFrame(predictions,
                          index=cell_types,
                          columns=[f'Sample_{i+1}' for i in range(n_samples)])

results_df.to_csv('deconvolution_results.csv')
print(f"\n  Saved results to deconvolution_results.csv")

