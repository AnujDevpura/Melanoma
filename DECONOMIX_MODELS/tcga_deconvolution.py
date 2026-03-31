"""
Apply GSE115978 Reference to TCGA Melanoma Bulk RNA-seq
Deconvolves cell type proportions in TCGA SKCM samples
"""

import numpy as np
import pandas as pd
from deconomix.methods import ADTD
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("🧬 TCGA MELANOMA DECONVOLUTION 🧬")
print("="*70)

# ============================================================================
# STEP 1: LOAD REFERENCE (from GSE115978)
# ============================================================================
print("\n[STEP 1] Loading reference matrix from GSE115978...")

# Load the reference we already created
reference_df = pd.read_csv('melanoma_reference.csv', index_col=0)
cell_types = list(reference_df.columns)

print(f"  Reference: {reference_df.shape[0]} genes x {len(cell_types)} cell types")
print(f"  Cell types: {cell_types}")

# ============================================================================
# STEP 2: LOAD TCGA BULK DATA
# ============================================================================
print("\n[STEP 2] Loading TCGA SKCM bulk RNA-seq data...")

# Load count matrix
tcga_counts = pd.read_csv('../tcga_skcm_data/count_matrix.csv', index_col=0)
print(f"  TCGA data: {tcga_counts.shape[0]} genes x {tcga_counts.shape[1]} samples")

# Load metadata
metadata = pd.read_csv('../TCGA_metadata_clean.csv', index_col=0)
print(f"  Metadata: {metadata.shape[0]} samples")

# ============================================================================
# STEP 3: MATCH GENES BETWEEN REFERENCE AND BULK
# ============================================================================
print("\n[STEP 3] Matching genes between reference and TCGA...")

# Find common genes
common_genes = list(set(reference_df.index) & set(tcga_counts.index))
print(f"  Common genes: {len(common_genes)}")

# Filter both to common genes
ref_matched = reference_df.loc[common_genes]
tcga_matched = tcga_counts.loc[common_genes]

print(f"  Reference after matching: {ref_matched.shape}")
print(f"  TCGA after matching: {tcga_matched.shape}")

# ============================================================================
# STEP 4: NORMALIZE TCGA DATA
# ============================================================================
print("\n[STEP 4] Normalizing TCGA data...")

# CPM normalize TCGA data (same as reference)
tcga_cpm = tcga_matched / (tcga_matched.sum(axis=0) + 1e-9) * 1e6

# Log transform (BEST from comparison)
ref_log = np.log2(ref_matched + 1)
tcga_log = np.log2(tcga_cpm + 1)

print(f"  TCGA CPM range: {tcga_cpm.values.min():.2f} - {tcga_cpm.values.max():.2f}")
print(f"  TCGA log range: {tcga_log.values.min():.2f} - {tcga_log.values.max():.2f}")

# ============================================================================
# STEP 5: RUN DECONVOLUTION
# ============================================================================
print("\n[STEP 5] Running ADTD deconvolution on TCGA samples...")
print(f"  This will deconvolve {tcga_log.shape[1]} TCGA samples...")

# Initialize gamma
gamma_init = pd.DataFrame(np.ones((len(common_genes), 1)),
                          index=common_genes,
                          columns=['weight'])

# Run ADTD
print("  Running ADTD optimization (this may take a few minutes)...")
deconv = ADTD(X_mat=ref_log, Y_mat=tcga_log, gamma=gamma_init, max_iterations=100)
deconv.run()

# Get predictions
predictions = deconv.C_est  # DataFrame: cell_types x samples

print(f"  Predictions shape: {predictions.shape}")

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================
print("\n[STEP 6] Saving deconvolution results...")

# Save full results
predictions.to_csv('TCGA_SKCM_cell_proportions.csv')
print(f"  ✓ Saved TCGA_SKCM_cell_proportions.csv")

# ============================================================================
# STEP 7: MERGE WITH METADATA AND ANALYZE
# ============================================================================
print("\n[STEP 7] Merging with clinical metadata...")

# Transpose predictions (samples x cell_types)
props_df = predictions.T

# Merge with metadata
# TCGA sample IDs need cleaning (first 15 chars = patient ID)
props_df.index = [idx[:15] if len(idx) >= 15 else idx for idx in props_df.index]
metadata.index = [idx[:15] if len(idx) >= 15 else idx for idx in metadata.index]

# Merge
merged = props_df.join(metadata, how='inner')
print(f"  Merged data: {merged.shape}")

# Save merged results
merged.to_csv('TCGA_SKCM_proportions_with_metadata.csv')
print(f"  ✓ Saved TCGA_SKCM_proportions_with_metadata.csv")

# ============================================================================
# STEP 8: SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print("\nAverage cell type proportions across all TCGA samples:")
print(props_df[cell_types].mean().sort_values(ascending=False))

print("\nStandard deviation:")
print(props_df[cell_types].std().sort_values(ascending=False))

# Check if we have vital status info
if 'vital_status' in merged.columns:
    print("\n" + "="*70)
    print("CELL PROPORTIONS BY SURVIVAL STATUS")
    print("="*70)

    for status in merged['vital_status'].unique():
        if pd.notna(status):
            subset = merged[merged['vital_status'] == status]
            print(f"\n{status} (n={len(subset)}):")
            print(subset[cell_types].mean().sort_values(ascending=False))

print("\n" + "="*70)
print("✅ DECONVOLUTION COMPLETE!")
print("="*70)
print("\nNext steps:")
print("  1. Analyze TCGA_SKCM_proportions_with_metadata.csv")
print("  2. Correlate cell proportions with survival outcomes")
print("  3. Find immune infiltration patterns in different tumor stages")
print("="*70)
