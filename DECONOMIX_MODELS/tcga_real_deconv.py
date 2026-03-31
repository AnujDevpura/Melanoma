"""
TCGA SKCM REAL DECONVOLUTION
Uses TCGA-SKCM TPM data with GSE115978 melanoma reference
"""

import numpy as np
import pandas as pd
from deconomix.methods import ADTD
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("🧬 TCGA SKCM DECONVOLUTION - REAL DATA 🧬")
print("="*70)

# ============================================================================
# STEP 1: LOAD REFERENCE (GSE115978 melanoma single-cell)
# ============================================================================
print("\n[STEP 1] Loading melanoma reference...")

reference_df = pd.read_csv('melanoma_reference.csv', index_col=0)
cell_types = list(reference_df.columns)
marker_genes = list(reference_df.index)

print(f"  Reference: {len(marker_genes)} marker genes x {len(cell_types)} cell types")
print(f"  Cell types: {cell_types}")

# ============================================================================
# STEP 2: LOAD TCGA BULK TPM DATA
# ============================================================================
print("\n[STEP 2] Loading TCGA SKCM bulk RNA-seq (TPM)...")

bulk_df = pd.read_csv('Data/TCGA-SKCM.star_tpm.tsv.gz', sep='\t', index_col=0)
print(f"  TCGA shape: {bulk_df.shape[0]} genes x {bulk_df.shape[1]} samples")
print(f"  Sample IDs preview: {list(bulk_df.columns[:3])}")

# ============================================================================
# STEP 3: CONVERT ENSEMBL IDs TO GENE SYMBOLS
# ============================================================================
print("\n[STEP 3] Converting Ensembl IDs to gene symbols...")

# Download gene mapping if not exists
import urllib.request
import os

mapping_file = 'Data/gencode.v36.gene.probemap'
if not os.path.exists(mapping_file):
    print("  Downloading gene ID mapping...")
    url = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/gencode.v36.annotation.gtf.gene.probemap"
    urllib.request.urlretrieve(url, mapping_file)
    print("  OK Downloaded mapping file")

# Load mapping
mapping = pd.read_csv(mapping_file, sep='\t')
mapping = mapping[['id', 'gene']].copy()
mapping.columns = ['ensembl', 'symbol']

# Remove version numbers from Ensembl IDs
bulk_df.index = bulk_df.index.str.split('.').str[0]
mapping['ensembl'] = mapping['ensembl'].str.split('.').str[0]

print(f"  Mapping file: {len(mapping)} genes")

# Merge with bulk data
# Reset index creates a column with the index name, need to get its actual name
bulk_df_reset = bulk_df.reset_index()
index_col_name = bulk_df_reset.columns[0]  # Get the actual name of the index column
bulk_df = bulk_df_reset.merge(mapping, left_on=index_col_name, right_on='ensembl', how='left')
bulk_df = bulk_df.dropna(subset=['symbol'])  # Drop unmapped genes
bulk_df = bulk_df.drop(columns=[index_col_name, 'ensembl'])
bulk_df = bulk_df.set_index('symbol')

# Handle duplicates (take mean if multiple Ensembl IDs map to same symbol)
bulk_df = bulk_df.groupby(bulk_df.index).mean()

print(f"  After mapping: {bulk_df.shape[0]} genes x {bulk_df.shape[1]} samples")

# ============================================================================
# STEP 4: MATCH GENES BETWEEN REFERENCE AND BULK
# ============================================================================
print("\n[STEP 4] Matching genes between reference and TCGA...")

common_genes = sorted(set(marker_genes) & set(bulk_df.index))
print(f"  Common marker genes: {len(common_genes)}")

if len(common_genes) < 100:
    print("  WARNING  WARNING: Very few genes overlap. Check gene naming.")
else:
    print(f"  OK Good overlap: {len(common_genes)}/{len(marker_genes)} markers found")

# Filter both to common genes
ref_matched = reference_df.loc[common_genes]
bulk_matched = bulk_df.loc[common_genes]

print(f"  Reference: {ref_matched.shape}")
print(f"  TCGA: {bulk_matched.shape}")

# ============================================================================
# STEP 5: PREPARE DATA FOR ADTD
# ============================================================================
print("\n[STEP 5] Preparing data (log transform)...")

# TCGA data is already TPM, just need log transform to match reference
# Reference is in CPM (similar scale to TPM), so we log2 both
ref_log = np.log2(ref_matched + 1)
bulk_log = np.log2(bulk_matched + 1)

print(f"  Reference log range: {ref_log.values.min():.2f} - {ref_log.values.max():.2f}")
print(f"  TCGA log range: {bulk_log.values.min():.2f} - {bulk_log.values.max():.2f}")

# ============================================================================
# STEP 6: RUN DECONVOLUTION
# ============================================================================
print("\n[STEP 6] Running ADTD deconvolution...")
print(f"  Deconvolving {bulk_log.shape[1]} TCGA samples...")
print("  This may take 5-10 minutes...")

# Initialize gamma
gamma_init = pd.DataFrame(np.ones((len(common_genes), 1)),
                          index=common_genes,
                          columns=['weight'])

# Run ADTD
deconv = ADTD(X_mat=ref_log, Y_mat=bulk_log, gamma=gamma_init, max_iterations=100)
deconv.run()

# Get predictions with proper labels
# Convert to DataFrame with TCGA sample IDs preserved
predictions = pd.DataFrame(
    deconv.C_est.values,
    index=cell_types,              # Rows = cell types
    columns=bulk_log.columns       # Columns = TCGA sample IDs
)

print(f"  OK Predictions shape: {predictions.shape}")
print(f"  OK Sample IDs preserved: {list(predictions.columns[:3])}")

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================
print("\n[STEP 7] Saving results...")

# Save cell_types x samples orientation
predictions.to_csv('TCGA_SKCM_deconvolution_results.csv')
print(f"  OK Saved TCGA_SKCM_deconvolution_results.csv (cell_types x samples)")

# Transpose for easier analysis (samples x cell_types)
props_transposed = predictions.T
props_transposed.to_csv('TCGA_SKCM_cell_proportions.csv')
print(f"  OK Saved TCGA_SKCM_cell_proportions.csv (samples x cell_types)")
print(f"  OK Row labels: TCGA sample IDs preserved!")

# ============================================================================
# STEP 8: SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY - CELL TYPE PROPORTIONS IN TCGA SKCM")
print("="*70)

print("\nAverage proportions across all samples:")
print(props_transposed.mean().sort_values(ascending=False))

print("\nStandard deviation:")
print(props_transposed.std().sort_values(ascending=False))

print("\n" + "="*70)
print("SUCCESS DECONVOLUTION COMPLETE!")
print("="*70)
print(f"\nResults saved:")
print(f"  • TCGA_SKCM_deconvolution_results.csv  (cell_types x samples)")
print(f"  • TCGA_SKCM_cell_proportions.csv       (samples x cell_types)")
print(f"\nYou can now:")
print(f"  1. Merge with clinical metadata to analyze survival")
print(f"  2. Correlate immune infiltration with outcomes")
print(f"  3. Identify immune subtypes in melanoma")
print("="*70)
