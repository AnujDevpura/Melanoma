"""
COMPREHENSIVE DECONVOLUTION METHOD COMPARISON
Tests multiple methods and parameters to find the best approach
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import nnls
from deconomix.methods import ADTD, DTD
import warnings
import time

warnings.filterwarnings('ignore')

print("="*80)
print("🧬 GSE115978 MELANOMA DECONVOLUTION - METHOD COMPARISON 🧬")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading GSE115978 single-cell data...")

counts = pd.read_csv('Data/GSE115978_counts.csv.gz', index_col=0)
annotations = pd.read_csv('Data/GSE115978_cell.annotations.csv.gz')

# Clean cell types
annotations = annotations[annotations['cell.types'] != '?']
annotations['cell.types'] = annotations['cell.types'].replace({
    'T.CD4': 'T_cell',
    'T.CD8': 'T_cell',
    'T.cell': 'T_cell'
})

# OPTIONAL: Filter rare cell types (>100 cells)
counts_per_type = annotations['cell.types'].value_counts()
print(f"\n  Cells per type:\n{counts_per_type}")
good_types = counts_per_type[counts_per_type > 100].index
annotations = annotations[annotations['cell.types'].isin(good_types)]

print(f"\n  After filtering: {len(annotations)} cells, {len(good_types)} cell types")

# ============================================================================
# STEP 2: BUILD REFERENCE
# ============================================================================
print("\n[STEP 2] Building reference matrix...")

cell_types = sorted(annotations['cell.types'].unique())
reference_profiles = []

for ct in cell_types:
    cell_ids = annotations[annotations['cell.types'] == ct]['cells'].values
    ct_counts = counts[cell_ids].values
    cell_sums = ct_counts.sum(axis=0, keepdims=True)
    ct_cpm = ct_counts / (cell_sums + 1e-9) * 1e6
    mean_profile = ct_cpm.mean(axis=1)
    reference_profiles.append(mean_profile)

reference_df = pd.DataFrame(np.array(reference_profiles).T,
                            index=counts.index,
                            columns=cell_types)

print(f"  Reference shape: {reference_df.shape}")

# ============================================================================
# STEP 3: SIMULATE BULK
# ============================================================================
print("\n[STEP 3] Simulating pseudo-bulk samples...")

def simulate_bulk(counts_df, annotations_df, n_samples=100, seed=42):
    np.random.seed(seed)
    bulk_samples = []
    true_props = []
    all_cell_ids = annotations_df['cells'].values

    for i in range(n_samples):
        n_cells = np.random.randint(500, 1500)
        sampled_cells = np.random.choice(all_cell_ids, n_cells, replace=False)
        bulk = counts_df[sampled_cells].sum(axis=1)
        bulk_cpm = bulk / (bulk.sum() + 1e-9) * 1e6
        bulk_samples.append(bulk_cpm)

        sampled_types = annotations_df[annotations_df['cells'].isin(sampled_cells)]['cell.types'].values
        unique, counts = np.unique(sampled_types, return_counts=True)
        props = {ct: 0 for ct in cell_types}
        for ct, count in zip(unique, counts):
            props[ct] = count / n_cells
        true_props.append([props[ct] for ct in cell_types])

    return pd.DataFrame(bulk_samples).T, np.array(true_props).T

bulk_df, true_proportions = simulate_bulk(counts, annotations, n_samples=100)
print(f"  Simulated {bulk_df.shape[1]} samples")
print(f"  True proportions shape: {true_proportions.shape}")

# ============================================================================
# EXPERIMENT: TEST DIFFERENT MARKER COUNTS
# ============================================================================

results_summary = []

for n_markers in [100, 200, 300, 400]:
    print(f"\n{'='*80}")
    print(f"TESTING WITH {n_markers} MARKERS PER CELL TYPE")
    print(f"{'='*80}")

    # Select markers
    markers = set()
    for ct in cell_types:
        ct_expr = reference_df[ct]
        other_expr = reference_df.drop(columns=ct).mean(axis=1)
        fc = np.log2((ct_expr + 1) / (other_expr + 1))
        expressed = ct_expr > 1
        strong = fc > 1
        candidate = fc[expressed & strong]
        top_genes = candidate.nlargest(n_markers).index.tolist()
        markers.update(top_genes)

    marker_genes = sorted(list(markers))
    print(f"  Total unique markers: {len(marker_genes)}")

    # Filter data
    ref_matrix = reference_df.loc[marker_genes]
    bulk_matrix = bulk_df.loc[marker_genes]

    # Test multiple methods
    methods_to_test = [
        ('NNLS', 'baseline'),
        ('ADTD', 'linear'),
        ('ADTD', 'log'),
        ('DTD', 'linear'),
        ('DTD', 'log')
    ]

    for method_name, transform in methods_to_test:
        print(f"\n  → Testing {method_name} with {transform} transform...")

        try:
            start = time.time()

            # Apply transform
            if transform == 'log':
                ref_data = np.log2(ref_matrix + 1)
                bulk_data = np.log2(bulk_matrix + 1)
            else:
                ref_data = ref_matrix.copy()
                bulk_data = bulk_matrix.copy()

            # Run method
            if method_name == 'NNLS':
                predictions = np.zeros((len(cell_types), bulk_data.shape[1]))
                for i in range(bulk_data.shape[1]):
                    coef, _ = nnls(ref_data.values, bulk_data.values[:, i])
                    predictions[:, i] = coef / (coef.sum() + 1e-9)

            elif method_name == 'ADTD':
                gamma_init = pd.DataFrame(np.ones((len(marker_genes), 1)),
                                          index=marker_genes,
                                          columns=['weight'])
                deconv = ADTD(X_mat=ref_data, Y_mat=bulk_data, gamma=gamma_init,
                             max_iterations=50)  # Limit iterations for speed
                deconv.run()
                predictions = deconv.C_est.values

            elif method_name == 'DTD':
                # DTD needs initial guess C_mat
                C_init = pd.DataFrame(np.ones((bulk_data.shape[1], len(cell_types))) / len(cell_types),
                                     columns=cell_types,
                                     index=bulk_data.columns)
                deconv = DTD(X_mat=ref_data, Y_mat=bulk_data, C_mat=C_init)
                predictions = deconv.C_est.T.values  # DTD returns samples x cell_types

            elapsed = time.time() - start

            # Evaluate
            correlations = []
            for i in range(len(cell_types)):
                corr, _ = spearmanr(true_proportions[i, :], predictions[i, :])
                correlations.append(corr)

            avg_corr = np.mean(correlations)

            print(f"    ✓ Avg Correlation: {avg_corr:.3f} (Time: {elapsed:.1f}s)")

            results_summary.append({
                'n_markers': n_markers,
                'method': method_name,
                'transform': transform,
                'avg_correlation': avg_corr,
                'time': elapsed
            })

        except Exception as e:
            print(f"    ✗ Failed: {str(e)[:60]}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 FINAL COMPARISON SUMMARY")
print("="*80)

summary_df = pd.DataFrame(results_summary)
summary_df = summary_df.sort_values('avg_correlation', ascending=False)

print("\nTop 10 Best Configurations:\n")
print(summary_df.head(10).to_string(index=False))

# Save results
summary_df.to_csv('method_comparison_results.csv', index=False)
print(f"\n✅ Full results saved to method_comparison_results.csv")

# Print best config
best = summary_df.iloc[0]
print(f"\n{'='*80}")
print(f"🏆 BEST CONFIGURATION:")
print(f"   Method: {best['method']}")
print(f"   Transform: {best['transform']}")
print(f"   Markers: {int(best['n_markers'])} per cell type")
print(f"   Avg Correlation: {best['avg_correlation']:.3f}")
print(f"   Time: {best['time']:.1f}s")
print(f"{'='*80}")
