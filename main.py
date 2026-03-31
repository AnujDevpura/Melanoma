"""
Differential Expression Analysis using PyDESeq2
FIXED VERSION: Handles duplicate gene names properly
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Try to import PyDESeq2
try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
except ImportError:
    print("ERROR: PyDESeq2 not installed!")
    print("Install it with: pip install pydeseq2")
    exit(1)

print("=" * 60)
print("DIFFERENTIAL EXPRESSION ANALYSIS - PyDESeq2")
print("=" * 60)

# =============================================================================
# Configuration - CHANGE THIS FOR DIFFERENT COMPARISONS
# =============================================================================
COMPARISON_TYPE = "vital_status"  # Options: "vital_status", "gender", "stage_simple", "survival_group"
GROUP_A = "Alive"  # First group
GROUP_B = "Dead"  # Second group

print(f"\nComparison: {GROUP_A} vs {GROUP_B}")
print(f"Using clinical column: {COMPARISON_TYPE}")

# =============================================================================
# Load matched data
# =============================================================================
print("\n[1] Loading matched data...")

data_dir = Path("tcga_skcm_data")
counts = pd.read_csv(data_dir / "count_matrix_matched.csv", index_col=0)
clinical = pd.read_csv(data_dir / "clinical_data_matched.csv")

print(f"✓ Count matrix: {counts.shape[0]} genes × {counts.shape[1]} samples")
print(f"✓ Clinical data: {clinical.shape[0]} samples")

# =============================================================================
# FIX: Handle duplicate gene names - SIMPLE APPROACH
# =============================================================================
print("\n[1.5] Checking for duplicate gene names...")

duplicates = counts.index.duplicated()
n_duplicates = duplicates.sum()

if n_duplicates > 0:
    print(f"⚠ Found {n_duplicates} duplicate gene names")
    print(f"  Fixing by making gene names unique...")

    # Simple solution: Use pandas' built-in method to make names unique
    # This adds .1, .2, .3 etc to duplicates
    new_index = []
    name_counts = {}

    for name in counts.index:
        if name in name_counts:
            name_counts[name] += 1
            new_name = f"{name}.{name_counts[name]}"
        else:
            name_counts[name] = 0
            new_name = name
        new_index.append(new_name)

    counts.index = new_index

    print(f"✓ Fixed: All gene names are now unique")
    print(f"  Example: Duplicates like 'GENE' become 'GENE', 'GENE.1', 'GENE.2', etc.")
else:
    print(f"✓ No duplicate gene names found")

# Verify fix worked
assert not counts.index.duplicated().any(), "ERROR: Still have duplicate gene names!"
print(f"✓ Verified: All {len(counts.index)} gene names are unique")

# =============================================================================
# Filter samples for this comparison
# =============================================================================
print(f"\n[2] Filtering samples for {GROUP_A} vs {GROUP_B}...")

# Get samples for each group
group_a_samples = clinical[clinical[COMPARISON_TYPE] == GROUP_A]["submitter_id"].values
group_b_samples = clinical[clinical[COMPARISON_TYPE] == GROUP_B]["submitter_id"].values

print(f"✓ {GROUP_A}: {len(group_a_samples)} samples")
print(f"✓ {GROUP_B}: {len(group_b_samples)} samples")

# Make sure samples exist in count matrix
group_a_samples = [s for s in group_a_samples if s in counts.columns]
group_b_samples = [s for s in group_b_samples if s in counts.columns]

print(
    f"✓ After filtering: {GROUP_A}={len(group_a_samples)}, {GROUP_B}={len(group_b_samples)}"
)

# Combine samples
all_samples = list(group_a_samples) + list(group_b_samples)
counts_filtered = counts[all_samples]

# Create metadata dataframe for PyDESeq2
metadata = pd.DataFrame(
    {
        "sample_id": all_samples,
        "condition": [GROUP_A] * len(group_a_samples)
        + [GROUP_B] * len(group_b_samples),
    }
)
metadata = metadata.set_index("sample_id")

print(f"✓ Total samples for analysis: {len(all_samples)}")

# =============================================================================
# Optional: Filter low-count genes
# =============================================================================
print("\n[3] Filtering low-count genes...")

# Keep genes with at least 10 counts in at least 10 samples
min_counts = 10
min_samples = 10

keep_genes = (counts_filtered >= min_counts).sum(axis=1) >= min_samples
counts_filtered = counts_filtered[keep_genes]

print(f"✓ Kept {counts_filtered.shape[0]} / {counts.shape[0]} genes")
print(f"  (genes with ≥{min_counts} counts in ≥{min_samples} samples)")

# =============================================================================
# Prepare data for PyDESeq2
# =============================================================================
print("\n[4] Preparing data for DESeq2...")

# PyDESeq2 expects counts as integers
counts_int = counts_filtered.astype(int)

# Transpose: PyDESeq2 wants samples as rows, genes as columns
counts_deseq = counts_int.T

print(f"✓ Data shape for DESeq2: {counts_deseq.shape} (samples × genes)")
print(f"✓ Metadata shape: {metadata.shape}")

# Double-check no duplicates in gene names
assert not counts_deseq.columns.duplicated().any(), "Still have duplicate gene names!"
print(f"✓ Verified: All gene names are unique")

# =============================================================================
# Run DESeq2
# =============================================================================
print("\n[5] Running DESeq2 analysis...")
print("   This may take a few minutes for ~23k genes...")

# Create DESeq2 dataset
dds = DeseqDataSet(
    counts=counts_deseq,
    metadata=metadata,
    design_factors="condition",  # Will show deprecation warning - that's okay
    refit_cooks=True,
    n_cpus=4,  # Use multiple cores if available
)

# Run the analysis step by step to see progress
print("\n   Step 1/3: Fitting size factors...")
dds.fit_size_factors()
print("   ✓ Size factors fitted")

print("   Step 2/3: Fitting dispersions...")
dds.fit_genewise_dispersions()
print("   ✓ Genewise dispersions fitted")

print("   Step 3/3: Fitting dispersion trend and LFC...")
dds.fit_dispersion_trend()
dds.fit_dispersion_prior()
dds.fit_LFC()
print("   ✓ All fitting complete")

print("   Step 4/4: Refitting Cooks outliers...")
dds.refit()
print("   ✓ Refitting complete")

print("\n✓ DESeq2 analysis complete!")

# =============================================================================
# Get results
# =============================================================================
print("\n[6] Computing differential expression statistics...")

# Compute statistics
stat_res = DeseqStats(
    dds,
    contrast=["condition", GROUP_A, GROUP_B],
    alpha=0.05,
    cooks_filter=True,
    independent_filter=True,
)

stat_res.summary()
results = stat_res.results_df

print("✓ Statistical testing complete!")

# =============================================================================
# Process and save results
# =============================================================================
print("\n[7] Processing results...")

# Clean up gene names for display (remove .1, .2 suffixes)
results_display = results.copy()
results_display.index = results_display.index.str.replace(r"\.\d+$", "", regex=True)

# Sort by adjusted p-value
results_sorted = results_display.sort_values("padj")

# Add interpretation columns
results_sorted["significant"] = results_sorted["padj"] < 0.05
results_sorted["highly_significant"] = results_sorted["padj"] < 0.01
results_sorted["direction"] = results_sorted["log2FoldChange"].apply(
    lambda x: f"Higher in {GROUP_A}" if x > 0 else f"Higher in {GROUP_B}"
)

# Summary statistics
n_significant = (results_sorted["padj"] < 0.05).sum()
n_up = ((results_sorted["padj"] < 0.05) & (results_sorted["log2FoldChange"] > 0)).sum()
n_down = (
    (results_sorted["padj"] < 0.05) & (results_sorted["log2FoldChange"] < 0)
).sum()

print(f"\n✓ Significant genes (padj < 0.05): {n_significant}")
print(f"  - Higher in {GROUP_A}: {n_up}")
print(f"  - Higher in {GROUP_B}: {n_down}")

# =============================================================================
# Save results
# =============================================================================
output_dir = data_dir / "deseq_results"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / f"DESeq2_{GROUP_A}_vs_{GROUP_B}.csv"
results_sorted.to_csv(output_file)

print(f"\n✓ Saved full results: {output_file}")

# Save top genes
top_genes_file = output_dir / f"DESeq2_{GROUP_A}_vs_{GROUP_B}_top_genes.csv"
top_genes = results_sorted[results_sorted["significant"]].head(100)
top_genes.to_csv(top_genes_file)

print(f"✓ Saved top 100 significant genes: {top_genes_file}")

# =============================================================================
# Display top results
# =============================================================================
print("\n" + "=" * 80)
print(f"TOP 20 DIFFERENTIALLY EXPRESSED GENES: {GROUP_A} vs {GROUP_B}")
print("=" * 80)

top_20 = results_sorted.head(20)
display_cols = ["baseMean", "log2FoldChange", "pvalue", "padj", "direction"]
print(top_20[display_cols].to_string())

print(f"\nlog2FC interpretation:")
print(f"  Positive = higher in {GROUP_A}")
print(f"  Negative = higher in {GROUP_B}")
print(f"  |log2FC| > 1 means > 2-fold difference")
print(f"  |log2FC| > 2 means > 4-fold difference")

# =============================================================================
# Create visualization
# =============================================================================
print("\n[8] Creating visualizations...")

# Volcano plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Prepare data for plotting
results_plot = results_sorted.copy()
# Handle padj=0 or very small values
results_plot["-log10(padj)"] = -np.log10(results_plot["padj"].clip(lower=1e-300))

# Color by significance
colors = []
for _, row in results_plot.iterrows():
    if pd.notna(row["padj"]) and row["padj"] < 0.05:
        if row["log2FoldChange"] > 0:
            colors.append("red")
        else:
            colors.append("blue")
    else:
        colors.append("gray")

ax1.scatter(
    results_plot["log2FoldChange"],
    results_plot["-log10(padj)"],
    c=colors,
    alpha=0.5,
    s=10,
)
ax1.axhline(
    -np.log10(0.05), color="black", linestyle="--", linewidth=0.5, label="padj=0.05"
)
ax1.axvline(1, color="black", linestyle="--", linewidth=0.5)
ax1.axvline(-1, color="black", linestyle="--", linewidth=0.5)
ax1.set_xlabel("log2(Fold Change)", fontsize=12)
ax1.set_ylabel("-log10(adjusted p-value)", fontsize=12)
ax1.set_title(f"Volcano Plot: {GROUP_A} vs {GROUP_B}", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="red", label=f"Higher in {GROUP_A}"),
    Patch(facecolor="blue", label=f"Higher in {GROUP_B}"),
    Patch(facecolor="gray", label="Not significant"),
]
ax1.legend(handles=legend_elements, loc="upper right")

# MA plot
ax2.scatter(
    np.log10(results_plot["baseMean"].clip(lower=1)),
    results_plot["log2FoldChange"],
    c=colors,
    alpha=0.5,
    s=10,
)
ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
ax2.axhline(1, color="black", linestyle="--", linewidth=0.5)
ax2.axhline(-1, color="black", linestyle="--", linewidth=0.5)
ax2.set_xlabel("log10(Mean Expression)", fontsize=12)
ax2.set_ylabel("log2(Fold Change)", fontsize=12)
ax2.set_title(f"MA Plot: {GROUP_A} vs {GROUP_B}", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = output_dir / f"volcano_ma_plot_{GROUP_A}_vs_{GROUP_B}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
print(f"✓ Saved plots: {plot_file}")
plt.close()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("DIFFERENTIAL EXPRESSION ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\nComparison: {GROUP_A} vs {GROUP_B}")
print(f"Total genes tested: {len(results_sorted)}")
print(f"Significant genes (padj < 0.05): {n_significant}")
print(f"  - Upregulated in {GROUP_A}: {n_up}")
print(f"  - Downregulated in {GROUP_A}: {n_down}")

print(f"\nFiles created:")
print(f"  1. Full results: DESeq2_{GROUP_A}_vs_{GROUP_B}.csv")
print(f"  2. Top genes: DESeq2_{GROUP_A}_vs_{GROUP_B}_top_genes.csv")
print(f"  3. Plots: volcano_ma_plot_{GROUP_A}_vs_{GROUP_B}.png")

print(f"\n" + "=" * 60)
print("NEXT STEPS (Task b & c):")
print("=" * 60)
print("Task b) Look up interesting genes:")
print("  - Check CellMarker 2.0: http://bio-bigdata.hrbmu.edu.cn/CellMarker/")
print("  - Search PubMed: https://pubmed.ncbi.nlm.nih.gov/")
print("  - Use GeneCards: https://www.genecards.org/")
print("\nTask c) Pathway analysis:")
print("  - Run: python 03_pathway_analysis.py")
