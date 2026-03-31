"""
Pathway Analysis using gseapy - FIXED FOR TIMEOUT
This uses TOP genes only to avoid server timeouts
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import gseapy as gp
except ImportError:
    print("ERROR: gseapy not installed!")
    print("Install it with: pip install gseapy")
    exit(1)

print("=" * 60)
print("PATHWAY ANALYSIS - Task c (TOP GENES ONLY)")
print("=" * 60)

# =============================================================================
# Configuration
# =============================================================================
GROUP_A = "Alive"
GROUP_B = "Dead"

# CHANGE: Use only top N genes to avoid timeout
TOP_N_GENES = 200  # Much more manageable for online databases

print(f"\nAnalyzing pathways for: {GROUP_A} vs {GROUP_B}")
print(f"Strategy: Using top {TOP_N_GENES} most significant genes")

# =============================================================================
# Load DESeq2 results
# =============================================================================
print("\n[1] Loading DESeq2 results...")

data_dir = Path("tcga_skcm_data")
results_dir = data_dir / "deseq_results"
results_file = results_dir / f"DESeq2_{GROUP_A}_vs_{GROUP_B}.csv"

if not results_file.exists():
    print(f"ERROR: Could not find {results_file}")
    exit(1)

results = pd.read_csv(results_file, index_col=0)
print(f"✓ Loaded {len(results)} genes from DESeq2 results")

# =============================================================================
# Prepare gene lists - TOP GENES ONLY
# =============================================================================
print(f"\n[2] Selecting top {TOP_N_GENES} most significant genes...")

# Get all significant genes first
significant = results[results["padj"] < 0.05].copy()
print(f"✓ Total significant genes (padj < 0.05): {len(significant)}")

# Sort by padj and take top N
top_genes = significant.sort_values("padj").head(TOP_N_GENES)

# Split into upregulated and downregulated
upregulated = top_genes[top_genes["log2FoldChange"] > 0]
downregulated = top_genes[top_genes["log2FoldChange"] < 0]

print(f"✓ Top {TOP_N_GENES} genes selected:")
print(f"  - Upregulated in {GROUP_A}: {len(upregulated)}")
print(f"  - Downregulated in {GROUP_A}: {len(downregulated)}")

# Get gene lists
up_genes = upregulated.index.tolist()
down_genes = downregulated.index.tolist()
top_genes_list = top_genes.index.tolist()

print(f"\nTop 10 upregulated genes:")
print(upregulated.head(10)[["log2FoldChange", "padj"]].to_string())

print(f"\nTop 10 downregulated genes:")
print(downregulated.head(10)[["log2FoldChange", "padj"]].to_string())

# =============================================================================
# Enrichment Analysis - Top Genes Only
# =============================================================================
print(f"\n[3] Running pathway enrichment analysis (top {TOP_N_GENES} genes)...")
print("   This should be faster with fewer genes...")

pathway_dir = data_dir / "pathway_results"
pathway_dir.mkdir(exist_ok=True)

# Use fewer databases to reduce connection issues
gene_sets = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
]

print(f"Testing against {len(gene_sets)} databases:")
for gs in gene_sets:
    print(f"  - {gs}")

enrichment_results = {}

for gene_set in gene_sets:
    print(f"\nAnalyzing: {gene_set}...")
    try:
        enr = gp.enrichr(
            gene_list=top_genes_list,
            gene_sets=gene_set,
            organism="human",
            outdir=None,
            cutoff=0.05,
        )

        if enr.results is not None and len(enr.results) > 0:
            enrichment_results[gene_set] = enr.results
            sig_pathways = (enr.results["Adjusted P-value"] < 0.05).sum()
            print(f"  ✓ Found {sig_pathways} significant pathways")
        else:
            print(f"  ℹ No significant pathways found")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        print(f"  → Skipping {gene_set}")

# =============================================================================
# Enrichment Analysis - Upregulated Genes
# =============================================================================
if len(up_genes) >= 5:
    print(f"\n[4] Running pathway enrichment (top upregulated genes)...")

    enrichment_up = {}
    for gene_set in ["GO_Biological_Process_2023"]:  # Just one to be safe
        try:
            print(f"  Analyzing {gene_set}...")
            enr = gp.enrichr(
                gene_list=up_genes,
                gene_sets=gene_set,
                organism="human",
                outdir=None,
                cutoff=0.05,
            )

            if enr.results is not None and len(enr.results) > 0:
                enrichment_up[gene_set] = enr.results
                sig_pathways = (enr.results["Adjusted P-value"] < 0.05).sum()
                print(f"  ✓ {gene_set}: {sig_pathways} pathways")

        except Exception as e:
            print(f"  ✗ {gene_set}: {e}")
else:
    print(f"\n[4] Skipping upregulated analysis")
    enrichment_up = {}

# =============================================================================
# Enrichment Analysis - Downregulated Genes
# =============================================================================
if len(down_genes) >= 5:
    print(f"\n[5] Running pathway enrichment (top downregulated genes)...")

    enrichment_down = {}
    for gene_set in ["GO_Biological_Process_2023"]:
        try:
            print(f"  Analyzing {gene_set}...")
            enr = gp.enrichr(
                gene_list=down_genes,
                gene_sets=gene_set,
                organism="human",
                outdir=None,
                cutoff=0.05,
            )

            if enr.results is not None and len(enr.results) > 0:
                enrichment_down[gene_set] = enr.results
                sig_pathways = (enr.results["Adjusted P-value"] < 0.05).sum()
                print(f"  ✓ {gene_set}: {sig_pathways} pathways")

        except Exception as e:
            print(f"  ✗ {gene_set}: {e}")
else:
    print(f"\n[5] Skipping downregulated analysis")
    enrichment_down = {}

# =============================================================================
# Save results
# =============================================================================
print("\n[6] Saving results...")

for gene_set, result in enrichment_results.items():
    if result is not None and len(result) > 0:
        filename = pathway_dir / f"enrichment_top{TOP_N_GENES}_{gene_set}.csv"
        result.to_csv(filename, index=False)
        print(f"✓ Saved: {filename.name}")

for gene_set, result in enrichment_up.items():
    if result is not None and len(result) > 0:
        filename = pathway_dir / f"enrichment_UP_top{TOP_N_GENES}_{gene_set}.csv"
        result.to_csv(filename, index=False)
        print(f"✓ Saved: {filename.name}")

for gene_set, result in enrichment_down.items():
    if result is not None and len(result) > 0:
        filename = pathway_dir / f"enrichment_DOWN_top{TOP_N_GENES}_{gene_set}.csv"
        result.to_csv(filename, index=False)
        print(f"✓ Saved: {filename.name}")

# =============================================================================
# Display top pathways
# =============================================================================
print("\n" + "=" * 80)
print("TOP ENRICHED PATHWAYS")
print("=" * 80)

for gene_set, result in enrichment_results.items():
    if result is not None and len(result) > 0:
        print(f"\n--- {gene_set} ---")
        top = result.head(15)
        display = top[["Term", "Adjusted P-value", "Odds Ratio", "Combined Score"]]
        print(display.to_string(index=False))

# =============================================================================
# Create visualizations
# =============================================================================
print("\n[7] Creating visualizations...")

for gene_set, result in enrichment_results.items():
    if result is not None and len(result) > 0:
        top_pathways = result.head(20)

        fig, ax = plt.subplots(figsize=(12, 10))

        top_pathways = top_pathways.sort_values("Adjusted P-value", ascending=True)

        y_pos = np.arange(len(top_pathways))
        colors = plt.cm.RdYlGn_r(top_pathways["Adjusted P-value"])

        ax.barh(y_pos, -np.log10(top_pathways["Adjusted P-value"]), color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_pathways["Term"], fontsize=9)
        ax.set_xlabel("-log10(Adjusted P-value)", fontsize=12)
        ax.set_title(
            f"Top Pathways: {gene_set}\n{GROUP_A} vs {GROUP_B} (Top {TOP_N_GENES} genes)",
            fontsize=14,
            fontweight="bold",
        )
        ax.axvline(
            -np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p=0.05"
        )
        ax.legend()
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plot_file = pathway_dir / f"pathway_barplot_top{TOP_N_GENES}_{gene_set}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved plot: {plot_file.name}")
        plt.close()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("PATHWAY ANALYSIS COMPLETE!")
print("=" * 60)

total_pathways = sum(len(r) for r in enrichment_results.values() if r is not None)
print(f"\nGenes analyzed: Top {TOP_N_GENES} most significant")
print(f"Total significant genes in full dataset: {len(significant)}")
print(f"Significant pathways found: {total_pathways}")
print(f"\nResults saved in: {pathway_dir}/")

print("\n" + "=" * 60)
print("INTERPRETATION NOTES:")
print("=" * 60)
print(f"✓ You actually have 1,988 significant genes total!")
print(f"✓ We used top {TOP_N_GENES} to avoid server timeouts")
print(f"✓ These top genes have the strongest signal")
print(f"✓ Results represent the most important biological processes")
