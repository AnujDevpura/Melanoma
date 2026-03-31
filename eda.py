"""
Script to explore TCGA-SKCM data and prepare for differential expression analysis
This matches samples between count matrix and clinical data
"""

from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 60)
print("EXPLORING TCGA-SKCM DATA")
print("=" * 60)

data_dir = Path("tcga_skcm_data")

# =============================================================================
# Load the data
# =============================================================================
print("\n[1] Loading data files...")

counts = pd.read_csv(data_dir / "count_matrix.csv", index_col=0)
clinical = pd.read_csv(data_dir / "clinical_data.csv")

print(f"✓ Count matrix: {counts.shape[0]} genes × {counts.shape[1]} samples")
print(f"✓ Clinical data: {clinical.shape[0]} patients × {clinical.shape[1]} features")

# =============================================================================
# Check sample overlap
# =============================================================================
print("\n[2] Checking sample overlap...")

# Get sample IDs from both datasets
count_samples = set(counts.columns)
clinical_samples = set(clinical["submitter_id"])

# Find overlap
overlap = count_samples & clinical_samples
only_counts = count_samples - clinical_samples
only_clinical = clinical_samples - count_samples

print(f"\nSamples in count matrix: {len(count_samples)}")
print(f"Samples in clinical data: {len(clinical_samples)}")
print(f"Samples in BOTH: {len(overlap)}")
print(f"Only in counts: {len(only_counts)}")
print(f"Only in clinical: {len(only_clinical)}")

# =============================================================================
# Explore clinical variables
# =============================================================================
print("\n[3] Exploring clinical variables...")

print("\n--- Vital Status ---")
vital_counts = clinical["vital_status"].value_counts()
print(vital_counts)

print("\n--- Gender ---")
gender_counts = clinical["gender"].value_counts()
print(gender_counts)

print("\n--- Tumor Stage ---")
stage_counts = clinical["tumor_stage"].value_counts()
print(stage_counts)
print(f"Missing tumor stage: {clinical['tumor_stage'].isna().sum()}")

print("\n--- Progression/Recurrence ---")
prog_counts = clinical["progression_or_recurrence"].value_counts()
print(prog_counts)
print(f"Missing progression data: {clinical['progression_or_recurrence'].isna().sum()}")

# =============================================================================
# Survival time statistics
# =============================================================================
print("\n[4] Survival time statistics...")

# For dead patients
dead_patients = clinical[clinical["vital_status"] == "Dead"]
if len(dead_patients) > 0:
    days_to_death = dead_patients["days_to_death"].dropna()
    print(f"\nDead patients with days_to_death: {len(days_to_death)}")
    print(f"Survival time (days):")
    print(f"  - Mean: {days_to_death.mean():.0f}")
    print(f"  - Median: {days_to_death.median():.0f}")
    print(f"  - Min: {days_to_death.min():.0f}")
    print(f"  - Max: {days_to_death.max():.0f}")
    print(f"  - In years (median): {days_to_death.median() / 365:.1f}")

# For alive patients
alive_patients = clinical[clinical["vital_status"] == "Alive"]
if len(alive_patients) > 0:
    days_to_followup = alive_patients["days_to_last_follow_up"].dropna()
    print(f"\nAlive patients with follow-up: {len(days_to_followup)}")
    print(f"Follow-up time (days):")
    print(f"  - Mean: {days_to_followup.mean():.0f}")
    print(f"  - Median: {days_to_followup.median():.0f}")
    print(f"  - In years (median): {days_to_followup.median() / 365:.1f}")

# =============================================================================
# Create matched dataset
# =============================================================================
print("\n[5] Creating matched dataset...")

# Keep only samples that have both count and clinical data
matched_samples = list(overlap)
matched_samples.sort()  # Keep consistent order

# Filter count matrix
counts_matched = counts[matched_samples]

# Filter clinical data
clinical_matched = clinical[clinical["submitter_id"].isin(matched_samples)]
# Sort to match count matrix order
clinical_matched = (
    clinical_matched.set_index("submitter_id").loc[matched_samples].reset_index()
)

print(f"✓ Matched dataset: {len(matched_samples)} samples")

# Save matched datasets
counts_matched.to_csv(data_dir / "count_matrix_matched.csv")
clinical_matched.to_csv(data_dir / "clinical_data_matched.csv", index=False)

print(f"✓ Saved: count_matrix_matched.csv")
print(f"✓ Saved: clinical_data_matched.csv")

# =============================================================================
# Summary for each comparison type
# =============================================================================
print("\n[6] Available comparisons for Task a)...")

comparisons = []

# Vital Status comparison
alive_n = (clinical_matched["vital_status"] == "Alive").sum()
dead_n = (clinical_matched["vital_status"] == "Dead").sum()
if alive_n > 0 and dead_n > 0:
    comparisons.append(
        {
            "comparison": "Vital Status",
            "groups": f"Alive ({alive_n}) vs Dead ({dead_n})",
            "column": "vital_status",
            "feasible": "Yes ✓",
        }
    )

# Gender comparison
males = (clinical_matched["gender"] == "male").sum()
females = (clinical_matched["gender"] == "female").sum()
if males > 0 and females > 0:
    comparisons.append(
        {
            "comparison": "Gender",
            "groups": f"Male ({males}) vs Female ({females})",
            "column": "gender",
            "feasible": "Yes ✓",
        }
    )

# Tumor Stage comparison
# Simplify stages to Early vs Late
clinical_matched["stage_simple"] = clinical_matched["tumor_stage"].apply(
    lambda x: "Early"
    if isinstance(x, str)
    and ("i" in x.lower() or "ii" in x.lower())
    and "iii" not in x.lower()
    and "iv" not in x.lower()
    else (
        "Late"
        if isinstance(x, str) and ("iii" in x.lower() or "iv" in x.lower())
        else None
    )
)
early = (clinical_matched["stage_simple"] == "Early").sum()
late = (clinical_matched["stage_simple"] == "Late").sum()
if early > 0 and late > 0:
    comparisons.append(
        {
            "comparison": "Tumor Stage",
            "groups": f"Early ({early}) vs Late ({late})",
            "column": "stage_simple",
            "feasible": "Yes ✓",
        }
    )

# Survival time comparison (for dead patients only)
dead_with_time = clinical_matched[
    (clinical_matched["vital_status"] == "Dead")
    & (clinical_matched["days_to_death"].notna())
]
if len(dead_with_time) > 10:  # Need enough samples
    median_survival = dead_with_time["days_to_death"].median()
    short = (dead_with_time["days_to_death"] < median_survival).sum()
    long = (dead_with_time["days_to_death"] >= median_survival).sum()
    comparisons.append(
        {
            "comparison": "Survival Time",
            "groups": f"Short survival ({short}) vs Long survival ({long})",
            "column": "survival_group",
            "feasible": "Yes ✓",
        }
    )
    # Add this column
    clinical_matched["survival_group"] = None
    clinical_matched.loc[dead_with_time.index, "survival_group"] = dead_with_time[
        "days_to_death"
    ].apply(lambda x: "Short" if x < median_survival else "Long")

# Progression/Recurrence
prog_yes = (clinical_matched["progression_or_recurrence"] == "yes").sum()
prog_no = (clinical_matched["progression_or_recurrence"] == "no").sum()
if prog_yes > 0 and prog_no > 0:
    comparisons.append(
        {
            "comparison": "Progression/Recurrence",
            "groups": f"Yes ({prog_yes}) vs No ({prog_no})",
            "column": "progression_or_recurrence",
            "feasible": "Yes ✓",
        }
    )

# Print comparison table
comparison_df = pd.DataFrame(comparisons)
print("\n" + "=" * 80)
print("AVAILABLE COMPARISONS FOR DIFFERENTIAL EXPRESSION")
print("=" * 80)
print(comparison_df.to_string(index=False))

# Save updated clinical data with computed columns
clinical_matched.to_csv(data_dir / "clinical_data_matched.csv", index=False)

# =============================================================================
# Quick data quality checks
# =============================================================================
print("\n[7] Data quality checks...")

# Check for low count genes (might want to filter)
gene_totals = counts_matched.sum(axis=1)
low_count_genes = (gene_totals < 10).sum()
print(f"\nGenes with <10 total counts: {low_count_genes} / {len(gene_totals)}")
print(
    f"({low_count_genes / len(gene_totals) * 100:.1f}% - these might be filtered out)"
)

# Check for samples with very different sequencing depth
sample_totals = counts_matched.sum(axis=0)
print(f"\nSequencing depth (total counts per sample):")
print(f"  - Mean: {sample_totals.mean():,.0f}")
print(f"  - Median: {sample_totals.median():,.0f}")
print(f"  - Min: {sample_totals.min():,.0f}")
print(f"  - Max: {sample_totals.max():,.0f}")
print(f"  - Range: {sample_totals.max() / sample_totals.min():.1f}x difference")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("READY FOR DIFFERENTIAL EXPRESSION!")
print("=" * 60)
print(f"\nYour matched dataset has:")
print(f"  - {counts_matched.shape[0]} genes")
print(f"  - {counts_matched.shape[1]} samples")
print(f"  - {len(comparisons)} possible comparisons")
print(f"\nRecommended first comparison: Alive vs Dead")
print(f"  - Clear biological question")
print(f"  - Good sample sizes")
print(f"  - Directly relates to survival")
print(f"\nFiles created:")
print(f"  - count_matrix_matched.csv")
print(f"  - clinical_data_matched.csv")
print(f"\nNext step: Run differential expression analysis with PyDESeq2!")
