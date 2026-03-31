"""
DECONOMIX - WORKING IMPLEMENTATION
Using the actual DTD, ADTD, and HPS classes
"""

import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🧬 DECONOMIX - COMPLETE WORKING PIPELINE 🧬")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading skin atlas data...")
adata = sc.read_h5ad('Data/rna_data.h5ad')
adata_control = adata[adata.obs['disease'] == 'control'].copy()

# Filter to immune cells
immune_keywords = ['T cell', 'B cell', 'NK cell', 'Macrophage', 'Monocyte', 
                   'DC', 'Plasma cell', 'Mast cell']
immune_cell_types = [ct for ct in adata_control.obs['cell_type'].unique() 
                     if any(keyword in ct for keyword in immune_keywords)]
adata_immune = adata_control[adata_control.obs['cell_type'].isin(immune_cell_types)].copy()

# Gene filtering
min_cells = int(0.05 * adata_immune.n_obs)
sc.pp.filter_genes(adata_immune, min_cells=min_cells)
adata_immune.var['mt'] = adata_immune.var_names.str.startswith('MT-')
adata_immune.var['ribo'] = adata_immune.var_names.str.match(r'^RP[SL]')
adata_immune = adata_immune[:, ~(adata_immune.var['mt'] | adata_immune.var['ribo'])].copy()

print(f"Prepared: {adata_immune.n_obs:,} cells, {adata_immune.n_vars:,} genes")

cell_types = sorted(adata_immune.obs['cell_type'].unique())
gene_names = adata_immune.var_names.tolist()
print(f"Cell types: {len(cell_types)}")

# ============================================================================
# PREPARE DATA FOR DECONOMIX
# ============================================================================
print("\n[STEP 2] Preparing data for DeconomiX...")

# Convert to dense if needed
if hasattr(adata_immune.X, 'toarray'):
    sc_expression = adata_immune.X.toarray()
else:
    sc_expression = np.array(adata_immune.X)

cell_type_labels = adata_immune.obs['cell_type'].values

# Create reference signature matrix
reference = []
for ct in cell_types:
    ct_mask = cell_type_labels == ct
    ct_expression = sc_expression[ct_mask, :]
    mean_expr = ct_expression.mean(axis=0)
    reference.append(mean_expr)

reference_matrix = np.array(reference).T  # genes × cell_types
print(f"Reference matrix: {reference_matrix.shape}")

# ============================================================================
# SIMULATE PSEUDO-BULK DATA
# ============================================================================
print("\n[STEP 3] Simulating pseudo-bulk samples...")

def simulate_bulk_samples(adata, n_bulks=500, seed=42):
    """Simulate bulk RNA-seq samples"""
    np.random.seed(seed)
    bulks = []
    props = []
    
    cell_type_array = adata.obs['cell_type'].values
    cell_types_list = sorted(adata.obs['cell_type'].unique())
    
    print(f"Generating {n_bulks} pseudo-bulk samples...")
    for i in range(n_bulks):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_bulks}...")
        
        n_cells = np.random.randint(500, 2000)
        indices = np.random.choice(adata.n_obs, n_cells, replace=True)
        
        if hasattr(adata.X, 'toarray'):
            bulk = adata.X[indices].toarray().sum(axis=0)
        else:
            bulk = np.array(adata.X[indices].sum(axis=0))
        
        bulks.append(bulk.flatten())
        
        # True proportions
        sampled_cts = cell_type_array[indices]
        unique, counts = np.unique(sampled_cts, return_counts=True)
        prop_dict = dict(zip(unique, counts / n_cells))
        prop_vec = [prop_dict.get(ct, 0.0) for ct in cell_types_list]
        props.append(prop_vec)
    
    return np.array(bulks).T, np.array(props).T

train_bulks, train_props = simulate_bulk_samples(adata_immune, n_bulks=800, seed=42)
test_bulks, test_props = simulate_bulk_samples(adata_immune, n_bulks=300, seed=123)

print(f"\nTraining: {train_bulks.shape[1]} samples")
print(f"Test: {test_bulks.shape[1]} samples")

# ============================================================================
# METHOD 1: DTD (Digital Tissue Deconvolution)
# ============================================================================
print("\n" + "="*70)
print("METHOD 1: DTD (Digital Tissue Deconvolution)")
print("="*70)

from deconomix.methods import DTD
import torch

# 1. CAST TO FLOAT32
X_train = train_bulks.astype(np.float32)      # (Genes x Samples) -> (6846, 800)
Y_ref   = reference_matrix.astype(np.float32) # (Genes x CellTypes) -> (6846, 14)

# 2. TRANSPOSE C_TRAIN (THE FIX 🔧)
# The library checks if X_train columns == C_train rows.
# So we must provide C as (Samples x CellTypes)
C_train = train_props.T.astype(np.float32)    # (Samples x CellTypes) -> (800, 14)

print(f"Input Shapes:")
print(f"  X (Bulk):      {X_train.shape} (Cols=Samples)")
print(f"  Y (Reference): {Y_ref.shape}")
print(f"  C (Truth):     {C_train.shape} (Rows=Samples)")

try:
    print("\nInitializing DTD...")
    dtd_model = DTD(X_train, Y_ref, C_train)
    print("✓ DTD Initialized!")
    
    print("Training DTD...")
    dtd_model.run(num_epochs=500, lr=1e-3)
    print("✓ DTD Training Complete")
    
    # Predict
    dtd_model.Model.eval()
    X_test_tensor = torch.from_numpy(test_bulks.astype(np.float32))
    Y_ref_tensor = torch.from_numpy(Y_ref)
    
    with torch.no_grad():
        # Model returns (Samples x CellTypes)
        predicted_dtd = dtd_model.Model(X_test_tensor, Y_ref_tensor).numpy()
        
    # Transpose prediction back to (CellTypes x Samples) for evaluation code later
    predicted_dtd = predicted_dtd.T
    dtd_worked = True
    print(f"✓ DTD Prediction Shape: {predicted_dtd.shape}")

except Exception as e:
    print(f"❌ DTD Failed: {e}")
    dtd_worked = False

# ============================================================================
# METHOD 2: ADTD (Adaptive DTD)
# ============================================================================
print("\n" + "="*70)
print("METHOD 2: ADTD (Adaptive DTD)")
print("="*70)

from deconomix.methods import ADTD

try:
    # ADTD needs 'gamma' instead of C_mat
    gamma_val = 1.0 
    
    print(f"Initializing ADTD (gamma={gamma_val})...")
    adtd_model = ADTD(X_train, Y_ref, gamma_val) # No C_train here!
    print("✓ ADTD Initialized!")
    
    print("Training ADTD...")
    adtd_model.run(num_epochs=500, lr=1e-3)
    
    # Predict
    adtd_model.Model.eval()
    with torch.no_grad():
        predicted_adtd = adtd_model.Model(X_test_tensor, Y_ref_tensor).numpy()
    
    predicted_adtd = predicted_adtd.T
    adtd_worked = True
    print(f"✓ ADTD Prediction Shape: {predicted_adtd.shape}")

except Exception as e:
    print(f"❌ ADTD Failed: {e}")
    adtd_worked = False

# ============================================================================
# METHOD 3: HPS
# ============================================================================
print("\n" + "="*70)
print("METHOD 3: HPS")
print("="*70)

from deconomix.methods import HPS

try:
    # Based on DTD/ADTD logic, HPS likely needs X and Y + maybe alpha/beta?
    # Let's inspect briefly to avoid crash
    import inspect
    args = inspect.getfullargspec(HPS.__init__).args
    print(f"HPS Init Args: {args}")
    
    # Safe Initialization attempt
    if 'C_mat' in args:
        hps_model = HPS(X_train, Y_ref, C_train) # Needs Truth?
    elif 'gamma' in args:
        hps_model = HPS(X_train, Y_ref, 1.0)     # Needs Gamma?
    else:
        hps_model = HPS(X_train, Y_ref)          # Maybe just X, Y?
        
    print("✓ HPS Initialized!")
    hps_model.run(num_epochs=500, lr=1e-3)
    
    # Predict
    hps_model.Model.eval()
    with torch.no_grad():
        predicted_hps = hps_model.Model(X_test_tensor, Y_ref_tensor).numpy()
    
    predicted_hps = predicted_hps.T
    hps_worked = True
    print(f"✓ HPS Prediction Shape: {predicted_hps.shape}")

except Exception as e:
    print(f"❌ HPS Failed: {e}")
    hps_worked = False
    
# ============================================================================
# EVALUATE ALL METHODS THAT WORKED
# ============================================================================
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

results = {}

if dtd_worked:
    print("\n[Evaluating DTD...]")
    corrs_dtd = []
    for i in range(len(cell_types)):
        # Handle different output shapes
        if predicted_dtd.shape[0] == len(cell_types):
            pred = predicted_dtd[i, :]
        else:
            pred = predicted_dtd[:, i]
        
        corr, _ = spearmanr(test_props[i, :], pred)
        corrs_dtd.append(corr)
    
    avg_corr_dtd = np.mean(corrs_dtd)
    results['DTD'] = {'correlations': corrs_dtd, 'avg': avg_corr_dtd, 'predictions': predicted_dtd}
    print(f"  Average correlation: {avg_corr_dtd:.3f}")

if adtd_worked:
    print("\n[Evaluating ADTD...]")
    corrs_adtd = []
    for i in range(len(cell_types)):
        if predicted_adtd.shape[0] == len(cell_types):
            pred = predicted_adtd[i, :]
        else:
            pred = predicted_adtd[:, i]
        
        corr, _ = spearmanr(test_props[i, :], pred)
        corrs_adtd.append(corr)
    
    avg_corr_adtd = np.mean(corrs_adtd)
    results['ADTD'] = {'correlations': corrs_adtd, 'avg': avg_corr_adtd, 'predictions': predicted_adtd}
    print(f"  Average correlation: {avg_corr_adtd:.3f}")

if hps_worked:
    print("\n[Evaluating HPS...]")
    corrs_hps = []
    for i in range(len(cell_types)):
        if predicted_hps.shape[0] == len(cell_types):
            pred = predicted_hps[i, :]
        else:
            pred = predicted_hps[:, i]
        
        corr, _ = spearmanr(test_props[i, :], pred)
        corrs_hps.append(corr)
    
    avg_corr_hps = np.mean(corrs_hps)
    results['HPS'] = {'correlations': corrs_hps, 'avg': avg_corr_hps, 'predictions': predicted_hps}
    print(f"  Average correlation: {avg_corr_hps:.3f}")

# ============================================================================
# SELECT BEST METHOD AND SAVE
# ============================================================================
if results:
    print("\n" + "="*70)
    print("🏆 RESULTS SUMMARY")
    print("="*70)
    
    for method_name, data in sorted(results.items(), key=lambda x: x[1]['avg'], reverse=True):
        print(f"{method_name:10s} Average Correlation: {data['avg']:.3f}")
    
    best_method = max(results, key=lambda x: results[x]['avg'])
    best_data = results[best_method]
    
    print(f"\n🎯 Best Method: {best_method} ({best_data['avg']:.3f})")
    
    # Save
    np.save('deconomix_reference_final.npy', reference_matrix)
    np.save(f'deconomix_{best_method.lower()}_predictions.npy', best_data['predictions'])
    
    with open('deconomix_cell_types_final.txt', 'w') as f:
        for ct in cell_types:
            f.write(f"{ct}\n")
    
    print("\n✅ Model saved!")
    print(f"\n🎉 DeconomiX training complete!")
    print(f"   Ready for Part 3: Apply to TCGA melanoma data!")
    
else:
    print("\n❌ All methods failed!")
    print("Let me check the actual DTD source code...")