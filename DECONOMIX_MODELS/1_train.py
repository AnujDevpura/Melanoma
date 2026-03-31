"""
02_train_optimized.py
FINAL OPTIMIZED MANUAL DTD
- Uses MinMax Scaling (Better for ReLU networks)
- Increases Model Capacity (512 neurons)
- Increases Epochs to 1000
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

print("="*60)
print("🚀 STEP 2: OPTIMIZED DEEP LEARNING (FINAL RUN)")
print("="*60)

# ============================================================================
# 1. LOAD DATA & SCALE
# ============================================================================
print("\n[1/4] Loading & Scaling Data...")
try:
    X_train = np.load('Processed_Data/X_train.npy').astype(np.float32)
    C_train = np.load('Processed_Data/C_train.npy').astype(np.float32) 
    X_test  = np.load('Processed_Data/X_test.npy').astype(np.float32)
    C_test  = np.load('Processed_Data/C_test.npy')
    cell_types = np.load('Processed_Data/cell_types.npy')
    
    C_train = C_train.T 
    
    # 1. LOG TRANSFORM
    X_train = np.log1p(X_train).T 
    X_test  = np.log1p(X_test).T  
    
    # 2. MIN-MAX SCALING (0 to 1) - CRITICAL CHANGE 🔧
    # This keeps zeros as zeros, which helps the network learn sparsity.
    print("      Applying MinMax Scaling (0-1)...")
    
    # Calculate Max per gene across samples
    max_vals = X_train.max(axis=0) + 1e-6
    
    X_train = X_train / max_vals
    X_test  = X_test / max_vals
    
    print(f"      Data Shapes: X={X_train.shape}, C={C_train.shape}")
    
except FileNotFoundError:
    print("❌ ERROR: Run '01_prepare_data.py' first.")
    exit()

# ============================================================================
# 2. DEFINE BIGGER MODEL 🧠
# ============================================================================
class ManualDTD(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ManualDTD, self).__init__()
        self.net = nn.Sequential(
            # Layer 1: Wider (512 neurons) to capture more gene interactions
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3), # Increased dropout slightly
            
            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Output
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1) 
        )
        
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ManualDTD(X_train.shape[1], C_train.shape[1]).to(device)

# ============================================================================
# 3. TRAINING LOOP
# ============================================================================
print("\n🔵 Starting Training (1000 Epochs)...")

LR = 0.0005          # Slightly faster learning rate
EPOCHS = 1000        # Give it time to converge
BATCH_SIZE = 64      # Larger batches for stability

train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(C_train))
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    # Print progress every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f"      Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.6f}")

print("      ✓ Training Complete!")

# ============================================================================
# 4. EVALUATION
# ============================================================================
print("\n[Evaluation] Running Inference...")
model.eval()

X_test_tensor = torch.from_numpy(X_test).to(device)
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy()

corrs = []
print("\n📊 PER-CELL TYPE CORRELATIONS:")
print("-" * 40)
for i, ct in enumerate(cell_types):
    true_vals = C_test[i, :]     # (Samples,)
    pred_vals = predictions[:, i] # (Samples,)
    
    corr = spearmanr(true_vals, pred_vals)[0]
    corrs.append(corr)
    print(f"{ct:20s}: {corr:.3f}")

avg_corr = np.mean(corrs)
print("-" * 40)
print(f"🏆 AVERAGE DTD CORRELATION: {avg_corr:.4f}")

# ============================================================================
# DECISION LOGIC
# ============================================================================
if avg_corr > 0.4:
    print("\n✅ SUCCESS! Deep Learning worked.")
    np.save('final_model_predictions.npy', predictions.T)
    torch.save(model.state_dict(), 'final_model.pt')
else:
    print("\n⚠️ RESULT: Deep Learning is unstable on this dataset.")
    print("👉 ACTION: Use your previous LASSO results (0.393) for Part 3.")
    print("   Science isn't about forcing a method; it's about reporting what works.")

print("\n" + "="*60)