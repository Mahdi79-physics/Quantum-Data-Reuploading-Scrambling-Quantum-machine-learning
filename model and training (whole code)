#Data Reuploading

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mendeleev import element

# Sklearn & Imbalance Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN  # <--- NEW IMPORT

# PennyLane Imports
import pennylane as qml
from pennylane import numpy as pnp

# ==========================================
# 1. Data Loading & Feature Engineering
# ==========================================

print("Loading and Processing Data...")
df = pd.read_csv('TableS1.csv')

# Encode Labels (-1 -> -1.0, 1 -> 1.0)
df['encoded_label'] = df['exp_label'].apply(lambda x: -1.0 if x == -1 else 1.0)

# Feature Calculations
df['rA_rX'] = df['rA (Ang)'] / df['rX (Ang)']
df['rB_rX'] = df['rB (Ang)'] / df['rX (Ang)']
df['t'] = df['t']
df['tau'] = df['tau']
df['d_AX'] = df['rA (Ang)'] + df['rX (Ang)']

# Element Lookup
unique_elements = set(df['A'].unique()) | set(df['B'].unique()) | set(df['X'].unique())
elem_lookup = {}
for symbol in unique_elements:
    try:
        el = element(symbol)
        elem_lookup[symbol] = (el.electronegativity(), el.atomic_number)
    except:
        elem_lookup[symbol] = (0.0, 0.0)

df['chi_B'] = df['B'].map(lambda x: elem_lookup.get(x, (0.0, 0.0))[0])
df['chi_X'] = df['X'].map(lambda x: elem_lookup.get(x, (0.0, 0.0))[0])
df['delta_chi_BX'] = abs(df['chi_B'] - df['chi_X'])

# Select 5 Features
features = ['tau', 't', 'rA_rX', 'rB_rX', 'delta_chi_BX']
df_clean = df.dropna(subset=features).copy()
X_raw = df_clean[features].values
y_raw = df_clean['encoded_label'].values

# Split Data First (To avoid data leakage during SMOTE)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# ==========================================
# 2. SMOTEENN Resampling (Balancing Data)
# ==========================================

print(f"Original Train Count: {len(X_train_raw)}")
print(f"Original Class Dist: {np.unique(y_train_raw, return_counts=True)[1]}")

print("Applying SMOTEENN...")
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_raw, y_train_raw)

print(f"Resampled Train Count: {len(X_train_resampled)}")
print(f"Resampled Class Dist: {np.unique(y_train_resampled, return_counts=True)[1]}")

# ==========================================
# 3. Scaling & Conversion
# ==========================================

# Scale to [0, pi]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_scaled = scaler.fit_transform(X_train_resampled)
# Important: Scale Test data using the scaler fitted on Train data
X_test_scaled = scaler.transform(X_test_raw)

# Convert to PennyLane-compatible Numpy arrays
X_train = pnp.array(X_train_scaled, requires_grad=False)
y_train = pnp.array(y_train_resampled, requires_grad=False)
X_test = pnp.array(X_test_scaled, requires_grad=False)
y_test = pnp.array(y_test_raw, requires_grad=False)

# ==========================================
# 4. Define Scrambling Architecture
# ==========================================

num_qubits = 5
num_layers = 4  # Depth of the model
dev = qml.device("default.qubit", wires=num_qubits)

def TwoQubitUnitary(params, w1, w2):
    """
    Local 2-qubit block: Ry - Ry - CNOT - Ry - Ry - CNOT
    Uses 4 parameters.
    """
    qml.RY(params[0], wires=w1)
    qml.RY(params[1], wires=w2)
    qml.CNOT(wires=[w1, w2])
    qml.RY(params[2], wires=w1)
    qml.RY(params[3], wires=w2)
    qml.CNOT(wires=[w2, w1]) # Reverse CNOT for better mixing

def ScramblingAnsatzLayer(params, wires):
    """
    Brick-wall layout for 5 qubits.
    Blocks: (0,1), (2,3), (1,2), (3,4), (4,0)
    Total blocks = 5. Params per block = 4.
    Total params per layer = 20.
    """
    n = len(wires)
    param_idx = 0
    
    # 1. Even Pairs: (0,1), (2,3)
    for i in range(0, n - 1, 2):
        TwoQubitUnitary(params[param_idx : param_idx + 4], wires[i], wires[i+1])
        param_idx += 4
        
    # 2. Odd Pairs: (1,2), (3,4)
    for i in range(1, n - 1, 2):
        TwoQubitUnitary(params[param_idx : param_idx + 4], wires[i], wires[i+1])
        param_idx += 4
        
    # 3. Boundary Pair: (4,0) - To close the loop
    TwoQubitUnitary(params[param_idx : param_idx + 4], wires[n-1], wires[0])

def DataEncoding(x):
    """Angle Encoding"""
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        qml.RZ(2 * x[i], wires=i)

# Using 'adjoint' differentiation for high speed
@qml.qnode(dev, interface="autograd", diff_method="adjoint")
def circuit(params, x):
    # Data Re-uploading: Encode -> Process -> Encode -> Process...
    for l in range(params.shape[0]):
        DataEncoding(x)
        ScramblingAnsatzLayer(params[l], range(num_qubits))
        
    return qml.expval(qml.PauliZ(0))

# ==========================================
# 5. Training Setup
# ==========================================

# 1. Margin/Hinge Loss (More robust for classification than MSE)
def cost_fn(params, x, y):
    predictions = [circuit(params, x_) for x_ in x]
    predictions = pnp.stack(predictions)
    # Penalize if (y * pred) < 1
    margin = 1 - y * predictions
    loss = pnp.mean(pnp.maximum(0, margin) ** 2)
    return loss

# 2. Parameters
# Shape: (Layers, Params_Per_Layer) = (3, 20)
params_per_layer = 20
total_params_shape = (num_layers, params_per_layer)

np.random.seed(42)
params = pnp.random.uniform(-0.1, 0.1, total_params_shape, requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.02)
epochs = 30
batch_size = 64

print(f"\nStarting Training (Adjoint Diff + SMOTEENN)...")
print(f"Total Params: {params.size}")
print("-" * 65)
print(f"{'Epoch':<6} | {'Loss':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
print("-" * 65)

history = {'loss': [], 'train_acc': [], 'test_acc': []}

for epoch in range(epochs):
    
    # Shuffle Batch
    permutation = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]
    
    # Optimization Loop (Mini-batches)
    epoch_losses = []
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_shuffled[i : i + batch_size]
        y_batch = y_train_shuffled[i : i + batch_size]
        
        # Gradient Step
        params, current_loss = opt.step_and_cost(lambda p: cost_fn(p, X_batch, y_batch), params)
        epoch_losses.append(current_loss)
    
    avg_loss = np.mean(epoch_losses)
    
    # Evaluation
    # Note: Predicting on full set might be slightly slow on simulator, 
    # but 'adjoint' method makes it manageable.
    
    # Train Acc
    train_preds_raw = [circuit(params, x) for x in X_train]
    train_preds = [1.0 if p >= 0 else -1.0 for p in train_preds_raw]
    train_acc = accuracy_score(y_train, train_preds)
    
    # Test Acc
    test_preds_raw = [circuit(params, x) for x in X_test]
    test_preds = [1.0 if p >= 0 else -1.0 for p in test_preds_raw]
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"{epoch+1:<6} | {avg_loss:.4f}     | {train_acc:.4f}     | {test_acc:.4f}")
    
    history['loss'].append(avg_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)

print("\nTraining Complete.")

# ==========================================
# 6. Visualization & Save
# ==========================================

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Margin Loss', color='purple')
plt.title('Training Loss (SMOTEENN + Scrambling)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', color='blue')
plt.plot(history['test_acc'], label='Test Acc', color='green')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Save Results
results_df = pd.DataFrame(X_test, columns=[f"Feat_{i}" for i in range(num_qubits)])
results_df['Actual'] = y_test
results_df['Predicted'] = test_preds
results_df.to_csv("SMOTE_Scrambling_Results.csv", index=False)
print("Saved results to 'SMOTE_Scrambling_Results.csv'")
