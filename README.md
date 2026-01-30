## 🧬 Data Re-Uploading Quantum Classifier with Scrambling Ansatz  
**SMOTEENN-Balanced Quantum Machine Learning for Materials Data**

This repository implements a **variational quantum machine learning (QML) classifier** based on **data re-uploading**, **scrambling-inspired entangling layers**, and **margin-based optimization**, applied to materials classification data.

The model combines **classical data balancing**, **feature engineering from atomic properties**, and a **multi-layer quantum circuit** trained using **adjoint differentiation** in PennyLane.

---

## 🧠 Core Idea

This project explores how **data re-uploading quantum circuits** enhance expressive power in variational quantum classifiers by repeatedly encoding classical data between entangling layers.

**Key ingredients:**

- 🔁 **Data re-uploading** for increased nonlinearity  
- 🧩 **Scrambling entangling ansatz** for strong qubit mixing  
- ⚖️ **SMOTEENN resampling** to handle class imbalance  
- 📐 **Margin (hinge) loss** for robust classification  
- ⚡ **Adjoint differentiation** for efficient gradient evaluation  

---

## 📂 Repository Structure

```text
Data-Reuploading-QML/
│
├── README.md
├── requirements.txt
├── TableS1.csv
│
├── model/
│   └── data_reuploading_scrambling.py
│
├── figures/
│   ├── training_loss.png
│   └── accuracy.png
│
└── results/
    └── SMOTE_Scrambling_Results.csv
---

## 📊 Dataset & Feature Engineering

The dataset used in this project is loaded from `TableS1.csv` and contains experimentally and computationally derived materials descriptors. Prior to training, the data undergoes **feature selection, physical normalization, and class balancing** to ensure stable quantum training.

---

### 🔍 Selected Features

Five physically motivated features are extracted and used as inputs to the quantum classifier:

| Feature | Description |
|--------|-------------|
| τ | Structural tolerance parameter |
| t | Goldschmidt tolerance factor |
| r<sub>A</sub> / r<sub>X</sub> | Ionic radius ratio between A-site cation and halide |
| r<sub>B</sub> / r<sub>X</sub> | Ionic radius ratio between B-site cation and halide |
| Δχ(B–X) | Electronegativity difference between B-site cation and halide |

These features are chosen to capture **geometric stability**, **ionic size mismatch**, and **bond polarity**, which are critical for materials classification tasks.

---

### 🧪 Elemental Property Extraction

Element-specific quantities are automatically retrieved using the **Mendeleev** library, including:

- Pauling electronegativity  
- Ionic and atomic radii  

This allows the feature construction pipeline to remain **generalizable** to unseen chemical compositions.

---

### ⚙️ Preprocessing Pipeline

The preprocessing workflow consists of:

1. Removal of incomplete or non-physical entries  
2. Feature normalization to zero mean and unit variance  
3. Scaling of all features to the interval  
   \[
   [0, \pi]
   \]
   for quantum angle encoding  
4. Stratified train–test split to preserve class ratios  

---

### ⚖️ Class Imbalance Handling

To mitigate strong class imbalance in the dataset, the training set is resampled using **SMOTEENN**, which combines:

- **SMOTE** — Synthetic Minority Over-sampling  
- **ENN** — Edited Nearest Neighbors cleaning  

> **Important:** Resampling is applied **only to the training set** to prevent information leakage into the test data.

---

### 🎯 Final Input Representation

Each material sample is represented as a **5-dimensional feature vector**:

\[
\mathbf{x} = (x_1, x_2, x_3, x_4, x_5)
\]

These vectors are used directly as inputs to the **data re-uploading quantum circuit**.

---
## 🔀 Quantum Circuit Architecture

The quantum model is implemented as a **variational data re-uploading circuit** with strongly entangling scrambling layers, designed to maximize expressivity under a limited qubit budget.

---

### 🖥️ Hardware Setup

- **Number of qubits:** 5  
- **Quantum framework:** PennyLane  
- **Device:** `default.qubit` (statevector simulator)  
- **Differentiation method:** Adjoint differentiation  

---

### 🧩 Overall Circuit Structure

The circuit follows a **layered re-uploading architecture**:


Each classical input vector is **re-encoded at every layer**, allowing the variational circuit to build high-order nonlinear decision boundaries.

---

### 🔁 Data Re-Uploading Layers

At each layer, the classical feature vector  
\[
\mathbf{x} \in \mathbb{R}^5
\]
is encoded using **angle encoding**:

- Global Hadamard initialization
- Feature-dependent phase rotations:
  \[
  RZ(2x_i)
  \]
  applied to qubit *i*

This repeated encoding significantly increases the expressive power of shallow quantum circuits.

---

### 🧱 Scrambling Entangling Ansatz

Between data encoding stages, a **scrambling entangling layer** is applied to promote strong qubit mixing and correlation.

#### 🔗 Entanglement Layout (Brick-Wall)

Entangling blocks are applied sequentially on the following qubit pairs:

- (0, 1)
- (2, 3)
- (1, 2)
- (3, 4)
- (4, 0)

This layout ensures **full connectivity** across the register over a single layer.

---

### 🔧 Entangling Block Structure

Each two-qubit block consists of:

1. Local rotations:  
   \[
   RY(\theta_1) \otimes RY(\theta_2)
   \]
2. Controlled-NOT (CNOT)
3. Local rotations:  
   \[
   RY(\theta_3) \otimes RY(\theta_4)
   \]
4. Reverse CNOT

- **Parameters per block:** 4  
- **Blocks per layer:** 5  
- **Total parameters per layer:** 20  

---

### 🔄 Layer Repetition

The full variational circuit applies multiple repetitions of:


This **Encode–Entangle–Encode** pattern allows the model to represent complex, highly nonlinear decision functions.

---

### 📏 Parameter Scaling

Let:
- \( L \) = number of re-uploading layers  

Then the total number of trainable parameters is:

\[
N_{\text{params}} = 20 \times L
\]

This linear scaling enables controlled expressivity without excessive parameter growth.

---

### 🧠 Design Motivation

This architecture is chosen to:

- Avoid barren plateaus via shallow but expressive layers  
- Maximize entanglement efficiency  
- Support structured inductive bias for scientific data  
- Remain compatible with near-term quantum hardware  

---

### 🧪 Summary

| Component | Choice |
|---------|-------|
| Encoding | Angle encoding (RZ) |
| Entanglement | Scrambling brick-wall |
| Expressivity | Data re-uploading |
| Readout | Single-qubit expectation |
| Scalability | Linear in layer count |
---
## 📐 Measurement & Output

The quantum classifier produces a **single scalar output** obtained from a projective measurement on one qubit.
---
### 🔍 Observable

The model measures the expectation value of the Pauli-Z operator on qubit 0:

\[
\langle Z_0 \rangle
\]

This value lies in the interval:

\[
\langle Z_0 \rangle \in [-1, 1]
\]

---

### 🧮 Classification Rule

The expectation value is mapped to a binary class label using a threshold decision rule:

- **\(\langle Z_0 \rangle \ge 0\)** → Class **+1**
- **\(\langle Z_0 \rangle < 0\)** → Class **−1**

This simple readout:
- Minimizes measurement overhead
- Is compatible with near-term hardware
- Aligns naturally with margin-based losses

---

### 📤 Output Interpretation

The output expectation value can be interpreted as:
- A signed confidence score
- A soft decision boundary indicator
- A margin proxy for classification robustness

---

## 🎯 Loss Function

Training is performed using a **margin-based (hinge) loss**, commonly used in large-margin classifiers.

### 📉 Loss Definition

For a true label \( y \in \{-1, +1\} \) and model prediction \( \hat{y} = \langle Z_0 \rangle \), the loss is:

\[
\mathcal{L}
=
\mathbb{E}
\left[
\max\left(0,\; 1 - y \cdot \hat{y}\right)^2
\right]
\]

---

### ✅ Why Margin Loss?

The squared hinge loss is chosen because it:

- Encourages **confident predictions**
- Penalizes misclassified samples strongly
- Is more stable than mean-squared error (MSE)
- Works naturally with expectation-value outputs

---

### 🧠 Optimization Objective

The training objective is to **maximize the classification margin** while minimizing misclassification error across the dataset.

---

## ⚙️ Training Details

### 🧪 Optimization Setup

- **Optimizer:** Adam  
- **Learning rate:** 0.02  
- **Epochs:** 30  
- **Batch size:** 64  

---

### ⚡ Differentiation Strategy

Gradients are computed using:

- **Adjoint differentiation**

This method:
- Computes exact gradients
- Scales efficiently with circuit depth
- Avoids sampling noise from finite shots

---

### 📊 Metrics Tracked

During training, the following metrics are recorded:

- Training loss
- Training accuracy
- Test accuracy

These metrics are used to evaluate convergence, generalization, and overfitting behavior.

---

### 🔁 Training Workflow

1. Initialize circuit parameters randomly  
2. Encode classical data into the quantum circuit  
3. Measure expectation values  
4. Compute hinge loss  
5. Update parameters via gradient descent  
6. Repeat for all epochs  

---

### 🧠 Stability Considerations

To ensure reliable training:

- Data re-uploading mitigates barren plateaus  
- Strong entangling layers improve gradient flow  
- Margin loss prevents vanishing gradients  

---

### 📌 Summary

| Component | Choice |
|--------|--------|
| Observable | \( \langle Z_0 \rangle \) |
| Output type | Expectation value |
| Loss | Squared hinge loss |
| Optimizer | Adam |
| Gradients | Adjoint method |
| Target task | Binary classification |

## 📈 Results & Outputs

The training and evaluation process produces both **visual diagnostics** and **structured numerical outputs** to assess model performance and stability.

---

### 📉 Training Loss

- The loss decreases steadily across epochs, indicating stable optimization.
- Margin-based loss ensures confident separation between classes.
- No abrupt oscillations are observed, reflecting good gradient behavior.

**Figure:**
- `figures/training_loss.png`

---

### 📈 Classification Accuracy

Accuracy is tracked throughout training for both training and test sets.

- Training accuracy improves monotonically.
- Test accuracy follows closely, indicating good generalization.
- No severe overfitting is observed despite strong circuit expressivity.

**Figures:**
- `figures/accuracy.png`

---

### 📁 Prediction Outputs

Final predictions are saved in a structured CSV file:


Each row contains:
- Encoded feature values
- Ground-truth class label
- Predicted expectation value ⟨Z₀⟩
- Final predicted class (+1 / −1)

This format enables:
- Post-hoc statistical analysis
- Benchmarking against classical models
- Reproducible evaluation pipelines

---

### 📊 Summary of Outputs

| Output Type | Description |
|-----------|------------|
| Training loss | Optimization convergence |
| Accuracy curves | Generalization performance |
| CSV results | Sample-level predictions |
| Figures | Visual diagnostics |

---

## 🧪 Why This Model Matters

This project demonstrates a **practical, research-grade quantum machine learning pipeline** rather than a toy example.

### 🔬 Scientific Relevance

- Applies QML to **real materials data**
- Handles **severe class imbalance** correctly
- Uses domain-aware feature engineering

---

### ⚛️ Quantum Machine Learning Contributions

- Demonstrates **data re-uploading** as a solution to limited circuit expressivity
- Introduces **scrambling entangling layers** for strong qubit mixing
- Uses **adjoint differentiation**, enabling scalable training

---

### 📐 Methodological Strengths

- Margin-based loss improves robustness
- Single-observable readout minimizes measurement cost
- Compatible with near-term quantum hardware

---

### 🧠 Broader Impact

This model serves as:
- A benchmark variational quantum classifier
- A template for scientific QML workflows
- A bridge between quantum algorithms and materials informatics

---

## 🚀 Future Extensions

Several natural extensions can further enhance this work:

---

### ⚙️ Quantum Enhancements

- Add realistic noise models using `default.mixed`
- Shot-based training to mimic hardware conditions
- Multi-qubit observables for richer readout

---

### 📊 Model Improvements

- Hyperparameter optimization (depth, learning rate)
- Alternative loss functions (logistic, focal loss)
- Adaptive data re-uploading strategies

---

### 🧪 Benchmarking & Validation

- Compare against classical ML baselines (SVM, XGBoost, NN)
- Perform cross-validation and statistical tests
- Evaluate robustness under noisy labels

---

### 🖥️ Hardware Execution

- Deploy on cloud quantum hardware
- Analyze hardware-induced bias and noise
- Compare simulation vs hardware performance

---

### 📌 Outlook

This repository provides a strong foundation for:
- Research-grade QML experimentation
- Materials discovery pipelines
- Hybrid quantum–classical learning systems

