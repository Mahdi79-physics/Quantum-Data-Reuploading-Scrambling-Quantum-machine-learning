##🧬 Data Re-Uploading Quantum Classifier with Scrambling Ansatz
SMOTEENN-Balanced Quantum Machine Learning for Materials Data

This repository implements a variational quantum machine learning (QML) classifier based on data re-uploading, entangling scrambling layers, and margin-based optimization, applied to materials classification data.

The model combines classical data balancing, feature engineering from atomic properties, and a multi-layer quantum circuit trained using adjoint differentiation in PennyLane.

##🧠 Core Idea

This project explores how data re-uploading quantum circuits can enhance expressive power in variational quantum classifiers by repeatedly encoding classical data between entangling layers.

Key ingredients:

#🔁 Data re-uploading for increased nonlinearity

#🧩 Scrambling entangling ansatz for strong qubit mixing

#⚖️ SMOTEENN resampling to handle class imbalance

#📐 Margin (hinge) loss for robust classification

#⚡ Adjoint differentiation for efficient gradient evaluation

#📂 Repository Structure
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
│   ├── accuracy.png
│
└── results/
    └── SMOTE_Scrambling_Results.csv

##📊 Dataset & Feature Engineering

The dataset is loaded from TableS1.csv and processed using domain-aware feature construction:

Selected Features (5)
Feature	Description
τ	Structural tolerance parameter
t	Goldschmidt tolerance factor
rA / rX	Ionic radius ratio
rB / rX	Ionic radius ratio
Δχ(B–X)	Electronegativity difference

Elemental properties are extracted automatically using Mendeleev:

electronegativity

atomic number (if needed for extensions)

##⚖️ Class Imbalance Handling

To address strong label imbalance, the training data is balanced using:

SMOTEENN

Synthetic Minority Over-sampling (SMOTE)

Edited Nearest Neighbors (ENN) cleaning

##✅ Resampling is applied only to the training set to avoid data leakage.

##🔢 Data Encoding

Each classical feature vector

𝑥
∈
𝑅
5
x∈R
5

is scaled to 
[
0
,
𝜋
]
[0,π] and encoded using angle encoding:

Hadamard initialization

Phase encoding via RZ(2x_i)

This encoding is repeated at every layer (data re-uploading).

##🔀 Quantum Circuit Architecture
Hardware

5 qubits

PennyLane default.qubit device

Ansatz Structure

Each layer consists of:

Data Encoding

Scrambling Entangling Layer

Scrambling Layout (Brick-Wall)

Entangling blocks act on:

(0,1), (2,3), (1,2), (3,4), (4,0)


Each block:

RY ⊗ RY → CNOT → RY ⊗ RY → CNOT (reversed)


4 parameters per block

20 parameters per layer

Data Re-Uploading

The full circuit applies:

Encode → Scramble → Encode → Scramble → ...


for multiple layers.

##📐 Measurement & Output

The model outputs a single expectation value:

⟨
𝑍
0
⟩
⟨Z
0
	​

⟩

Classification rule:

≥ 0 → class +1

< 0 → class −1

##🎯 Loss Function

A margin-based (hinge) loss is used:

𝐿
=
𝐸
[
max
⁡
(
0
,
1
−
𝑦
⋅
𝑦
^
)
2
]
L=E[max(0,1−y⋅
y
^
	​

)
2
]

Why margin loss?

More stable than MSE

Encourages confident classification

Common in large-margin classifiers

##⚙️ Training Details

Optimizer: Adam

Learning rate: 0.02

Epochs: 30

Batch size: 64

Differentiation: adjoint method

Training metrics tracked:

Training loss

Training accuracy

Test accuracy

##📈 Results & Outputs

During training, the following are produced:

##📉 Loss vs Epoch

##📈 Train/Test Accuracy vs Epoch

##📁 CSV file with predictions:

SMOTE_Scrambling_Results.csv


Each row contains:

Encoded test features

Ground-truth label

Model prediction

##🧪 Why This Model Matters

This implementation demonstrates:

Practical data re-uploading in QML

Strong entangling expressivity via scrambling

Correct handling of imbalanced scientific datasets

Scalable gradient evaluation using adjoint differentiation

It is suitable as:

A QML research prototype

A benchmark variational classifier

A base model for materials informatics

##🚀 Future Extensions

Noise models (default.mixed)

Shot-based training

Multi-observable readout

Comparison with classical baselines

Hardware execution

##👤 Author

Mahdi
Quantum Machine Learning • Variational Circuits • Materials Informatics
