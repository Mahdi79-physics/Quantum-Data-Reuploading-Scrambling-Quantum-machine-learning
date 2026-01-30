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
