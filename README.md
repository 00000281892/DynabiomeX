# DynaBiomeX: Interpretable Dual-Strategy Deep Learning for Microbiome Risk Stratification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/1_DynaBiomeX_Full_Pipeline.ipynb)

This repository contains the official implementation of the **DynaBiomeX** framework, as described in the paper:
> **"DynaBiomeX: An Interpretable Dual-Strategy Deep Learning Framework for Architectural Noise Filtration in Sparse Longitudinal Microbiome Data"**
> *Awais Qureshi et al.*
> *Submitted to Journal of Biomedical Informatics (2026)*

## 📌 Overview

**DynaBiomeX** is a deep learning framework designed to predict gut dysbiosis following hematopoietic cell transplantation (HCT). It addresses the dual challenges of microbiome data sparsity (zero-inflation) and clinical alarm fatigue.

The framework utilizes a **Screener-Sentinel** workflow:
1.  **The Screener (Sensitivity):** Stacking Ensembles identify broad risk patterns to minimise false negatives.
2.  **The Sentinel (Precision):** An adapted **Temporal Fusion Transformer (TFT)** uses Physiological Gating to filter false positives.

### Key Contributions
* **Physiological Gating:** Uses clinical metadata to validate latent dysbiosis signals.
* **Calibration Analysis:** Validates probabilistic trustworthiness using Brier Scores and Expected Calibration Error (ECE = 0.0085).
* **Interpretability:** Variable Selection Networks (VSN) identify key microbial drivers.

---

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/DynaBiomeX.git](https://github.com/YOUR_USERNAME/DynaBiomeX.git)
    cd DynaBiomeX
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📁 Repository Structure

```text
DynaBiomeX/
├── data/
│   ├── processed/      # Contains dummy_data.csv for testing
│   └── README.md       # Data access instructions
├── notebooks/          # Jupyter notebooks for training & visualization
├── src/                # Modular Python source code
│   ├── layers.py       # Custom Keras layers (Attention, GRN, VSN)
│   ├── models.py       # Bi-LSTM, GRU, and TFT model builders
│   ├── evaluation.py   # Calibration metrics and plotting
│   └── utils.py        # Data loading and model management
├── results/
│   └── saved_models/   # Pre-trained .keras files
└── README.md
```
### Model Zoo (Pre-trained Models)
We provide the best-performing models from our experiments to facilitate instant reproduction of results without retraining.

The models are located in results/saved_models/:

* **tft: Best Enhanced Temporal Fusion Transformer (ECE: 0.0085)

* **gru: GRU + Attention Mechanism

* **bilstm: Bidirectional LSTM Baseline

### How to Load Pre-trained Models
You can load these models using our utility function, which automatically handles custom layers (Attention, GRN, etc.).
```bash
from src import load_trained_model

# Load the best TFT model (The "Sentinel")
model = load_trained_model('tft') 

# Load the GRU baseline
gru_model = load_trained_model('gru')

# Run inference
# y_pred = model.predict(X_test)
```
### Usage
To run the full training and evaluation pipeline using the provided dummy data:

Navigate to the notebooks/ directory.

Open 1_DynaBiomeX_Full_Pipeline.ipynb.

Run all cells.

To generate Calibration Curves (Figure 4 in the manuscript):
```bash
from src import load_trained_model, plot_calibration_curve

# Load models
models = {
    'TFT': load_trained_model('tft'),
    'Bi-LSTM': load_trained_model('bilstm')
}

# Generate predictions and plot
# (See notebook for full implementation)
```

Citation
If you use this code or framework in your research, please cite:
```text
@article{Qureshi2026DynaBiomeX,
  title={DynaBiomeX: An Interpretable Dual-Strategy Deep Learning Framework for Architectural Noise Filtration in Sparse Longitudinal Microbiome Data},
  author={Qureshi, Awais and Wahid, Abdul and Qazi, Shams and Shahzad, Muhammad K. and Kiani, Hashir Moheed},
  journal={Journal of Biomedical Informatics},
  year={2026},
  note={Under Review}
}
```
### License
This project is licensed under the MIT License - see the LICENSE file for details.
