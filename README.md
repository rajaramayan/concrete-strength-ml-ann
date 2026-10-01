# 🏗️ Concrete Compressive Strength ML & ANN Research Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-111111?style=for-the-badge)](https://xgboost.readthedocs.io/)

An interactive machine learning and deep learning research platform for predicting **Concrete Compressive Strength (MPa)** based on 8 concrete mix parameters and curing age.

---

## 📌 Research Overview

Predicting the compressive strength of concrete is crucial in civil and structural engineering to ensure structural performance, material optimization, and safety. This research platform evaluates six standalone Machine Learning (ML) regressors and a 6-layer Deep Artificial Neural Network (ANN), alongside an operationally defined **Hybrid Equal-Weight Blending Ensemble** ($\hat{y}_{hybrid} = 0.5 \hat{y}_{XGBoost} + 0.5 \hat{y}_{ANN}$) across 1,030 empirical concrete formulations from the UCI Repository (Yeh, 1998).

The evaluation pipeline enforces strict data leakage prevention: feature standard scaling ($z$-score) parameters are fit strictly on the 80% training set and evaluated across held-out test data (20%) and 10-fold cross-validation ($10-CV$). Feature attribution analysis and dynamic sensitivity simulations (1–180 days curing age, $w/c$ ratio) are integrated alongside an interactive Streamlit web application prototype for decision support.

### 📊 Model Performance Comparison

| Model | MAE (MPa) | RMSE (MPa) | R² Score |
| :--- | :---: | :---: | :---: |
| **XGBoost** 🏆 | **2.61** | **4.20** | **0.941** |
| **Hybrid XGBoost + ANN** | 3.09 | 4.79 | 0.923 |
| **Gradient Boosting** | 3.65 | 5.02 | 0.915 |
| **Random Forest** | 3.51 | 5.19 | 0.910 |
| **Support Vector Regressor (SVR)** | 4.02 | 5.97 | 0.880 |
| **Artificial Neural Network (ANN)** | 4.30 | 6.10 | 0.875 |
| **Linear Regression** | 8.90 | 11.19 | 0.580 |

---

> [!WARNING]
> **Field Deployment & Recalibration Notice**:
> The models function as interpolation tools bounded strictly by the empirical training dataset limits (Cement 102–540 kg/m³, Water 127–247 kg/m³, Age 1–365 days, lab moist curing ~20°C). Field engineers must recalibrate model parameters with local batch plant trial mixes before applying predictions in commercial structural compliance.

---

## ⚙️ Features of the Web Application

- 🧪 **Interactive Mix Strength Predictor**: Adjust ingredient proportions and curing age with real-time strength prediction, concrete category classification, and recommended applications.
- ⚡ **Mix Formulation Presets**: One-click presets (*Standard 28-Day*, *High-Strength*, *Eco Fly-Ash*, *Early 7-Day*).
- 🧮 **Derived Engineering Metrics**: Water-to-Binder ratio ($w/b$), Total Binder content ($kg/m^3$), and estimated concrete density.
- 📊 **Multi-Model Comparison**: Compare predictions across all 7 models simultaneously.
- 📁 **Batch CSV Predictor & Exporter**: Upload batch concrete mix datasets and export predictions.
- 🔍 **Explainability & Sensitivity Simulator**: Feature importance analysis, 1-180 day age curing growth curve simulator, and water-to-cement ratio sensitivity curves.
- 📁 **Research Dataset Explorer**: Search, filter, and download test predictions and cross-validation metrics.

---

## 🧠 Deep Learning & Ensemble Architecture

### Artificial Neural Network (ANN)
- **Input Layer**: 8 Mix Features (`Cement`, `Blast Furnace Slag`, `Fly Ash`, `Water`, `Superplasticizer`, `Coarse Aggregate`, `Fine Aggregate`, `Age`)
- **Hidden Layers**: Dense (128, ReLU) $\rightarrow$ Dense (64, ReLU) $\rightarrow$ Dropout (0.2) $\rightarrow$ Dense (32, ReLU) $\rightarrow$ Dense (16, ReLU)
- **Output Layer**: Dense (1, Linear)
- **Optimizer**: Adam (lr=0.001), Loss: MSE, Scaler: StandardScaler

### Hybrid XGBoost + ANN Equal-Weight Blending Ensemble
$$\hat{y}_{hybrid} = 0.5 \cdot \hat{y}_{XGBoost} + 0.5 \cdot \hat{y}_{ANN}$$

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/rajaramayan/concrete-strength-ml-ann.git
cd concrete-strength-ml-ann
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Reproduce All Models & Artifacts (Optional)
```bash
python train.py
```

### 4. Run Streamlit Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
├── app.py                      # Main Streamlit web application
├── train.py                    # Full training pipeline (reproduces all artifacts)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── xgboost.joblib              # Saved XGBoost model
├── ann_model.keras             # Keras ANN model weights
├── ann_scaler.joblib           # StandardScaler for ANN
├── random_forest.joblib        # Saved Random Forest model
├── gradient_boosting.joblib    # Saved Gradient Boosting model
├── svr.joblib                  # Saved SVR model
├── linear_regression.joblib    # Saved Linear Regression model
├── model_comparison.csv        # Summary evaluation metrics
├── 10_fold_cross_validation.csv# 10-fold CV results
├── feature_importance.csv      # Feature importance rankings
└── test_predictions.csv        # Out-of-sample test predictions
```

---

## 📜 License
This project is open-source under the MIT License.
