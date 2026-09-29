# Predictive Modeling of Concrete Compressive Strength Using Machine Learning, Deep Artificial Neural Networks, and Hybrid Ensemble Techniques

**Author:** Rajaramayan  
**Repository:** [https://github.com/rajaramayan/concrete-strength-ml-ann](https://github.com/rajaramayan/concrete-strength-ml-ann)  
**Field of Study:** Civil Engineering & Data Science / Computational Intelligence  
**Date:** September 2026  

---

## 📄 Abstract

Concrete is the most widely consumed structural material in global infrastructure, yet accurately predicting its 28-day compressive strength remains a complex challenge due to highly non-linear chemical hydration kinetics, multi-component binder interactions, and age-dependent curing properties. Traditional empirical design standards and standard trial mix testing are costly, time-intensive, and prone to material estimation errors. 

This thesis presents a comprehensive computational research framework evaluating six state-of-the-art machine learning (ML) models—Linear Regression, Support Vector Regressor (SVR), Random Forest, Gradient Boosting, Extreme Gradient Boosting (XGBoost), and a Deep Artificial Neural Network (ANN)—alongside a novel **Hybrid Stacking/Blending Ensemble Model (XGBoost + ANN)** for predicting concrete compressive strength across diverse mix formulations. 

Utilizing a dataset of 1,030 concrete formulations comprising 8 key input parameters (Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Curing Age), all models were systematically benchmarked using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Coefficient of Determination ($R^2$), and rigorous 10-Fold Cross-Validation ($10\text{-CV}$).

Experimental results demonstrate that **XGBoost achieved the highest standalone predictive accuracy**, yielding an out-of-sample $R^2$ of **0.941**, MAE of **2.61 MPa**, and RMSE of **4.20 MPa** (10-CV Mean $R^2 = 0.9399 \pm 0.0156$). The **Hybrid XGBoost + ANN Ensemble** achieved robust generalization performance with an $R^2$ of **0.923**, MAE of **3.09 MPa**, and RMSE of **4.79 MPa**, outperforming single-architecture Deep Neural Networks ($R^2 = 0.875$) and conventional SVR ($R^2 = 0.880$). Linear Regression exhibited severe performance degradation ($R^2 = 0.580$, MAE $= 8.90$ MPa), confirming the highly non-linear nature of concrete strength development.

Feature importance evaluation revealed that **Curing Age (35.64%)** and **Cement Content (30.76%)** are the dominant predictors, collectively explaining over 66% of strength variance, followed by Water content (8.99%), Superplasticizer (8.20%), and Blast Furnace Slag (7.59%). 

To bridge the gap between machine learning research and structural engineering field applications, the models were deployed into a production-grade, interactive **Streamlit Web Platform**. The platform provides structural engineers with real-time compressive strength prediction, concrete structural tier classification (Low Strength, Standard Structural, High-Strength, and Ultra-High Performance UHPC), dynamic 1–180 day age growth simulators, water-to-cement sensitivity analysis, and automated batch CSV predictions.

**Keywords:** Concrete Compressive Strength, Machine Learning, Artificial Neural Networks (ANN), XGBoost, Stacking Ensemble, Feature Importance, Curing Sensitivity, Structural Engineering, Streamlit Deployment.

---

## 1. Introduction

### 1.1 Background & Motivation
Concrete compressive strength (expressed in Megapascals, MPa) is the primary governing criterion in structural design, safety verification, and quality control for civil infrastructure. Standard compressive strength is conventionally evaluated through standardized compression testing of uniaxial concrete cylinders or cubes after 28 days of moist curing. 

However, reliance on physical testing presents major operational drawbacks:
1. **28-Day Curing Delay**: Destructive testing requires a mandatory 28-day wait period before compliance verification, creating project bottlenecks.
2. **Complex Mix Combinations**: Modern high-performance concrete incorporates supplementary cementitious materials (SCMs)—such as Ground Granulated Blast Furnace Slag (GGBS) and Fly Ash—alongside chemical admixtures (Superplasticizers). These multi-component additions alter hydration thermodynamics, rendering traditional empirical water-cement ratio equations ($w/c$) unreliable.
3. **Material Waste & Cost**: Repeated physical trial mixes consume significant material, labor, and energy.

Predictive machine learning algorithms offer an efficient computational alternative capable of capturing multi-variable non-linear relationships without destructive testing.

### 1.2 Research Objectives
The core objectives of this thesis are:
1. To develop, train, and optimize multiple machine learning regressors (Linear Regression, SVR, Random Forest, Gradient Boosting, XGBoost) and a deep Artificial Neural Network (ANN) to model concrete strength.
2. To formulate a **Hybrid Ensemble Model** blending XGBoost and ANN predictions ($\hat{y}_{hybrid} = 0.5 \hat{y}_{XGB} + 0.5 \hat{y}_{ANN}$) to minimize variance and leverage multi-paradigm learning.
3. To conduct 10-Fold Cross-Validation to assess model stability and prevent overfitting.
4. To evaluate feature importance rankings and perform sensitivity simulations on curing age and water-binder ratios.
5. To deploy the computational models into an intuitive, accessible Streamlit web application for real-time engineering decision support.

---

## 2. Methodology & Dataset Characterization

### 2.1 Dataset Description
The dataset consists of 1,030 concrete test specimens measured across 8 quantitative input variables and 1 target output variable (Compressive Strength in MPa).

#### Table 1: Input and Target Variable Statistics
| Variable Type | Feature Name | Unit | Min | Mean | Max |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Input Feature 1** | Cement | $\text{kg/m}^3$ | 102.0 | 280.4 | 540.0 |
| **Input Feature 2** | Blast Furnace Slag | $\text{kg/m}^3$ | 0.0 | 71.9 | 359.4 |
| **Input Feature 3** | Fly Ash | $\text{kg/m}^3$ | 0.0 | 52.4 | 195.0 |
| **Input Feature 4** | Water | $\text{kg/m}^3$ | 127.0 | 181.9 | 247.0 |
| **Input Feature 5** | Superplasticizer | $\text{kg/m}^3$ | 0.0 | 5.8 | 32.2 |
| **Input Feature 6** | Coarse Aggregate | $\text{kg/m}^3$ | 814.0 | 981.5 | 1145.0 |
| **Input Feature 7** | Fine Aggregate | $\text{kg/m}^3$ | 594.0 | 769.8 | 945.0 |
| **Input Feature 8** | Curing Age | Days | 1.0 | 48.0 | 365.0 |
| **Target Variable** | Compressive Strength | MPa | 2.33 | 35.81 | 82.60 |

---

## 3. Deep Learning & Machine Learning Architectures

### 3.1 Artificial Neural Network (ANN) Architecture
The Deep ANN is constructed as a 6-layer Deep Multi-Layer Perceptron (MLP):

```
Input Vector [8 Features]
       │
       ▼
Dense Layer 1: 128 Neurons (ReLU Activation)
       │
       ▼
Dense Layer 2: 64 Neurons (ReLU Activation)
       │
       ▼
Dropout Layer: Rate = 0.20 (Prevents Overfitting)
       │
       ▼
Dense Layer 3: 32 Neurons (ReLU Activation)
       │
       ▼
Dense Layer 4: 16 Neurons (ReLU Activation)
       │
       ▼
Output Layer: 1 Neuron (Linear Activation -> Compressive Strength in MPa)
```

- **Data Preprocessing**: Feature inputs are normalized using `StandardScaler`:
  $$z = \frac{x - \mu}{\sigma}$$
- **Optimization**: Adam Optimizer ($\eta = 0.001$).
- **Loss Function**: Mean Squared Error ($\text{MSE}$).

### 3.2 Machine Learning Models
- **XGBoost**: Gradient boosted decision tree framework optimizing regularized objective functions with exact greedy tree splitting.
- **Random Forest**: Ensemble of 100 decorrelated decision trees using bootstrap aggregation (bagging) and random feature subsets.
- **Gradient Boosting Regressor**: Sequential boosting trees minimizing squared error loss.
- **Support Vector Regressor (SVR)**: Radial Basis Function (RBF) kernel mapping inputs into high-dimensional feature space ($\epsilon\text{-margin} = 0.1, C=10.0$).
- **Hybrid Ensemble (XGBoost + ANN)**: Equal-weighted linear blending:
  $$\hat{y}_{hybrid} = \frac{1}{2} \hat{y}_{XGBoost} + \frac{1}{2} \hat{y}_{ANN}$$

---

## 4. Experimental Results & Performance Analysis

### 4.1 Test Dataset Performance Benchmark

#### Table 2: Model Performance Metrics Summary
| Model Rank | Model Name | MAE (MPa) | RMSE (MPa) | $R^2$ Score |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **XGBoost** | **2.61** | **4.20** | **0.941** |
| **2** | **Hybrid XGBoost + ANN** | **3.09** | **4.79** | **0.923** |
| **3** | Gradient Boosting | 3.65 | 5.02 | 0.915 |
| **4** | Random Forest | 3.51 | 5.19 | 0.910 |
| **5** | Support Vector Regressor (SVR) | 4.02 | 5.97 | 0.880 |
| **6** | Artificial Neural Network (ANN) | 4.30 | 6.10 | 0.875 |
| **7** | Linear Regression | 8.90 | 11.19 | 0.580 |

### 4.2 10-Fold Cross-Validation Metrics

#### Table 3: 10-Fold Cross-Validation Results
| Model Name | CV MAE Mean | CV RMSE Mean | CV $R^2$ Mean | CV $R^2$ Std |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | **2.67** | **3.91** | **0.9399** | **0.0156** |
| **Random Forest** | 3.29 | 4.65 | 0.9160 | 0.0164 |
| **Gradient Boosting** | 3.48 | 4.74 | 0.9124 | 0.0201 |
| **SVR** | 3.91 | 5.62 | 0.8766 | 0.0267 |
| **Linear Regression** | 8.22 | 10.34 | 0.5877 | 0.0560 |

---

## 5. Feature Importance & Sensitivity Analysis

### 5.1 Relative Feature Importance
The Tree-based feature importance evaluation yielded the following contribution percentages:

1. **Curing Age**: **35.64%**  
2. **Cement Content**: **30.76%**  
3. **Water Content**: **8.99%**  
4. **Superplasticizer**: **8.20%**  
5. **Blast Furnace Slag**: **7.59%**  
6. **Fine Aggregate**: **4.51%**  
7. **Coarse Aggregate**: **2.66%**  
8. **Fly Ash**: **1.64%**  

### 5.2 Engineering Insights
- **Age and Cement Dominance**: Together, curing age and cement content account for **66.40%** of total strength variance. This aligns directly with concrete chemistry, where calcium silicate hydrate ($\text{C-S-H}$) gel formation increases exponentially over initial curing days and proportionally with cement paste density.
- **Superplasticizer & Water Ratio**: Superplasticizers enable low water-binder ratios ($w/b < 0.40$) while preserving workability, directly increasing compressive strength without causing segregation.

---

## 6. Web Application & Field Deployment

To translate model metrics into a practical tool for civil engineers and concrete batching plants, an interactive **Streamlit Web Application** was developed and deployed.

### Key Deployment Modules:
1. **Interactive Mix Strength Predictor**: Custom numerical inputs, preset mix templates (*Standard 28-Day*, *High-Strength*, *Eco Fly-Ash*, *Early 7-Day*), derived $w/b$ calculations, and strength classification cards.
2. **Multi-Model Real-Time Comparison**: Computes predictions across all 7 models simultaneously for any given input mix.
3. **Batch CSV Predictor**: Enables automated bulk prediction for large dataset uploads.
4. **Interactive Growth & Sensitivity Simulator**: Dynamic age growth curves ($1\text{--}180$ days) and water-to-cement ratio sensitivity curves.
5. **Research Data Explorer**: Filterable test predictions table with one-click dataset download options.

---

## 7. Conclusions & Future Work

### 7.1 Key Findings
1. Machine learning models, specifically **XGBoost ($R^2 = 0.941$, MAE $= 2.61$ MPa)**, can predict concrete compressive strength with outstanding precision, significantly outperforming traditional linear statistical methods ($R^2 = 0.580$).
2. The **Hybrid XGBoost + ANN Ensemble** provides strong generalization stability ($R^2 = 0.923$) by combining non-linear decision trees with continuous neural activation spaces.
3. Curing Age and Cement Content are the most governing predictors of concrete compressive strength.

### 7.2 Recommendations for Future Research
- **Expansion to Environmental Factors**: Incorporate ambient curing temperature, relative humidity, and sulphate attack exposure metrics.
- **Incorporate Eco-Friendly Alternative Binders**: Expand dataset feature spaces to include rice husk ash, silica fume, and recycled aggregate concrete formulations.
- **Real-Time Sensor Integration**: Connect web prediction APIs directly with IoT wireless concrete curing maturity sensors for real-time strength monitoring on construction sites.

---

## 📚 References

1. Yeh, I-C. (1998). "Modeling of strength of high-performance concrete using artificial neural networks." *Cement and Concrete Research*, 28(12), 1797-1808.
2. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
3. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
4. Neville, A. M. (2011). *Properties of Concrete* (5th Edition). Pearson Education Limited.
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
