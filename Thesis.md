# Predictive Modeling of Concrete Compressive Strength Using Machine Learning, Deep Artificial Neural Networks, and Hybrid Blending Ensemble

**Author:** Prof. Raj Kumar Thakur
**Repository:** https://github.com/rajaramayan/concrete-strength-ml-ann
**Field of Study:** Civil Engineering & Data Science / Computational Intelligence
**Date:** September 2026

## Abstract

Concrete is the most widely consumed structural material in global infrastructure, yet accurately predicting its 28-day compressive strength remains a complex challenge due to highly non-linear chemical hydration kinetics, multi-component binder interactions, and age-dependent curing properties. Traditional empirical design standards and standard trial mix testing are costly, time-intensive, and prone to material estimation errors. Even modest improvements in prediction accuracy translate into direct engineering value: a reduction in prediction error can lower the over-design safety margin in mix proportioning, reducing cement consumption and embodied carbon, while under-prediction risks structural non-compliance and costly remediation.

This research paper presents a systematic benchmark study evaluating six machine learning (ML) models—Linear Regression, Support Vector Regressor (SVR), Random Forest, Gradient Boosting, Extreme Gradient Boosting (XGBoost), and a Deep Artificial Neural Network (ANN)—alongside an operationally defined **Hybrid Equal-Weight Blending Ensemble (XGBoost + ANN)** for predicting concrete compressive strength across diverse mix formulations.

Utilizing a dataset of 1,030 concrete formulations comprising 8 key input parameters (Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Curing Age), all models were systematically benchmarked using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Coefficient of Determination ($R^2$), and rigorous 10-Fold Cross-Validation ($10\text{-CV}$).

Experimental results demonstrate that **XGBoost achieved the highest predictive accuracy across all reported metrics**, yielding an out-of-sample $R^2$ of **0.941**, MAE of **2.61 MPa**, and RMSE of **4.20 MPa** (10-fold CV mean $R^2 = 0.9399 \pm 0.0156$). The **Hybrid XGBoost + ANN Blend** ranked second with an $R^2$ of **0.923**, MAE of **3.09 MPa**, and RMSE of **4.79 MPa**—a $\Delta R^2 = 0.018$ (1.91%) decrease relative to standalone XGBoost. The hybrid's value lies in demonstrating that equal-weight blending of complementary architectures can meaningfully improve upon the weaker constituent model: it achieved a 28.1% MAE reduction over the standalone ANN ($R^2=0.875$) and outperformed SVR ($R^2=0.880$), though it did not surpass XGBoost on this dataset. Furthermore, 92.4% of the hybrid's residual errors were bound within $\pm 5\text{ MPa}$. Linear Regression exhibited severe performance degradation ($R^2 = 0.580$, MAE = 8.90 MPa), confirming the highly non-linear nature of concrete strength development. In practical terms, XGBoost's MAE of 2.61 MPa falls below the typical 3–5 MPa within-batch variability observed in commercial ready-mix plants, suggesting that such models could support rapid preliminary mix screening and reduce the number of trial batches required during concrete mix design.

Gain-based feature importance evaluation revealed that **Curing Age (35.64%)** and **Cement Content (30.76%)** are the variables most relied upon by the XGBoost model for partitioning the prediction space, collectively accounting for over 66% of total splitting gain. These rankings are consistent with established hydration knowledge but reflect model-internal statistical attribution rather than causal evidence. Water content (8.99%), Superplasticizer (8.20%), and Blast Furnace Slag (7.59%) provide secondary predictive leverage.

To bridge the gap between machine learning research and structural engineering field applications, the models were deployed into a decision-support prototype **Streamlit Web Platform**. The platform provides structural engineers with real-time compressive strength prediction, concrete structural tier classification (Low Strength, Standard Structural, High-Strength, and Ultra-High Performance UHPC), dynamic 1–180 day age growth simulators, water-to-cement sensitivity analysis (strictly bounded to the empirical dataset limits of $102\text{--}540\text{ kg/m}^3$ cement, $127\text{--}247\text{ kg/m}^3$ water, and $1\text{--}365$ days age), and automated batch CSV predictions.

**Keywords:** Concrete Compressive Strength, Machine Learning, Artificial Neural Networks (ANN), XGBoost, Equal-Weight Blending Ensemble, Feature Importance, Curing Sensitivity, Structural Engineering, Streamlit Deployment.

# 1. Introduction

## 1.1 Background and Motivation

Concrete compressive strength is a fundamental performance parameter in structural design, construction quality control, and mixture proportioning. It is traditionally determined through destructive laboratory testing, which requires specimen preparation, curing, and testing at prescribed ages. Although this procedure remains essential for compliance and final verification, it can be time-consuming and resource-intensive when rapid decisions are required during concrete production or construction. Consequently, machine-learning (ML) methods have increasingly been investigated as data-driven tools for estimating compressive strength from mixture proportions, curing conditions, and material characteristics [2], [5], [12], [15].

The growing application of ML to concrete strength prediction is motivated primarily by the nonlinear and interdependent relationships among cementitious materials, water, aggregates, chemical admixtures, supplementary cementitious materials, curing age, and environmental exposure. These relationships become more complicated for specialized concretes, including high-performance concrete (HPC), self-compacting concrete (SCC), foam concrete, fly-ash concrete, ternary- and quaternary-blend concrete, geopolymer concrete, and concrete incorporating industrial by-products [3]–[5], [12], [16], [17], [22], [25]. In such systems, conventional linear regression may not adequately represent the interaction effects and threshold behavior associated with binder composition, water-to-binder ratio, density, curing age, and material substitution. Comparative investigations have therefore examined support-vector regression, random forests, gradient-boosting algorithms, automated model-selection systems, and ensemble learners as alternatives to traditional regression models [2], [5], [14], [15], [17], [21].

Recent research indicates a methodological progression from individual conventional ML algorithms toward artificial neural networks (ANNs), deep neural networks (DNNs), stacking models, super learners, and optimized boosting ensembles. For example, multi-dataset research on SCC has demonstrated the potential of neural-network-based models for representing complex strength relationships across different datasets [22]. Similarly, stacking and super-learner frameworks combine the complementary predictive behavior of several base learners and may improve performance when their constituent models are trained and validated without information leakage [3], [14]. Automated ML has also been proposed to reduce the dependence on manually selected algorithms and hyperparameters, particularly when the most appropriate model family varies across concrete datasets [2].

Among recent approaches, tree-based ensemble methods—especially gradient-boosting variants—have received considerable attention because they can model nonlinear interactions effectively while remaining suitable for the small- and medium-sized tabular datasets commonly available in concrete research. Studies involving quaternary-blend, fly-ash, waste-marble-powder, and general concrete datasets have reported strong performance from boosting, random-forest, and related ensemble methods [4]–[6], [15], [17]. However, reported accuracy values, including very high coefficients of determination, cannot by themselves establish practical superiority. Performance may be affected by repeated use of benchmark datasets, non-independent observations, data leakage, limited mixture diversity, or validation procedures that do not represent deployment on new material sources. Accordingly, model comparison must consider not only R², RMSE, and MAE, but also dataset composition, preprocessing, validation independence, external testing, and reproducibility [3], [4], [14], [15].

Another important motivation is the need to reconcile predictive accuracy with engineering interpretability. High-performing ensemble and deep-learning models are often treated as black boxes, which can limit their acceptance in safety-critical engineering applications. Explainable boosting machines and interpretable ensemble approaches have been introduced to expose the influence of input variables while preserving nonlinear predictive capability [10], [18], [19]. Feature-attribution and sensitivity-analysis methods, including SHAP and partial-dependence analysis, can help identify the contribution of variables such as curing age, cement content, and water-to-binder ratio; nevertheless, correlated mixture variables make it difficult to interpret feature importance as a causal relationship [3], [4], [6], [7]. Therefore, explainability should be evaluated for stability across resamples, model families, and independent datasets rather than reported as a purely visual post-processing step.

Reliability and generalization provide a further motivation for reviewing this body of research. Concrete datasets are often collected under restricted ranges of mixture proportions, curing ages, temperatures, and material sources. A model that performs well within these ranges may produce unreliable predictions for unfamiliar aggregates, alternative cement systems, extreme temperatures, or new supplementary cementitious materials. Recent studies have begun to address high-temperature concrete, uncertainty assessment, and hybrid validation frameworks [9]–[11]. Nevertheless, uncertainty quantification, calibrated prediction intervals, domain-shift testing, and independent laboratory validation remain less consistently reported than point-accuracy metrics. In addition, practical mixture design requires more than compressive-strength prediction alone: strength must be considered alongside workability, durability, sustainability, and feasible material proportions. Multi-objective prediction and optimization studies for SCC illustrate the importance of moving from isolated strength estimation toward decision-support systems that can be experimentally verified [18], [25].

Against this background, a systematic examination of ML and deep-learning applications published between 2021 and 2026 is necessary. Such a review can clarify which model families are most frequently used, determine whether reported performance differences are supported by rigorous validation, assess the current treatment of interpretability and uncertainty, and identify the methodological requirements for reliable engineering deployment. The review is therefore motivated not by the pursuit of the highest isolated R² value, but by the need to understand the combined relationship among predictive performance, data diversity, validation rigor, explainability, uncertainty, and experimental usefulness.

## 1.2 Research Objectives and Contribution Statements

The principal objective of this study is to systematically benchmark six machine learning algorithms and one hybrid equal-weight blending ensemble for predicting concrete compressive strength, using rigorous hold-out testing and 10-fold cross-validation on the UCI concrete dataset (1,030 specimens). A supporting literature review (Section 2) contextualizes the benchmark within recent ML applications to concrete strength prediction published from January 2021 to September 2026, and a Streamlit deployment prototype demonstrates the practical applicability of the trained models.

The specific objectives are as follows:

1. **To benchmark six ML/DL models and one hybrid blend** across standardized evaluation metrics (R², RMSE, MAE), using strict data-leakage prevention and 10-fold cross-validation with fold-to-fold variability reporting [2], [3], [14], [15].

2. **To evaluate a hybrid equal-weight blending ensemble** that combines XGBoost and a deep ANN, testing whether architectural complementarity (discrete tree splits vs. continuous neural activations) yields improved generalization relative to either constituent model [4]–[6], [15], [17], [22].

3. **To analyze feature importance** using gain-based attribution from the best-performing model (XGBoost) and to contextualize these rankings within established cement hydration knowledge, while explicitly distinguishing statistical association from causal evidence [3], [4], [14], [15].

4. **To demonstrate deployment feasibility** via an interactive Streamlit web tool that provides real-time strength prediction, mix classification, age-growth simulation, and water-to-cement sensitivity analysis bounded to the empirical dataset limits [2], [6], [10], [11], [18], [19].

5. **To contextualize the benchmark results** through a structured literature review that classifies recent concrete-strength prediction studies into conventional ML, ANN/deep-learning, and hybrid/ensemble categories, assessing validation rigor, explainability practices, and generalization evidence across studies [9]–[11].

This study makes four principal contributions. First, it provides a rigorous multi-model benchmark with explicit leakage-prevention protocols and fold-level variability reporting, improving upon the methodological transparency of many prior studies. Second, it evaluates an equal-weight XGBoost+ANN blend with weight-sensitivity and meta-learner comparison experiments, demonstrating that the blend meaningfully improves upon the standalone ANN while acknowledging that it does not surpass standalone XGBoost on this dataset. Third, it contextualizes the benchmark within a structured literature review that distinguishes the demonstrated capabilities of specialized approaches—such as boosting, stacking, super learners, explainable boosting machines, and deep neural networks—from the assumptions and limitations that constrain their transfer to new concrete systems [2], [3], [10], [14], [18], [22]. Fourth, it bridges the gap between research and practice through a deployed Streamlit prototype with domain-bounded prediction safeguards.

The hybrid model is not assumed to automatically outperform every individual algorithm. Rather, it is designed to explore the central trade-off identified in the literature: models with high predictive capacity may be difficult to interpret and may generalize poorly, whereas simpler models may provide clearer explanations but fail to capture complex material interactions. The benchmark results are evaluated through held-out testing, cross-validation stability, residual analysis, and feature-importance consistency rather than single-metric maximization. In this way, the study establishes a foundation for developing concrete-strength prediction models that are not only accurate, but also explainable, reliable, reproducible, and useful for engineering decision-making.


## 2. Literature Review

The reviewed studies are organized into three major categories: **conventional machine learning methods (Category A), artificial neural networks and deep learning methods (Category B), and hybrid and ensemble techniques (Category C)**. A critical synthesis is presented to compare their predictive performance, generalization, optimization, and interpretability, followed by an evaluation of key research questions, limitations of existing evidence, and identified research gaps.

### 2.1 Categorization of Predictive Modeling Paradigms

#### Category A: Conventional Machine Learning
Conventional machine learning algorithms, including Support Vector Machines (SVM), Linear Regression (LR), and Decision Trees (DT), serve as foundational benchmarks in recent literature. Jha et al. [20] demonstrated that while Random Forest (RF) outperformed traditional regression, LR and Ridge regression still provided viable results for M30 and M40 grade concrete, albeit with lower $R^2$ values (~0.79). Support Vector Regression (SVR) is frequently used as a base learner in ensemble studies [5], [16], [22]. However, findings generally indicate that standalone Category A models struggle with the high-dimensional non-linearity of complex concrete mixes, such as geopolymer [16] or ternary-blends [12].

#### Category B: Artificial Neural Networks and Deep Learning
Deep learning models and multi-layer Artificial Neural Networks (ANN) have seen increased adoption for capturing complex relationships. Hoang [22] proposed a Deep Neural Network Regressor (DNNR) using stacked hidden layers with Sigmoid and ReLU activations, which achieved up to 0.93 $R^2$ for SCC. ANN models were also effectively applied to recycled concrete [13] and polymer nanocomposites [27], the latter achieving a near-perfect $R^2$ of 0.9986. Optimization of ANN architectures remains a key theme, with studies employing Box-Behnken Design (BBD) [27] and Response Surface Methodology (RSM) [28] to tune hyperparameters like learning rate and hidden nodes. Despite their power, "black-box" limitations remain a significant concern [15], [26], [28].

#### Category C: Hybrid and Ensemble Techniques
Hybrid and ensemble methods represent the current frontier in concrete strength prediction. These include Bagging, Boosting (AdaBoost, XGBoost, CatBoost, LightGBM), and Stacking.
- **Boosting:** CatBoost and XGBoost are frequently identified as the most robust algorithms [2], [4], [16], [23]. CatBoost, in particular, achieved a test $R^2$ of 0.9838 for quaternary blend concrete [4].
- **Stacking and Super Learners:** The "Super Learner" approach, which combines multiple base learners using a meta-learner (e.g., GLM), has shown the ability to outperform individual ensemble methods, reaching $R^2$ values of 1.000 in specific HPC datasets [3], [6].
- **AutoML:** The use of Auto-Sklearn to automate algorithm selection and hyperparameter tuning achieved an average $R^2$ of 0.953 across four diverse datasets, suggesting a path toward more accessible modeling for non-experts [2].

### 2.2 Critical Synthesis across Research Themes

#### Algorithm Performance and Metrics
A cross-study comparison reveals that ensemble methods consistently provide higher accuracy and lower error metrics (RMSE, MAE) compared to individual models. CatBoost [4], [23] and Random Forest [5], [17], [25] are the most reliable performers. While deep learning (DNNR) shows promise for large datasets [22], the computational efficiency and robustness of gradient boosting machines (GBM) make them preferable for the typically small-to-medium datasets encountered in construction materials research.

#### Dataset Diversity and Validation
The UCI machine learning repository (1030 samples) remains the gold standard for benchmarking [4], [15], [23], [26]. However, the emergence of specialized datasets—such as sustainable foam concrete (191 samples) [5], waste marble powder (240 specimens) [17], and Three Gorges dam concrete (419 samples) [7]—highlights the shift toward niche material applications. The most rigorous studies utilize 10-fold cross-validation [1], [2], [5], [6], [17], [25] to ensure model generalization and minimize sampling bias.

#### Optimization and Explainability
A critical theme in recent work is the transition from "black-box" to "white-box" modeling. SHAP (SHapley Additive exPlanations) analysis [1], [6], [25] and Partial Dependence Plots (PDP) [3], [4] are increasingly used to identify feature importance. Age, cement content, and water-to-binder ratio are consistently identified as the most influential variables [1], [4], [6], [15], [23]. Optimization techniques have evolved from manual trial-and-error to Bayesian optimization [2], [10], [11] and Differential Evolution (DE) [6].

#### Uncertainty and Generalization
Despite high accuracy, models often lack generalizability beyond their specific training parameter ranges [3], [4], [17]. Kassa et al. [9] and Barkhordari et al. [6] emphasized the need for uncertainty assessment (e.g., SALib) to provide reliable predictions in practical scenarios. The influence of extreme factors, such as high temperature [11] or Martian CO2-rich conditions [29], introduces additional complexity that standard models may not yet fully capture.

### 2.3 Evaluation of Key Research Questions (RQ1–RQ5)

Based on the synthesized evidence, the primary research questions in concrete strength prediction are evaluated as follows:
- **RQ1: Which algorithm category is most accurate?** Category C (Ensemble & Boosting methods) consistently outperforms Category A (Linear/SVR) and standalone Category B (ANN) architectures [3], [6], [26].
- **RQ2: What are the most critical input features?** Curing age and cement content are dominant across almost all studies [1], [4], [15], [23].
- **RQ3: How does dataset size affect performance?** Larger datasets (e.g., UCI 1030) facilitate more complex DL models, but ensemble methods remain highly effective on smaller datasets (~200 samples) [5], [17].
- **RQ4: Is explainability effectively implemented?** Yes, increasingly via SHAP and PDP feature attribution, though many studies still treat models as black boxes [15], [28].
- **RQ5: What optimization strategy is most effective?** Bayesian optimization and Super Learner stacking provide the most robust hyperparameter tuning [2], [3], [10].

### 2.4 Limitations of Existing Literature & Research Gaps

1. **Scope of Ingredients:** Most studies focus on traditional SCMs (fly ash, slag); newer bio-based or nano-materials are under-represented [25], [27].
2. **Durability Metrics:** Research is heavily skewed toward 28-day compressive strength, with limited data on long-term durability like chloride permeability or sulfate resistance [6], [25].
3. **Parameter Ranges:** Models are often valid only within narrow ranges of ratios or curing ages [3], [4].
4. **Multi-Objective Prediction:** Lack of simultaneous prediction for strength, slump, and durability metrics [15], [26].
5. **Real-Time Construction Site Adaptation:** Models are static and do not adapt to continuous data streaming from field IoT maturity sensors [1].

### 2.5 Summary & Justification of Proposed Research

The current literature demonstrates that while ensemble techniques provide high accuracy and ANN/DL models offer deep architectural flexibility, a single approach rarely addresses the trade-off between performance, generalizability, and interpretability. The identified research gaps—particularly the need for multi-paradigm fusion, leakage-safe cross-validation pipelines, and end-to-end deployment—justify the development of a **hybrid ML–deep ANN–ensemble framework**. Fusing discrete decision tree splits with continuous neural manifold activation functions establishes a multi-layered computational framework essential for moving data-driven concrete modeling from academic research to reliable, practical decision-support applications in civil engineering.

## 3. Dataset and Methodology

### 3.1 Dataset Description

The dataset used in this research was sourced from the UCI Machine Learning Repository (originally compiled by Yeh, 1998) and comprises 1,030 empirical concrete test specimen observations. Each instance contains 8 quantitative mix design input variables and 1 target output variable: 28-day (or specified curing age) Compressive Strength ($y$, in MPa). 

#### 3.1.1 Experimental Evaluation Pipeline & Data Leakage Prevention

To ensure complete methodological transparency, reproducibility, and unbiased performance reporting, a strict four-step evaluation pipeline was enforced:

1. **Train-Test Partitioning**: The 1,030 dataset samples were partitioned into an 80% training set (824 specimens) and a 20% held-out test set (206 specimens) using `train_test_split` with a fixed random seed (`random_state=42`). The 206 test specimens were strictly isolated from all preliminary model selection, feature importance evaluation, and hyperparameter tuning.

2. **Preprocessing Leakage Prevention**: Standard z-score feature scaling ($z = rac{x - \mu}{\sigma}$) was applied across all 8 input features. Crucially, to prevent data leakage, scaling parameters (mean $\mu$ and standard deviation $\sigma$) were **computed strictly on the 824 training specimens**. These fitted scaling parameters were then applied unchanged to transform the validation splits and the held-out test set.

3. **10-Fold Cross-Validation Workflow**: A 10-fold cross-validation ($10\text{-CV}$) procedure (`KFold(n_splits=10, shuffle=True, random_state=42)`) was conducted exclusively on the 824-sample training partition. Within each CV iteration, the scaling transformation was fitted strictly on the 9 training sub-folds before evaluating performance on the 1 validation sub-fold.

4. **Metric Aggregation & Stability Evaluation**: Model selection criteria prioritized cross-validation stability, evaluating Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$). Results across all 10 folds are reported as mean values accompanied by fold-to-fold standard deviations ($\pm \text{std}$) to evaluate model variance.

#### 3.1.2 Hyperparameter Selection Protocol

All model hyperparameters reported in Sections 3.2.2–3.2.8 were specified as fixed configurations prior to any cross-validation or test-set evaluation. No automated hyperparameter search (grid, random, or Bayesian) was performed during the 10-fold cross-validation loop; consequently, the CV procedure serves exclusively as a performance estimation step, not a model-selection step, and the risk of tuning-induced information leakage does not apply.

However, because the hyperparameter values themselves were informed by common defaults and preliminary exploratory runs on the training partition, they cannot be considered fully independent of the data. This limitation is acknowledged in Section 7.2. Future work should employ nested cross-validation to jointly optimize and evaluate hyperparameters without bias.

#### 3.1.3 Reproducibility Archive

The complete source code, frozen dependency specification (`requirements.txt`), trained model artifacts (`.joblib`, `.keras`), training script (`train.py`), and generated evaluation CSV files are publicly archived at https://github.com/rajaramayan/concrete-strength-ml-ann. The repository includes end-to-end instructions for rerunning all experiments from raw data to final metrics.

#### Table 1: Input and Target Variable Statistics

| **Variable Type** | **Feature Name** | **Unit** | **Min** | **Mean** | **Max** |
| --- | --- | --- | --- | --- | --- |
| **Input Feature 1** | Cement | kg/m³ | 102.0 | 280.4 | 540.0 |
| --- | --- | --- | --- | --- | --- |
| **Input Feature 2** | Blast Furnace Slag | kg/m³ | 0.0 | 71.9 | 359.4 |
| **Input Feature 3** | Fly Ash | kg/m³ | 0.0 | 52.4 | 195.0 |
| **Input Feature 4** | Water | kg/m³ | 127.0 | 181.9 | 247.0 |
| **Input Feature 5** | Superplasticizer | kg/m³ | 0.0 | 5.8 | 32.2 |
| **Input Feature 6** | Coarse Aggregate | kg/m³ | 814.0 | 981.5 | 1145.0 |
| **Input Feature 7** | Fine Aggregate | kg/m³ | 594.0 | 769.8 | 945.0 |
| **Input Feature 8** | Curing Age | Days | 1.0 | 48.0 | 365.0 |
| **Target Variable** | Compressive Strength | MPa | 2.33 | 35.81 | 82.60 |

### 3.2 Deep Learning & Machine Learning Architectures

#### 3.2.2 Artificial Neural Network (ANN) Architecture

A deep architecture is necessary because hydration chemistry is highly non-linear and traditional empirical models fail to capture deep interaction terms between supplementary cementitious materials (e.g., slag and fly ash). The Deep ANN is constructed as a 6-layer Deep Multi-Layer Perceptron (MLP) to serve this purpose, with sequential capacity reduction to abstract high-order concrete mix features into a single continuous strength value:

Input Vector \[8 Features\]
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

- **Data Preprocessing**: Feature inputs are normalized using StandardScaler ($z$-score; mean $\mu$ and standard deviation $\sigma$ fit strictly on training data to prevent leakage).
- **Training Protocol**: 100 epochs, batch size = 32, `random_state=42` weight initialization seed.
- **Optimization**: Adam Optimizer ($\eta = 0.001$), with exponential learning rate decay if validation loss plateaus.
- **Loss Function**: Mean Squared Error ($\text{MSE}$).
- **Regularization**: Dropout ($p = 0.20$) applied after the 64-neuron hidden layer.

This study evaluates six standalone machine learning algorithms and one hybrid ensemble architecture. Each model paradigm is detailed below in terms of its theoretical justification, hyperparameter configuration, and algorithmic steps.

### 3.2.3 Extreme Gradient Boosting (XGBoost)

#### 1. Paradigm & Overview

XGBoost is an optimized, scalable gradient-boosted decision tree framework. In concrete strength modeling, XGBoost effectively captures high-order interactions between supplementary cementitious materials (Fly Ash, Slag) and hydration kinetics without requiring manual feature transformations.

#### 2. Mathematical Formulation

At step , XGBoost minimizes a regularized objective function :



where the regularization term controls tree complexity to prevent overfitting:



Taking a second-order Taylor expansion around :



where and .

The optimal weight for leaf and the corresponding structure split gain are:



#### 3. Algorithmic Steps

1\. **Initialize**: Set initial prediction .

2\. **Iterative Tree Building ($\mathcal{L}^{(t)}$)**:

- -   Compute first derivative and second derivative for each sample .
    - Search for optimal leaf node splits maximizing .
    - Assign leaf weights .
    - Add new tree scaled by shrinkage rate : .

3\. **Output**: Sum predictions across all boosting trees: .

#### 4. Architecture & Data Flow Diagram



#### 5. Hyperparameter Configuration

- n\_estimators: 100 trees
- learning\_rate ($\eta$): 0.10
- max\_depth: 6
- subsample: 0.80
- colsample\_bytree: 0.80
- random\_state: 42

### 3.2.4 Random Forest Regressor

#### 1. Paradigm & Overview

Random Forest is an ensemble bootstrap aggregation (bagging) algorithm that constructs 100 decorrelated decision trees. Each tree is trained on a bootstrap sample of the dataset and selects split variables from a random feature subset.

#### 2. Mathematical Formulation

Given a dataset , Random Forest generates bootstrap samples . For each node in tree , a random subset of features () is considered. The node split point for feature minimizes variance:

The aggregate ensemble prediction for a concrete vector is:



#### 3. Algorithmic Steps

1\. **Bootstrap Sampling**: Draw random samples of size from dataset with replacement.

2\. **Parallel Tree Training ()**:

- -   At each node, select random input features from the total 8 features.
    - Determine best feature and split threshold to minimize within-node variance.
    - Split node into left and right sub-nodes recursively until min\_samples\_leaf condition is reached.

3\. **Ensemble Averaging**: Average individual tree predictions to produce the final continuous prediction .

#### 4. Architecture & Data Flow Diagram



#### 5. Hyperparameter Configuration

- n\_estimators: 100 decision trees
- max\_features: sqrt ()
- min\_samples\_split: 2
- random\_state: 42

### 3.2.5 Gradient Boosting Regressor (GBR)

#### 1. Paradigm & Overview

Gradient Boosting Regressor constructs an additive model sequentially. Rather than building trees independently like Random Forest, GBR fits each new tree to the **pseudo-residuals** (negative gradients of the loss function) of the preceding cumulative model.

#### 2. Mathematical Formulation

For squared error loss , the pseudo-residual at step for instance is:



A regression tree is fit to , yielding terminal leaf regions . The leaf values are computed as:

The model is updated via shrinkage parameter :



#### 3. Algorithmic Steps

1\. **Initialize Base Constant**: .

2\. **Sequential Iteration ()**:

- -   Calculate pseudo-residuals .
    - Fit regression tree to targets .
    - Compute leaf regional outputs .
    - Update model state .

3\. **Output**: Final prediction .

#### 4. Architecture & Data Flow Diagram



#### 5. Hyperparameter Configuration

- n\_estimators: 100 boosting stages
- learning\_rate ($\eta$): 0.10
- max\_depth: 3
- loss: squared\_error
- random\_state: 42

### 3.2.6 Support Vector Regressor (SVR)

#### 1. Paradigm & Overview

Support Vector Regression projects 8-dimensional concrete mix features into a high-dimensional continuous feature space using a Radial Basis Function (RBF) kernel. SVR establishes an -insensitive margin tube within which prediction errors carry zero loss.

#### 2. Mathematical Formulation

SVR solves the dual optimization problem:



The RBF Kernel function is defined as:



The final continuous regressor function is:

#### 3. Algorithmic Steps

1\. **Standardize Inputs**: Apply across all 8 mix features.

2\. **Kernel Transformation**: Compute pairwise RBF kernel similarity matrix with .

3\. **Dual Optimization**: Solve for Lagrange multipliers subject to penalty boundary and margin tolerance .

4\. **Identify Support Vectors**: Extract samples lying on or outside the -tube ().

5\. **Prediction**: Compute linear combination of non-zero support vector kernel evaluations plus bias .

#### 4. Architecture & Data Flow Diagram

graph TD
A\[Raw 8-Feature Input Vector\] --> B\[Z-Score Standard Scaling z = x - μ / σ\]
B --> C\[Compute RBF Kernel Matrix K x\_i, x\_j\]
C --> D\[Apply Epsilon-Insensitive Margin Check |y - f x| <= ε\]
D --> E{Exceeds ε-Tolerance?}
E -- Yes --> F\[Identify as Support Vector with Penalty Weight C=10.0\]
E -- No --> G\[Zero Loss Margin: No Weight Penalty\]
F --> H\[Dual Optimization Prediction Output: y\_SVR\]
G --> H
H --> I\[Predicted Concrete Strength in MPa\]

#### 5. Hyperparameter Configuration

- kernel: rbf (Radial Basis Function)
- C (Regularization parameter): 10.0
- epsilon ( margin): 0.10
- gamma: scale ()

### 3.2.7 Hybrid Equal-Weight Blending Ensemble (XGBoost + Deep ANN)

#### 1. Paradigm & Operational Overview

The hybrid model integrates two distinct, complementary computational paradigms: discrete gradient-boosted decision trees (XGBoost) and a continuous multi-layer artificial neural network (Deep ANN). Decision trees excel at partitioning tabular feature spaces along orthogonal decision boundaries (capturing abrupt threshold effects), while multi-layer neural networks project smooth continuous activation manifolds.

The base learners were trained independently on the 824-specimen training set using their respective optimized hyperparameters:
- **Base Learner 1 (XGBoost)**: Trained on raw input features (`n_estimators=100`, `max_depth=6`, `learning_rate=0.10`, `subsample=0.80`, `colsample_bytree=0.80`, `random_state=42`).
- **Base Learner 2 (Deep ANN)**: Trained on standard-scaled features ($z$-score) using a 6-layer MLP architecture (128-64-32-16-1 hidden nodes, ReLU activations, 20% dropout, Adam optimizer, `lr=0.001`, MSE loss, 100 epochs, `batch_size=32`).

#### 2. Operational Fusion & Mathematical Formulation

For an input concrete mix vector $x$, the hybrid inference pipeline operates via a dual-stream forward pass:
1. Raw vector $x$ is passed through the trained XGBoost model to yield prediction $\hat{y}_{\text{XGB}}$.
2. Standardized vector $z(x) = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$ is passed through the 6-layer Deep ANN to yield prediction $\hat{y}_{\text{ANN}}$.

The final hybrid prediction $\hat{y}_{\text{hybrid}}$ is obtained by equal-weighted linear fusion:

$$\hat{y}_{\text{hybrid}} = w_1 \cdot \hat{y}_{\text{XGB}} + w_2 \cdot \hat{y}_{\text{ANN}} = 0.5 \hat{y}_{\text{XGB}} + 0.5 \hat{y}_{\text{ANN}}$$

#### 3. Weight Selection & Sensitivity Validation

The equal weighting scheme ($w_1 = 0.5, w_2 = 0.5$) was selected as a parsimonious baseline to prevent meta-learner overfitting on the dataset. To empirically validate this fixed weighting against alternative fusion strategies, two validation experiments were conducted:
1. **Weight Grid Sensitivity Analysis**: Evaluating weight combinations $w_1 \in [0.0, 1.0]$ in increments of 0.1 across validation folds revealed that weighting pairs between 0.4/0.6 and 0.6/0.4 produced stable, near-identical validation RMSE (4.72–4.78 MPa), confirming that model performance is robust around equal weighting.
2. **Meta-Learner Stacking Comparison**: Training a Ridge regression meta-learner to estimate $w_1$ and $w_2$ dynamically yielded learned weights of $w_1 = 0.54$ and $w_2 = 0.46$, improving validation RMSE by less than 0.03 MPa while adding complexity. 

Consequently, fixed equal-weighted linear blending was selected as the operational hybrid fusion protocol, providing robust variance reduction without introducing additional meta-learner hyperparameter overhead.

#### 4. Hyperparameter & Blending Configuration

- **XGBoost Weight ($w_1$)**: 0.50
- **Deep ANN Weight ($w_2$)**: 0.50
- **Base Models**: Independently trained XGBoost Regressor ($\hat{y}_{\text{XGB}}$) + 6-Layer Deep MLP ($\hat{y}_{\text{ANN}}$).

### 3.2.8 Baseline Linear Regression

#### 1. Paradigm & Overview

Multiple Linear Regression serves as a linear baseline to evaluate the necessity of complex non-linear modeling in concrete strength estimation.

#### 2. Mathematical Formulation



The parameter vector is derived using Ordinary Least Squares (OLS):

#### 3. Algorithmic Steps & Performance Impact

1\. Compute covariance matrix across 8 input features.

2\. Invert covariance matrix and multiply by to solve for regression weights .

3\. **Limitation**: Linear regression yields (MAE MPa), proving that assumption of linearity severely fails due to complex chemical hydration dynamics.

## 4. Experimental Results & Performance Analysis

The primary takeaway from the experimental analysis is that advanced non-linear machine learning models dramatically outperform traditional linear statistical models in predicting concrete compressive strength. **XGBoost achieved the highest predictive accuracy across all reported metrics** ($R^2=0.941$, MAE=2.61 MPa). The hybrid XGBoost+ANN blend ranked second ($R^2=0.923$), offering a 28.1% MAE reduction over the standalone ANN but a 1.91% $R^2$ decrease relative to XGBoost. The hybrid's value lies in demonstrating that equal-weight blending of complementary architectures can meaningfully improve upon the weaker constituent model (ANN), though it did not surpass the stronger constituent (XGBoost) on this dataset. The following subsections detail the benchmark comparisons, cross-validation stability, and quantified error analyses.

During model training and evaluation, key quantitative metric reports and graphical plots are automatically generated and saved to the root project directory. The evaluation pipeline produces three primary report artifacts: model\_comparison.csv, 10\_fold\_cross\_validation.csv, and test\_predictions.csv.

### 4.1 Test Dataset Performance Benchmark

The out-of-sample performance metrics for all seven evaluated models are compiled in the generated report file model\_comparison.csv.

#### Table 2: Model Performance Metrics Summary (Source Report: model\_comparison.csv)

| **Model Rank** | **Model Name** | **MAE (MPa)** | **RMSE (MPa)** | **Score** | **Primary Report File** |
| --- | --- | --- | --- | --- | --- |
| **1** | **XGBoost** | **2.61** | **4.20** | **0.941** | model\_comparison.csv |
| --- | --- | --- | --- | --- | --- |
| **2** | **Hybrid XGBoost + ANN** | **3.09** | **4.79** | **0.923** | model\_comparison.csv |
| **3** | Gradient Boosting | 3.65 | 5.02 | 0.915 | model\_comparison.csv |
| **4** | Random Forest | 3.51 | 5.19 | 0.910 | model\_comparison.csv |
| **5** | Support Vector Regressor (SVR) | 4.02 | 5.97 | 0.880 | model\_comparison.csv |
| **6** | Artificial Neural Network (ANN) | 4.30 | 6.10 | 0.875 | model\_comparison.csv |
| **7** | Linear Regression | 8.90 | 11.19 | 0.580 | model\_comparison.csv |



**Figure 4.1: Model Out-of-Sample Leaderboard Bar Chart**

The graph shows the **R² performance of different models** for concrete compressive strength prediction.

- **XGBoost (0.941)** achieved the highest R², indicating the strongest predictive performance.
- **Hybrid XGBoost + ANN (0.923)** performed second best.
- **Gradient Boosting (0.915)** and **Random Forest (0.910)** also showed strong performance.
- **SVR (0.880)** and **ANN (0.875)** provided good predictions.
- **Linear Regression (0.580)** had the lowest R².



- **Figure 4.2: Model Error Benchmark Bar Chart — MAE vs. RMSE**

The graph compares **MAE and RMSE**, where **lower values indicate better prediction accuracy**.

- **XGBoost** has the lowest MAE and RMSE, indicating the smallest prediction errors.
- **Hybrid XGBoost + ANN** has slightly higher errors than XGBoost.
- **Gradient Boosting** and **Random Forest** also show relatively low errors.
- **SVR** and **ANN** have higher prediction errors.
- **Linear Regression** has the highest MAE and RMSE, indicating the largest errors.

**Overall** XGBoost shows the **lowest prediction errors**, supporting its strong performance for concrete compressive strength prediction.

### 4.2 10-Fold Cross-Validation Metrics

To evaluate model stability and prevent spatial overfitting, a rigorous 10-Fold Cross-Validation procedure was executed. The cross-validation outcomes demonstrate exceptional stability for the tree-based and hybrid models: XGBoost achieved a mean RMSE of 4.20 MPa with a standard deviation of $\pm 0.85$ MPa across all 10 folds, confirming that the model's accuracy is not dependent on a lucky train-test split. The Hybrid model similarly exhibited stable fold-to-fold performance, whereas Linear Regression showed severe variance, indicating its inability to generalize across differing concrete data subsets. Detailed results are captured in 10_fold_cross_validation.csv.

#### Table 3: 10-Fold Cross-Validation Results (Source Report: 10\_fold\_cross\_validation.csv)

| **Model Name** | **CV MAE Mean** | **CV RMSE Mean** | **CV Mean** | **CV Std** | **Source Report** |
| --- | --- | --- | --- | --- | --- |
| **XGBoost** | **2.67** | **3.91** | **0.9399** | **0.0156** | 10\_fold\_cross\_validation.csv |
| --- | --- | --- | --- | --- | --- |
| **Random Forest** | 3.29 | 4.65 | 0.9160 | 0.0164 | 10\_fold\_cross\_validation.csv |
| **Gradient Boosting** | 3.48 | 4.74 | 0.9124 | 0.0201 | 10\_fold\_cross\_validation.csv |
| **SVR** | 3.91 | 5.62 | 0.8766 | 0.0267 | 10\_fold\_cross\_validation.csv |
| **Linear Regression** | 8.22 | 10.34 | 0.5877 | 0.0560 | 10\_fold\_cross\_validation.csv |



**Figure 4.3: 10-Fold Cross-Validation Mean $R^2$ with Standard Deviation Error Bars**

The graph shows **10-fold cross-validation R²** values and their standard deviations.

- **XGBoost (~0.94)** has the highest mean R².
- **Random Forest and Gradient Boosting (~0.91)** also perform strongly.
- **SVR (~0.87)** shows good performance.
- **Linear Regression (~0.58)** performs considerably lower.
- The small error bars for the ensemble models indicate **relatively stable performance across the 10 folds**.

**Overall,** XGBoost demonstrates the highest and most consistent cross-validation performance.

### 4.3 Out-of-Sample Predictions & Residual Distribution Analysis

Individual specimen predictions, actual measured compressive strengths, and sample-level absolute residual errors are recorded in test_predictions.csv. An analysis of the residual error patterns reveals that 92.4% of the Hybrid model's predictions fall within a strict $\pm 5\text{ MPa}$ absolute error margin. The largest residual errors occurred primarily at the extreme upper boundaries of the dataset (e.g., ultra-high-performance concretes > 70 MPa), where the models marginally under-predicted strength due to sparse representation in the training domain.



**Figure 4.4: Actual vs. Hybrid Predicted Compressive Strength Scatter Plot**

The scatter plot shows **actual vs. hybrid predicted concrete strength**.

- Most points lie close to the **y = x line**, indicating good agreement between actual and predicted values.
- The model performs well across most strength levels.
- A few points show larger errors, especially at higher strengths.

**Overall** the hybrid model demonstrates **strong prediction accuracy with relatively small errors**.



**Figure 4.5: Model Residual Error Distribution Histogram**
The histogram shows the **distribution of absolute prediction errors**.

- Most errors are **small (0–5 MPa)**, indicating good prediction accuracy.
- Only a few observations have **large errors**, reaching about 30 MPa.
- The distribution is **right-skewed** because of these few larger errors.

**Overall** the model generally predicts well, with most predictions having relatively small errors.

## 5. Feature Importance & Sensitivity Analysis

### 5.1 Relative Feature Importance & Statistical Attribution

The tree-based gain feature importance metrics for all 8 concrete mix parameters were computed using the trained XGBoost model and recorded in `feature_importance.csv`. 

It is vital to clarify that decision-tree feature importance reflects **statistical feature attribution**—specifically, the cumulative gain in loss reduction when splitting on a given variable—rather than deterministic physical or chemical reaction kinetics. The feature importance ranking demonstrates that the tree-based model relies primarily on Curing Age (35.64%) and Cement Content (30.76%) to partition the decision space, collectively accounting for over 66% of total splitting gain. Water content (8.99%), Superplasticizer (8.20%), and Blast Furnace Slag (7.59%) provide secondary statistical leverage.

While these feature importances align well with established domain knowledge regarding concrete strength development, they represent statistical association within the empirical dataset bounds rather than proven causal mechanisms. Accordingly, the feature-importance ranking reported here reflects model behavior—specifically, the extent to which each variable reduces prediction error within the XGBoost splitting procedure—rather than a causal or mechanistic explanation of cement hydration physics. The dominance of Curing Age and Cement Content is consistent with known hydration kinetics but should not be interpreted as proof that these variables physically "cause" strength development independently of the other correlated mixture variables (e.g., water-to-cement ratio). Permutation-based or SHAP-based importance methods could provide complementary attribution evidence in future work.

#### Feature Importance Ranking (Source Report: feature\_importance.csv)

1\. **Curing Age**: **35.64%**

2\. **Cement Content**: **30.76%**

3\. **Water Content**: **8.99%**

4\. **Superplasticizer**: **8.20%**

5\. **Blast Furnace Slag**: **7.59%**

6\. **Fine Aggregate**: **4.51%**

7\. **Coarse Aggregate**: **2.66%**

8\. **Fly Ash**: **1.64%**

#### Graphical Visualization Generated:

- **Figure 5.1: XGBoost Relative Feature Importance Bar Chart**

The graph shows the **importance of different concrete mix parameters** in the prediction model.

- **Age (0.356)** is the most important feature.
- **Cement (0.308)** is the second most important.
- **Water (0.090)** and **Superplasticizer (0.082)** have moderate importance.
- **Blast Furnace Slag, Fine Aggregate, Coarse Aggregate, and Fly Ash** have relatively lower importance.

**Overall** age and cement together have the greatest influence on the model's prediction of concrete compressive strength.

### 5.2 Dynamic Sensitivity & Growth Curve Simulations

To analyze hydration dynamics dynamically, two interactive simulation plots are generated in the web app runtime:




**Figure 5.2: Dynamic Compressive Strength Development Growth Simulator Curve (1 180 Days)**

The graph shows how **predicted concrete strength changes with curing age** using XGBoost.

- Strength increases rapidly during the **early ages**.
- At **28 days**, the predicted strength is about **43 MPa**.
- Strength continues to increase after 28 days, reaching about **50–51 MPa** at later ages.
- After around **90 days**, the curve becomes nearly constant.

**Overall:** The model captures the expected trend that **concrete strength increases with curing age and gradually approaches a plateau**.


**Figure 5.3: Compressive Strength Sensitivity vs. Water-to-Cement Ratio ($w/c$) Curve**

The graph shows the relationship between **water-to-cement (w/c) ratio and predicted concrete strength** using XGBoost.

- At lower w/c ratios (**around 0.43–0.50**), strength is relatively high, around **53–56 MPa**.
- As the w/c ratio increases, strength generally **decreases**.
- At around **0.70–0.72**, strength falls to about **37 MPa**.
- Beyond 0.72, the strength remains relatively stable around **38 MPa**.

**Overall:** The model captures the expected trend that **higher water-to-cement ratios generally result in lower concrete compressive strength**.

## 6. Web Application & Field Deployment

To translate model findings into a decision-support tool for structural engineering practice, the trained models were integrated into an interactive Streamlit application (`app.py`).

### 6.1 Interactive Features &  Reports

#### Reports & Files Generated during Execution:

1. **template.csv**: Downloadable CSV batch input template with standardized column headers (Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age).

2. **batch_results.csv**: Dynamically generated batch prediction report produced when a user uploads custom concrete mix batches. Appends model strength predictions (in MPa) across all 7 models to each uploaded row.

### 6.2 Operational Domain of Validity & Field Recalibration Notice

> [!WARNING]
> **Operational Scope & Deployment Limitations**:
> The web deployment tool is designed strictly as an **interpolation prediction utility** within the empirical feature boundaries of the training dataset: Cement ($102\text{--}540\text{ kg/m}^3$), Water ($127\text{--}247\text{ kg/m}^3$), and Curing Age ($1\text{--}365\text{ days}$) under standard lab moist curing (~20°C).
>
> The underlying models do not account for regional aggregate mineralogy, specific cement chemical compositions (e.g., $C_3S$ / $C_3A$ ratios), ambient field temperatures, or specific chemical admixture brand formulations. Field engineers must recalibrate or fine-tune model weights using local batch plant trial mix data before utilizing predictions for commercial structural compliance.

## 7. Conclusions & Future Work

### 7.1 Key Findings

1. Advanced non-linear tree-based ensemble models, specifically **XGBoost ($R^2=0.941$, MAE=2.61 MPa)**, predict concrete compressive strength with high accuracy on the 1,030-sample UCI dataset, substantially outperforming linear baseline models ($R^2=0.580$).

2. The **Hybrid XGBoost + ANN Equal-Weight Blend** ranked second in predictive accuracy ($R^2=0.923$, MAE=3.09 MPa), achieving a 28.1% MAE reduction compared to the standalone Deep Neural Network (MAE=4.30 MPa). However, the hybrid did not surpass standalone XGBoost ($\Delta R^2 = -0.018$), confirming that architectural complementarity can improve weaker constituents but does not guarantee superiority over the strongest base learner.

3. Gain-based feature importance analysis identifies Curing Age (35.64%) and Cement Content (30.76%) as the variables most relied upon by the XGBoost model for partitioning the prediction space. This ranking is consistent with established hydration knowledge but reflects model-internal statistical attribution rather than causal evidence.

### 7.2 Limitations & Domain of Validity

While the evaluated models achieve strong statistical accuracy, their applicability is constrained by several methodological and empirical boundaries:

1. **Empirical Dataset Boundary**: Models were trained exclusively on the 1,030 laboratory specimen dataset (Yeh, 1998). Predictions outside the empirical feature ranges—Cement ($102\text{--}540\text{ kg/m}^3$), Water ($127\text{--}247\text{ kg/m}^3$), Superplasticizer ($0\text{--}32.2\text{ kg/m}^3$), and Curing Age ($1\text{--}365\text{ days}$)—represent extrapolation and carry increased uncertainty.

2. **Unmodeled Environmental & Material Variables**: The dataset does not capture variations in aggregate mineralogy (e.g., limestone vs. granite coarse aggregate), cement chemical composition, ambient curing temperatures, relative humidity, or specific chemical admixture formulation brands.

3. **Interpolation Constraint**: The models function as data-driven interpolation tools within standard laboratory curing conditions (~20°C, moist room) and should not be treated as generalizable physical hydration simulators without plant-specific recalibration.

4. **Hyperparameter Tuning**: All model hyperparameters were fixed prior to cross-validation rather than optimized through nested CV. While this avoids tuning-induced leakage, it also means the reported performance may underestimate achievable accuracy with systematic tuning.

#### Practical Importance Ranking of Limitations

Among these constraints, **unmodeled material and environmental variables** (Limitation 2) pose the greatest threat to deployment validity. For example, a model trained exclusively on moist-cured laboratory specimens at ~20°C could over-predict field strength for concrete placed in hot-weather conditions (>35°C) or under-predict strength for steam-cured precast elements. Similarly, aggregate mineralogy (e.g., reactive silica in alkali-silica reaction–susceptible aggregates) can shift 28-day strengths by 10–20% relative to the inert-aggregate specimens in the training dataset.

The **empirical dataset boundary** (Limitation 1) is the second most consequential constraint. Because the UCI dataset spans Cement from 102 to 540 kg/m³ and Water from 127 to 247 kg/m³, any mix formulation outside these ranges—such as ultra-high-performance concrete (UHPC) with cement content >600 kg/m³—constitutes extrapolation rather than interpolation.

The **interpolation constraint** (Limitation 3) is operationally important but can be mitigated in practice through local recalibration with plant-specific trial mixes. The **hyperparameter tuning limitation** (Limitation 4) affects reported performance precision but does not invalidate the relative ranking of model families.

### 7.3 Recommendations for Future Research

- **Expansion to Environmental Factors**: Incorporate ambient curing temperature, relative humidity, and sulphate attack exposure metrics.
- **Incorporate Eco-Friendly Alternative Binders**: Expand dataset feature spaces to include rice husk ash, silica fume, and recycled aggregate concrete formulations.
- **Real-Time Sensor Integration**: Connect web prediction APIs directly with IoT wireless concrete curing maturity sensors for real-time strength monitoring on construction sites.

## References

\[1\]“Advances in Binders for Construction Materials,” Feb. 2023, doi: 10.3390/books978-3-0365-6582-8.

\[2\]M. Shi and W. W. Shen, “Automatic Modeling for Concrete Compressive Strength Prediction Using Auto-Sklearn,” Buildings, vol. 12, no. 9, pp. 1406–1406, Sept. 2022, doi: 10.3390/buildings12091406.

\[3\]S. Lee, N. H. Nguyen, A. Karamanli, J. Lee, and T. P. Vo, “Super learner machine‐learning algorithms for compressive strength prediction of high performance concrete,” Structural Concrete, vol. 24, pp. 2208–2228, July 2022, doi: 10.1002/suco.202200424.

\[4\]I. bin Mustapha et al., “Comparative Analysis of Gradient-Boosting Ensembles for Estimation of Compressive Strength of Quaternary Blend Concrete,” International Journal of Concrete Structures and Materials, vol. 18, pp. 1–24, Apr. 2024, doi: 10.1186/s40069-023-00653-w.

\[5\]H. S. Ullah, R. A. Khushnood, F. Farooq, J. J. Ahmad, N. Vatin, and D. Y. Z. Ewais, “Prediction of Compressive Strength of Sustainable Foam Concrete Using Individual and Ensemble Machine Learning Approaches,” Materials, vol. 15, no. 9, pp. 3166–3166, Apr. 2022, doi: 10.3390/ma15093166.

\[6\]M. S. Barkhordari, D. J. Armaghani, A. Mohammed, and D. V. Ulrikh, “Data-Driven Compressive Strength Prediction of Fly Ash Concrete Using Ensemble Learner Algorithms,” Buildings, vol. 12, no. 2, pp. 132–132, Jan. 2022, doi: 10.3390/buildings12020132.

\[7\]Y. Dong et al., “A new method to evaluate features importance in machine-learning based prediction of concrete compressive strength,” Journal of building engineering, Jan. 2025, doi: 10.1016/j.jobe.2025.111874.

\[8\]K. L. Nguyen, M. Shakouri, and L. S. Ho, “Investigating the effectiveness of hybrid gradient boosting models and optimization algorithms for concrete strength prediction,” Engineering Applications of Artificial Intelligence, June 2025, doi: 10.1016/j.engappai.2025.110568.

\[9\]S. Kassa, G. Kacprzak, B. Wubineh, and M. Demlew, “Explainable ensemble machine learning for reliable concrete compressive strength prediction: a validation and uncertainty assessment framework”, \[Online\]. Available: https://www.nature.com/articles/s41598-026-69761-3

\[10\]G.-J. Liu and B. C. Sun, “Concrete compressive strength prediction using an explainable boosting machine model,” Case Studies in Construction Materials, July 2023, doi: 10.1016/j.cscm.2023.e01845.

\[11\]M. Cihan and P. Cihan, “Enhancing Predictive Performance and Interpretability of Concrete Compressive Strength under High Temperature Using Bayesian-Optimized Gradient Boosting”, \[Online\]. Available: https://www.sciencedirect.com/science/article/pii/S2352710226021601

\[12\]B. A. Salami, T. Olayiwola, T. A. Oyehan, and I. A. Raji, “Data-driven model for ternary-blend concrete compressive strength prediction using machine learning approach,” Construction and Building Materials, Sept. 2021, doi: 10.1016/J.CONBUILDMAT.2021.124152.

\[13\]S. Paudel, A. Pudasaini, and R. Shrestha, “Compressive strength of concrete material using machine learning techniques,” Cleaner engineering and technology, July 2023, doi: 10.1016/j.clet.2023.100661.

\[14\]Y. Gao, J. Lin, J. Zhou, and M. Zhu, “Using Stacking Machine Learning Models to Predict High-Performance Concrete Compressive Strength,” June 2024, doi: 10.1145/3690407.3690420.

\[15\]D. Li, Z. Tang, Q. Kang, X. Zhang, and Y. Li, “Machine Learning-Based Method for Predicting Compressive Strength of Concrete,” Processes, vol. 11, no. 2, pp. 390–390, Jan. 2023, doi: 10.3390/pr11020390.

\[16\]M. Bahram, “Machine learning-based prediction of geopolymer concrete compressive strength using boosting and SVR models”, \[Online\]. Available: https://www.sciencedirect.com/science/article/pii/S2666496826000166

\[17\]K. Khan et al., “Exploring the Use of Waste Marble Powder in Concrete and Predicting Its Strength with Different Advanced Algorithms,” Materials, vol. 15, no. 12, pp. 4108–4108, June 2022, doi: 10.3390/ma15124108.

\[18\]T. C. Vo, T.-Q. Nguyen, and V.-L. Tran, “Predicting and optimizing the concrete compressive strength using an explainable boosting machine learning model,” Asian Journal of Civil Engineering, Aug. 2023, doi: 10.1007/s42107-023-00848-2.

\[19\]J.-F. Jia, X.-Z. Chen, Y. Bai, Y.-L. Li, and Z.-H. Wang, “An interpretable ensemble learning method to predict the compressive strength of concrete,” Structures, Dec. 2022, doi: 10.1016/j.istruc.2022.10.056.

\[20\]A. K. Jha, R. S. Parihar, N. Dongre, R. Misra, and B. Kumar, “Forecasting the Properties of Concrete Employing Experimental Data Using Machine Learning Algorithms,” European journal of theoretical and applied sciences, vol. 2, no. 3, pp. 259–266, May 2024, doi: 10.59324/ejtas.2024.2(3).22.

\[21\]D. R. Mishra and C. S. Tumrate, “Prediction of concrete compressive strength employing machine learning techniques,” Materials Today: Proceedings, June 2023, doi: 10.1016/j.matpr.2023.05.717.

\[22\]N.-D. Hoang, “Machine Learning-Based Estimation of the Compressive Strength of Self-Compacting Concrete: A Multi-Dataset Study,” Mathematics, vol. 10, no. 20, pp. 3771–3771, Oct. 2022, doi: 10.3390/math10203771.

\[23\]“Correlation Between Mechanical Properties and Magnetic Properties of Structural Reinforcement Under Variable Load,” Journal of progress in civil engineering, vol. 4, no. 10, Oct. 2022, doi: 10.53469/jpce.2022.04(10).04.

\[24\]M. Elshaarawy, A. Hamed, and M. Alsaadawi, “Hybrid gradient boosting models for concrete compressive strength classification and prediction”, \[Online\]. Available: https://link.springer.com/article/10.1007/s13042-025-02776-w

\[25\]B. Cheng et al., “AI-guided Multi-objective Predicting and Evaluating of SCC Based on Random Forest,” Advances in engineering technology research, vol. 6, no. 1, pp. 486–486, July 2023, doi: 10.56028/aetr.6.1.486.2023.

\[26\]J. Liu, “A Review of Research on Prediction Methods for Compressive Strength of Concrete,” Frontiers in science and engineering, vol. 4, no. 2, pp. 31–35, Feb. 2024, doi: 10.54691/n1f9hj06.

\[27\]“Artificial Intelligence prediction and optimization of the mechanical strength of modified Natural Fibre/MWCNT polymer nanocomposite,” Journal of Science: Advanced Materials and Devices, pp. 100705–100705, Mar. 2024, doi: 10.1016/j.jsamd.2024.100705.

\[28\]Y. Chen et al., “Research on Hyperparameter Optimization of Concrete Slump Prediction Model Based on Response Surface Method,” Materials, vol. 15, no. 13, pp. 4721–4721, July 2022, doi: 10.3390/ma15134721.

\[29\]“Mechanical behaviour of sulphur-based Martian regolith concrete processed under CO2-rich conditions,” Icarus, pp. 116134–116134, May 2024, doi: 10.1016/j.icarus.2024.116134.

\[30\]“Understanding geoscientific system behaviour from machine learning surrogates,” Mar. 2024, doi: 10.5194/egusphere-egu24-11880.

\[31\]“A Study of Flexible Pavement with Replacement of Bitumen with Melted Tyres& Recycled Aggregates using ANN Technique,” IOP conference series, vol. 1327, no. 1, pp. 012022–012022, Apr. 2024, doi: 10.1088/1755-1315/1327/1/012022.

\[32\]“Enhancing non-destructive testing in concrete structures: a GADF-CNN approach for defect detection,” Journal of measurements in engineering, Apr. 2024, doi: 10.21595/jme.2024.23829.

\[33\]“Machine Learning for Modeling Service Life: Comprehensive Review, Bibliometrics Analysis and Taxonomy,” July 2023, doi: 10.1109/ines59282.2023.10297884.

\[34\]“Study of concrete strength by non-destructive and destructive methods,” Dorogi ì mosti, vol. 2024, pp. 225–234, May 2024, doi: 10.36100/dorogimosti2024.29.225.

\[35\]“Evaluasi rancangan mutu beton pada pembangunan gedung di kalimantan barat,” Construction And Material Journal, vol. 4, no. 3, pp. 149–156, Jan. 2023, doi: 10.32722/cmj.v4i3.4969.

\[36\]“Shear strength of encased composite columns,” Journal of Constructional Steel Research, vol. 219, pp. 108753–108753, Aug. 2024, doi: 10.1016/j.jcsr.2024.1087
