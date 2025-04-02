## <center> Detection of Adulteration in Ground Cofee Via Hyperspectral Imaging, Machine Learning and Feature Selection </center>

### **Model Subprojects**

`To facilitate exploration`, `each model` and `related analysis` has been `documented in individual HTML files`. Access these subprojects from the GitHub page:

**Project Link**: 

**Purpose**: This structure allows focused examination of specific models and their performance under various preprocessing and feature selection methods.

## **Overview** 

This project aims to detect and quantify the adulteration of Arabica coffee with Robusta using hyperspectral imaging (HSI) and machine learning techniques. It explores preprocessing, wavelength selection, model optimization, and performance evaluation to establish robust methodologies for food fraud detection.

### **1. Dataset Preparation**

### **Coffee Samples**:

- 25 Arabica and 5 Robusta samples were sourced from major coffee-producing countries.

- Coffee was roasted (medium roast, 15 minutes at 220°C), cooled overnight, and ground into fine powder.

### **Adulteration Levels**:

- Adulteration levels included 0%, 1%, 5%, 10%, 20%, 40%, 60%, 80%, and 100% Robusta content.

- Training Set: 16 Arabica samples mixed with 3 Robusta samples.

- Test Set: 9 Arabica samples mixed with 2 Robusta samples.

- Samples were analyzed in triplicate using Hyperspectral Imaging (HSI).

- A total of 1470 samples were formulated and subjected to analysis using hyperspectral imaging.

- **Purpose**: To simulate real-world scenarios of coffee adulteration while maintaining diverse sample variability for robust model training and testing.

### **2. Spectral Preprocessing**

`Five preprocessing` treatments were applied to enhance signal-to-noise ratio and correct for scattering effects:

- Raw Spectra: Baseline for comparison.

- MSC + SG + 1st and 2nd Derivative:

    - MSC (Multiplicative Scatter Correction): Reduces scatter effects.

    - SG (Savitzky-Golay Smoothing): Smoothens spectra and removes noise.

    - 1st and 2nd Derivative: Enhances spectral features for better differentiation.

- SNV + 1st and 2nd Derivative:

    - SNV (Standard Normal Variate): Corrects for scatter and intensity variations.

**Purpose** To examine the impact of preprocessing on classification and regression performance.


### **3. Dimensionality Reduction**

**Principal Component Analysis (PCA)**

- PCA was used to reduce the high-dimensional spectral data into key components.

- By retaining the most informative principal components, PCA minimizes redundancy while preserving essential variance in the data.

**Purpose**: To address multicollinearity and reduce computational overhead during model training.

### **4. Unsupervised Learning**

**K-Means Clustering**

- Applied to the preprocessed spectra for unsupervised classification of adulteration levels.

- Evaluated cluster separability to identify inherent patterns in the data.

**Purpose**: To explore initial structure in the dataset and validate preprocessing effectiveness.

### **5. Wavelength Selection**

Two methods were employed to select the most informative wavelengths:

- **Boruta**:

    - Identifies all relevant features by iteratively comparing original and shuffled features.

- **Genetic Algorithm (GA) with Recursive Feature Elimination (RFE)**:

    - GA optimizes feature subsets based on fitness scores.

    - RFE iteratively eliminates less important features to refine the selection.

**Purpose**: To identify critical spectral regions for improving model performance and interpretability.

### **6. Classification**

**Binary Classification Models**

Models were trained on three datasets:

  1. Full spectra + PCA.

  2. Boruta-selected wavelengths + PCA.

  3. GA + RFE-selected wavelengths + PCA.

**Models Used**:

- Linear Discriminant Analysis (LDA)

- K-Nearest Neighbors (KNN)

- Support Vector Machine (SVM)

- Random Forest (RF)

- Neural Network (NNET)

- Stacked Ensemble:

    - Base Learners: LDA, KNN, SVM, RF, and NNET.

    - Meta Learner: Random Forest.
    

**SMOTE (Synthetic Minority Over-sampling Technique)**

- Applied to handle class imbalance.

- Models were compared with and without SMOTE to assess its impact.

**Performance Metrics**:

- Matthews Correlation Coefficient (MCC)

- Balanced Accuracy

- Sensitivity

- pecificity

- F1 Score

**Purpose**: To identify the best-performing models and preprocessing pipelines for robust classification.

### **7. Regression**

**Regression Models**

Models were trained on:

   1. Full spectra.

   2. Boruta-selected wavelengths.

   3. GA + RFE-selected wavelengths.

**Models Used**:

- **PLS (Partial Least Squares)**

- **Penalized Regression**:

    - LASSO (Least Absolute Shrinkage and Selection Operator)

    - Ridge Regression

    - Elastic Net

- **Stacked Ensemble**:

    - Base Learners: LASSO, PLS, Elastic Net.

    - Meta Learner: Random Forest.

**Performance Metrics**:

- Root Mean Squared Error (RMSE)

- Ratio of Performance to Deviation (RPD)

**Purpose**: To quantify adulteration levels and evaluate regression performance.

### **8. Results and Insights**

- Preprocessing significantly impacted model performance, with either SNV or MSC + SG + 2nd derivatives yielding the best results.

- PCA effectively reduced dimensionality while preserving model accuracy.

- SMOTE improved performance in imbalanced datasets.

- Stacked ensembles frequently outperformed individual models in both classification and regression tasks.

- Boruta and GA provided interpretable wavelength selection, further enhancing model performance.

### **9. Conclusion**

This study demonstrates the power of hyperspectral imaging and advanced machine learning techniques in detecting and quantifying coffee adulteration. The methodology and findings can be applied to other food fraud detection tasks, contributing to enhanced food safety and authenticity.



