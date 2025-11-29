## <center>**Detection of Coffee Adulteration via Hyperspectral Imaging, Chemometrics, and Machine Learning**</center>

**Purpose**: This notebook documents the full workflow for detecting and quantifying Robusta adulteration in Arabica coffee using near-infrared hyperspectral imaging (NIR-HSI), chemometric preprocessing, dimensionality reduction, feature selection, classification, regression, and deployment-ready Shiny applications.

## **Abstract**

Detecting coffee adulteration, particularly in instant coffee, remains challenging due to overlapping spectra and the complexity of the matrix. This study demonstrates the effectiveness of near-infrared hyperspectral imaging (NIR-HSI, 900–1700 nm) combined with chemometrics and machine learning (ML) for detecting and quantifying Robusta adulteration in both ground and instant Arabica coffee.

The analytical workflow integrates chemometric preprocessing, including multiplicative scatter correction (MSC), standard normal variate (SNV), and Savitzky–Golay derivatives, with dimensionality reduction and visualization using principal component analysis (PCA) and t-distributed stochastic neighbor embedding (t-SNE), feature selection through Boruta and genetic algorithm–recursive feature elimination (GA-RFE), and a suite of classification and regression models.

Among preprocessing techniques, SNV and MSC, together with second-order Savitzky–Golay derivatives, consistently yielded the highest classification accuracy. Feature selection improved computational efficiency without compromising performance, with Boruta demonstrating greater computational efficiency and reproducibility than GA-RFE, making it more suitable for routine or multi-instrument deployment.

Classification models, including Linear Discriminant Analysis (LDA), k-nearest neighbors (k-NN), support vector machine (SVM), random forest (RF), artificial neural network (ANN), and stacked ensembles, achieved perfect discrimination (balanced accuracy = 100%, MCC = 1.0) when combined with synthetic minority oversampling (SMOTE). Logistic regression, Ranger, and cost-sensitive SVM models also achieved near-perfect classification under class-weighted learning, providing an alternative that preserves spectral integrity.

For quantification, stacked ensembles attained high predictive accuracy (RMSEP < 7%, RPD > 3.0, R²P ≥ 0.98), with significant improvements observed in ground coffee (p < 0.0001) and consistent performance in instant coffee (p > 0.05). Regularized regressors (LASSO, Ridge, Elastic Net) and partial least squares (PLS) provided efficient alternatives for high-dimensional spectral data.

Overall, this approach integrates classical chemometric techniques with modern ML algorithms to enable accurate, non-destructive screening of fraud in both ground and instant coffee.


### **1. Dataset Preparation**

### **Coffee Samples**

- Different samples of coffee (Arabica and Robusta) sourced from the top 10 producing countries.
- Coffee samples were roasted at 220°C for 15 minutes.
- Instant coffee was produced via lyophilization.
- Both **ground** and **instant** coffee samples were included.
- Samples were analyzed in triplicate using near-infrared hyperspectral imaging (NIR-HSI, 900–1700 nm).

### **Adulteration Levels**

- Adulteration levels included: 0%, 1%, 5%, 10%, 20%, 40%, 60%, 80%, and 100% Robusta.
- **Training set**: 16 Arabica samples mixed with 3 Robusta samples.
- **Test set**: 9 Arabica samples mixed with 2 Robusta samples.
- In total, **1470 samples** were formulated and subjected to HSI analysis.

**Purpose**: To simulate real-world scenarios of coffee adulteration while maintaining diverse sample variability for robust model training and testing.

### **2. Spectral Preprocessing**

`Five preprocessing` treatments were applied to enhance signal-to-noise ratio and correct for scattering effects:

- **Raw Spectra**  
  - Used as a baseline for comparison.

- **MSC + SG + 1st and 2nd Derivative**
  - **MSC (Multiplicative Scatter Correction)**: Reduces multiplicative scatter and pathlength effects.
  - **SG (Savitzky–Golay Smoothing)**: Smoothens spectra and reduces high-frequency noise.
  - **1st and 2nd Derivatives**: Enhance subtle spectral features and improve separability.

- **SNV + 1st and 2nd Derivative**
  - **SNV (Standard Normal Variate)**: Corrects for scatter and intensity variations across samples.
  - Derivatives highlight important spectral transitions and reduce baseline effects.

**Purpose**: To examine the impact of different preprocessing strategies on classification and regression performance, and to identify the most effective pipelines for robust coffee adulteration detection.

### **3. Dimensionality Reduction**

#### **Principal Component Analysis (PCA)**

- Reduces high-dimensional spectral data into a smaller set of principal components.
- Minimizes redundancy while preserving most of the variance in the data.
- Used for:
  - Exploratory visualization (score plots).
  - Input to some classifiers.
  - Understanding class separation structure.

#### **t-Distributed Stochastic Neighbor Embedding (t-SNE)**

- Non-linear dimensionality reduction method.
- Provides 2D or 3D visualizations of complex spectral relationships.
- Helps reveal non-linear separation between pure and adulterated samples.

**Purpose**: To address multicollinearity, reduce computational overhead, and better visualize the structure of the dataset before supervised modeling.

### **4. Unsupervised Learning**

#### **K-Means Clustering**

- Applied to preprocessed spectra to perform unsupervised grouping.
- Evaluates how well samples cluster by adulteration level or coffee type.
- Helps validate:
  - Effectiveness of preprocessing.
  - Presence of natural groupings in the data.

**Purpose**: To explore the intrinsic structure of the dataset and provide an unsupervised perspective on separation between pure and adulterated samples.

### **5. Wavelength Selection**

Two main methods were used to identify informative wavelengths:

#### **Boruta**

- Wrapper method based on Random Forest importance.
- Identifies all relevant features by comparing real attributes with shadow (randomized) attributes.
- Provides:
  - Robustness across resamples.
  - Good reproducibility.
  - Improved interpretability of spectral regions.

#### **Genetic Algorithm – Recursive Feature Elimination (GA-RFE)**

- **GA**: Searches for optimal subsets of wavelengths using evolutionary optimization.
- **RFE**: Iteratively removes the least important features to refine the subset.
- Suitable for:
  - Aggressive dimensionality reduction.
  - Discovering compact subsets of key wavelengths.

**Purpose**: To identify critical spectral regions that drive classification and regression performance, while improving model interpretability and reducing computational costs.

### **6. Classification**

#### **Binary Classification Task**

- Objective: Discriminate between **pure Arabica** and **adulterated** samples.
- Models trained using:
  1. Full spectra.
  2. Boruta-selected wavelengths.
  3. GA-RFE-selected wavelengths.

#### **Models Used**

- Linear Discriminant Analysis (**LDA**)
- k-Nearest Neighbors (**k-NN**)
- Support Vector Machine (**SVM**)
- Random Forest (**RF**)
- Artificial Neural Network (**ANN**)
- **Stacked Ensemble**
  - Base learners: LDA, k-NN, SVM, RF, ANN.
  - Meta learner: Random Forest.

#### **Handling Class Imbalance – SMOTE**

- **SMOTE (Synthetic Minority Oversampling Technique)** was applied to address class imbalance.
- Models were compared:
  - With SMOTE.
  - Without SMOTE.
- **Cost-Sensitive Learning** was also used by assigning higher misclassification cost/weight to the minority class. 

#### **Performance Metrics**

- Matthews Correlation Coefficient (**MCC**)
- Balanced Accuracy
- Sensitivity (Recall)
- Specificity
- F1 Score

**Purpose**: To identify high-performing, robust classifiers that can reliably detect adulteration under different preprocessing and feature selection strategies.

### **7. Regression**

#### **Objective**

- Predict the **percentage of Robusta adulteration** in Arabica coffee.

#### **Models Evaluated**

- Partial Least Squares (**PLS**)
- **LASSO** (Least Absolute Shrinkage and Selection Operator)
- **Ridge Regression**
- **Elastic Net**
- **Stacked Ensemble**
  - Base learners: PLS, LASSO, Elastic Net.
  - Meta learner: Random Forest.

#### **Input Variants**

- Full spectra.
- Boruta-selected wavelengths.
- GA-RFE-selected wavelengths.

#### **Performance Metrics**

- Root Mean Squared Error (**RMSE**)
- Ratio of Performance to Deviation (**RPD**)

**Purpose**: To develop accurate, quantitative models capable of estimating Robusta adulteration levels in both ground and instant coffee.

### **8. Deployment: R Shiny Applications**

#### **SpectraVision – Cost-Sensitive Classifier**

- Implements cost-sensitive logistic regression and cost-sensitive Random Forest.
- Designed for binary classification of pure vs adulterated coffee.
- **Link**: https://dmalavi.shinyapps.io/SpectraVision-Cost-Sensitive-Classifier/

#### **SpectraVision – Instant Coffee Classifier**

- Classifies instant coffee samples based on PCA-transformed HSI spectra.
- Uses k-NN and Random Forest for prediction.
- **Link**: https://dmalavi.shinyapps.io/SpectraVision-Instant-Coffee-Classifier/

#### **SpectraQuant – Instant Coffee Regression App**

- Predicts the percentage of Robusta adulteration in instant coffee samples.
- Combines PLS, LASSO/Elastic Net, and stacked regression ensembles.
- **Link**: https://dmalavi.shinyapps.io/SpectraQuant-Instant-Coffee/

**Purpose**: To translate research models into usable decision-support tools for laboratories, regulators, and industry stakeholders.

### **9. Key Insights**

- SNV and MSC combined with second-order Savitzky–Golay derivatives produced the best classification performance.
- PCA and t-SNE offered clear visual separation between pure and adulterated samples.
- Boruta was more computationally efficient and reproducible than GA-RFE for wavelength selection.
- SMOTE and cost-sensitive learning improved minority class detection and overall robustness.
- Stacked ensembles outperformed individual models in both classification and regression.
- Regression models achieved RMSEP < 7, RPD > 3.0, and R² ≥ 0.98 for adulteration prediction.

### **10. Conclusion**

This notebook outlines an end-to-end pipeline for coffee fraud detection using hyperspectral imaging, chemometrics, and machine learning. The workflow demonstrates that combining advanced preprocessing, feature selection, and ensemble learning with deployable Shiny tools enables accurate, non-destructive, and scalable screening of adulteration in both ground and instant coffee.

### 11. Repository structure

<img width="771" height="547" alt="image" src="https://github.com/user-attachments/assets/f27b3b33-da62-4851-8ca1-93df64d45643" />

## **12. License**

This project is licensed under the **MIT License**.

You are free to:
- Use  
- Modify  
- Distribute  

as long as attribution is provided.

A `LICENSE` file has been included in the repository.

## **13. Additional Information**

- All large raw HSI files are intentionally excluded from GitHub due to size limits.
- Scripts are optimized for reproducibility in R (caret, mdatools, tidyverse ecosystem).
- Deployment tools are fully functional and accessible online.
