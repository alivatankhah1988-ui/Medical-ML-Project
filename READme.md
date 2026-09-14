Medical ML Project: Diabetes Classification

Overview

This project applies machine learning techniques to a structured diabetes-related laboratory dataset. The main objective is to build and compare multiple classification models for predicting the target variable "Result" based on demographic and laboratory features.

The project focuses on:

- Data preprocessing
- Categorical and numerical feature handling
- Machine learning classification
- Model comparison
- Cross-validation
- Overfitting analysis
- Confusion matrix and error analysis
- Feature importance
- Interpretation of model performance

The project was developed as an educational and research-oriented machine learning study, with particular emphasis on reproducible preprocessing and comparative model evaluation.

---

Dataset

The dataset contains laboratory and demographic information related to diabetes classification.

Features

Feature| Description| Type
"Age"| Age of the individual| Numerical
"Gender"| Gender| Categorical
"FBS"| Fasting Blood Sugar| Numerical
"HbA1c"| Glycated hemoglobin| Numerical
"TG"| Triglycerides| Numerical
"Chol"| Cholesterol| Numerical
"Result"| Target class ("NO" / "YES")| Categorical

The target variable is:

Result

with two classes:

NO
YES

---

Project Objectives

The main objectives of this project are to:

1. Prepare and preprocess structured health-related data.
2. Apply several supervised machine learning algorithms.
3. Compare model performance using multiple evaluation metrics.
4. Evaluate model stability using 5-fold cross-validation.
5. Examine potential overfitting.
6. Analyze classification errors using confusion matrices.
7. Investigate feature importance.
8. Identify the strongest-performing model within this experimental setup.

---

Data Preprocessing

The project uses a preprocessing pipeline to handle categorical and numerical variables appropriately.

Categorical Features

"Gender" is transformed using One-Hot Encoding:

Gender → Gender_male / Gender_female

"handle_unknown="ignore"" is used to make the preprocessing pipeline robust to unseen categorical values.

Numerical Features

The numerical variables are:

Age
FBS
HbA1c
TG
Chol

For distance- and scale-sensitive models such as KNN, numerical features are standardized using "StandardScaler".

Tree-based models such as Decision Tree and Random Forest do not require feature scaling.

Data Splitting

The dataset is divided into training and testing sets using:

Training set: 70%
Testing set: 30%

The split uses:

random_state=42
stratify=y

Stratification preserves the class distribution between the training and testing sets.

---

Machine Learning Models

Five classification algorithms were evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Naive Bayes
4. Decision Tree
5. Random Forest

Each model was evaluated using the same general train/test split and appropriate preprocessing.

---

Evaluation Method

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- 5-Fold Cross-Validation

Cross-Validation

A 5-fold cross-validation procedure was used to evaluate model stability across different subsets of the dataset.

Both the mean accuracy and standard deviation were considered.

A higher CV mean indicates stronger average performance, while a lower CV standard deviation indicates greater stability across folds.

---

Model Results

The final comparison is based on the 70/30 test split and 5-fold cross-validation.

Model| Test Accuracy| Precision| Recall| F1-score| CV Mean| CV Std
Random Forest| 100.00%| 100.00%| 100.00%| 100.00%| 99.76%| 0.49%
Logistic Regression| 99.19%| 100.00%| 97.56%| 98.77%| 99.51%| 0.98%
Decision Tree| 100.00%| 100.00%| 100.00%| 100.00%| 99.27%| 0.60%
Naive Bayes| 100.00%| 100.00%| 100.00%| 100.00%| 98.54%| 1.42%
KNN| 98.37%| 100.00%| 95.12%| 97.50%| 98.29%| 1.65%

---

Best Model

Among the evaluated models under the adopted experimental setup, Random Forest showed the strongest overall performance.

Random Forest Performance

Test Accuracy: 100.00%
Precision:     100.00%
Rec