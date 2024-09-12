# Breast Cancer Classification Using Random Forest

## Overview

This project aims to develop a predictive model for classifying breast cancer tumors based on various diagnostic features. The dataset used for this analysis is the Breast Cancer Wisconsin dataset, which contains several attributes pertaining to cell characteristics, as well as labels indicating whether the tumor is malignant (M) or benign (B). By employing a Random Forest classifier along with hyperparameter tuning, we seek to achieve high accuracy and generalization for tumor classification.

## Table of Contents

- [Breast Cancer Classification Using Random Forest](#breast-cancer-classification-using-random-forest)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Objective](#objective)
  - [Data Description](#data-description)
  - [Methodology](#methodology)
  - [Results](#results)
    - [Interpretation of Results:](#interpretation-of-results)

## Objective

The main objective of this project is to accurately classify breast cancer tumors based on provided features. We aim to achieve a high level of performance measured by accuracy, precision, recall, and F1 score.

## Data Description

The dataset is derived from the Breast Cancer Wisconsin (Diagnostic) Data Set, hosted on the UCI Machine Learning Repository. It contains:

- 569 instances (data points)
- 32 attributes (including diagnostic information)
- The target variable is the diagnosis, with 'M' for malignant and 'B' for benign tumors.

For the purpose of model training, the diagnosis labels are transformed into numerical values (1 for malignant, 0 for benign).

## Methodology

1. **Data Preparation**: The initial dataset is loaded into a pandas DataFrame. Irrelevant columns (like 'Unnamed: 32' and 'id') are dropped, and the diagnosis is mapped to numerical values.

2. **Feature Engineering**: The dataset is pre-processed by scaling the features using `StandardScaler` to ensure that the model is not biased towards certain features due to differences in scale.

3. **Data Splitting**: The data is divided into training and testing sets, using an 80/20 split while preserving class distribution via stratification.

4. **Model Training**: A Random Forest Classifier is employed to fit the model using the training data.

5. **Hyperparameter Tuning**: The model undergoes hyperparameter optimization using `RandomizedSearchCV` to identify the best model parameters through cross-validation.

6. **Model Evaluation**: The optimized model is evaluated using various metrics, including accuracy, precision, recall, F1 score, and confusion matrix.

## Results

After training and optimizing the Random Forest model, the following results were obtained:
Accuracy: 0.9649 Precision: 1.0000 Recall: 0.9048 F1 Score: 0.9500 Confusion Matrix: [[72 0] [ 4 38]]

### Interpretation of Results:

- **Accuracy**: The model achieved an accuracy of 96.49%, indicating a high level of correct classifications.
- **Precision**: The precision of 1.00 suggests that when the model predicts a tumor as malignant, it is correct all the time.
- **Recall**: The recall of 0.9048 indicates that about 90.48% of malignant tumors were correctly identified by the model.
- **F1 Score**: The F1 score of 0.9500 reflects a balance between precision and recall.
- **Confusion Matrix**: The confusion matrix shows that while the model made a few false negatives (4), there were no false positives.
