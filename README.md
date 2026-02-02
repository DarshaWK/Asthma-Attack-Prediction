# Asthma-Attack-Prediction
Using ML techniques to predict the risk of asthma attacks in NZ

# Predicting the Risk of Asthma Attacks in New Zealand Using Machine Learning

This repository contains the methodology and implementation framework used in the study:

**Predicting the Risk of Asthma Attacks in New Zealand Using Machine Learning**  
Proceedings of the 58th Hawaii International Conference on System Sciences (HICSS), 2025

The project develops and evaluates machine learning and statistical models to predict the risk of asthma attacks within a 3-month prediction window, using national-level hospital and pharmaceutical data from New Zealand.

---

## Overview

Asthma attacks represent a major public health burden in New Zealand. Early identification of patients at high risk enables timely clinical intervention and improved disease management.

This study:
- Constructs predictive models using Random Forest (RF), Extreme Gradient Boosting (XGBoost), and Logistic Regression (LR)
- Addresses severe class imbalance in health data
- Evaluates models using AUROC and F1-score
- Identifies clinically meaningful risk predictors using XGB model with RFE

---

## Data Description

- Sources:
  - National Minimum Dataset (NMDS): hospital admissions
  - Pharmaceutical Collection: subsidised medication dispensing
  - Patients aged ≥ 12 years with asthma

**Data access note:** Raw data cannot be shared publicly due to privacy and ethical restrictions.

---

## Outcome Definition

The binary target variable represents the occurrence of an asthma attack within a 3-month period:

- 1 (Attack): Asthma-related hospitalisation and/or oral corticosteroid prescription
- 0 (No attack): No such event during the observation window

---

## Methodology Pipeline

1. Data collection and cohort refinement  
2. Initial feature elimination (expert-driven)  
3. Train–test split (70% / 30%, stratified)  
4. Recursive Feature Elimination (RFE) using RF 
5. Categorical encoding (One-Hot Encoding)  
6. Numerical feature scaling (standardisation or normalisation)  
7. Class imbalance handling on training data only (RUS, SMOTE)  
8. Model training with stratified 10-fold cross-validation  - LR, RF, XGB
9. Hyperparameter tuning using Grid Search  
10. Model evaluation using AUROC and F1-score  
11. Indentify important features using XGB-RUS model

---

