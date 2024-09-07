# Predicting PCOS Using Machine Learning Models

This project analyzes and compares the performance of various machine learning classifier models to predict Polycystic Ovary Syndrome (PCOS) using patient data.

## Project Overview

The main objective of this project is to evaluate and compare multiple machine learning algorithms for predicting PCOS in patients. It explores models like Decision Trees, Random Forest, Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and others. The best-performing model is chosen based on several evaluation metrics, including accuracy, precision, recall, and F1-score.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Acknowledgements](#acknowledgements)

## Introduction

Polycystic Ovary Syndrome (PCOS) is a prevalent endocrine disorder among women of reproductive age. Early detection and management can significantly improve patient outcomes. This project applies machine learning techniques to identify the most suitable algorithm for PCOS prediction, aiding in early detection.

## Dataset

The dataset used for this project includes data from 1997 patients, with features related to patient health and medical history. The data is split into 1392 non-PCOS and 605 PCOS cases, and contains relevant features such as hormone levels, BMI, and others.

## Methodology

1. **Data Preprocessing**: Cleaning and organizing the dataset for model input, handling missing values, and performing feature scaling.
2. **Model Training**: Training several machine learning models using the processed data.
3. **Evaluation**: Comparing models using key metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
4. **Hyperparameter Tuning**: Optimizing model parameters to improve performance.

## Results

The results section includes a comparative analysis of the models and their performance metrics. The best-performing model, Decision Tree Classifer, achieved an accuracy of **97.14%**, with a precision of **97.5%**, and an F1-score of **97.5%**.

### Example of Model Comparison

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | xx%      | xx%       | xx%    | xx%      |
| Decision Tree         | xx%      | xx%       | xx%    | xx%      |
| Random Forest         | xx%      | xx%       | xx%    | xx%      |
| Support Vector Machine| xx%      | xx%       | xx%    | xx%      |
| Naive Bayes           | xx%      | xx%       | xx%    | xx%      |
| KNN                   | xx%      | xx%       | xx%    | xx%      |


Detailed visualizations such as confusion matrices and ROC curves can be found in the Jupyter notebook.

## Dependencies

This project uses the following Python libraries:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `jupyter`

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/repository-name.git
```
2. Install the required libraries.
3. Open the Jupyter notebook:
```bash
jupyter notebook
```
4. Run the notebook `Analyzing the performances of various ML Models in predicting PCOS.ipynb`.

## Acknowledgements

This project was completed as part of a research initiative on predictive modeling for healthcare. Special thanks to IGDTUW for providing guidance and resources throughout the process.

---
