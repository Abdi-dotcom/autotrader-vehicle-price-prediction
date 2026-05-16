# AutoTrader Vehicle Price Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end machine learning pipeline to predict UK used vehicle prices using **402,005 AutoTrader listings**. 
This project was completed as part of my MSc Data Science coursework at Manchester Metropolitan University.

## Objective
Build a regression model that accurately predicts vehicle selling price based on characteristics like mileage, brand, model, fuel type, and body type.

## Results

| Model | Test MAE | Test R² | Notes |
|-------|----------|---------|-------|
| Ridge (baseline) | £3,791 | 0.83 | Linear baseline |
| Decision Tree (tuned) | £1,788 | 0.949 | Strong single-tree performance |
| **Random Forest (tuned)** | **£272** | **0.992** | Best model |
| HistGradientBoosting | £1,554 | 0.98 | Strong but RF dominates |
| Weighted Ensemble | £272 | 0.992 | RF weight = 1.0 (optimal) |

**Key achievement:** 93% error reduction vs. linear baseline (MAE £3,791 → £272)

## Tech Stack
- **Python** · pandas · NumPy · scikit-learn
- **Feature Engineering:** Target encoding, log transforms, domain-aware imputation
- **Feature Selection:** SelectFromModel (ExtraTrees), RFECV
- **Ensemble Methods:** Weighted averaging, stacking
- **Interpretability:** SHAP values, Partial Dependence Plots
- **Dimensionality Reduction:** PCA, t-SNE, Isomap
- **Visualization:** matplotlib · seaborn

## Project Structure
