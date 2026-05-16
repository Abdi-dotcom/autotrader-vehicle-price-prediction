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

## Key Techniques

### Feature Engineering
- **Vehicle age** derived from registration year
- **Price per mile** ratio capturing usage intensity  
- **Mileage per year** distinguishing light vs. heavy use
- **Model popularity score** (frequency + price weighted)
- **Age × Mileage interaction** term
- **Domain-aware imputation:** UK DVLA registration code mapping for missing years

### Automated Feature Selection
- **SelectFromModel (ExtraTrees):** Reduced 38→19 features, best performance
- **RFECV (Ridge):** Selected 35 features, slightly weaker for tree models

### Model Interpretability
- **SHAP summary plot:** `price_per_mile` and `standard_model` dominant drivers
- **Partial Dependence Plots:** Nonlinear relationships confirmed
- **Permutation importance:** Validated robustness of key features

### Dimensionality Reduction (Exploratory)
- **PCA:** 25 components needed for 95% variance → linear compression loses predictive signal
- **t-SNE:** Clear local clustering by price range → explains why tree models excel
- **Isomap:** Global structure preserved but price separation weaker than t-SNE
  
##  What I Learned

- **Handling high-cardinality categorical features:** Target encoding outperformed one-hot for 110+ car brands and 500+ models
- **The importance of stratified cross-validation:** Ensured stable price distribution across train/validation/test splits
- **Tree-based vs. linear models:** Validated that vehicle pricing has strong nonlinear structure, linear models (Ridge, PCA) significantly underperformed
- **Ensemble wisdom:** Weighted averaging correctly identified Random Forest as the dominant model (weight = 1.0), avoiding accuracy dilution from weaker learners
- **Interpretability matters:** SHAP revealed that `price_per_mile` (engineered feature) was more influential than raw mileage, validating domain-driven feature engineering
##  How to Run

```bash
# Clone the repository
git clone https://github.com/Abdi-dotcom/autotrader-vehicle-price-prediction.git
cd autotrader-vehicle-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/


## Academic Context

- **Degree:** MSc Data Science, Manchester Metropolitan University
- **Modules:** Machine Learning & Advanced Machine Learning
- **Timeline:** Oct 2025 – May 2026
- **Grade:** [MLC-85% AML-Pending]
## 🔗 Connect

- [LinkedIn](https://www.linkedin.com/in/abdinassirhared/)
- [Email](mailto:abdinassir12@gmail.com)
