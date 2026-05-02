# 🏠 House Price Prediction — Regression Models from Scratch vs. scikit-learn

![](house-prices-scaled.jpg)

End-to-end machine learning pipeline for predicting residential property sale prices — implementing **Linear, Polynomial, Ridge (L2), and Lasso (L1) regression from scratch**, then benchmarking against scikit-learn equivalents on a real estate dataset.

---

## 🎯 What This Project Does

> *"How much does regularization actually help — and does a hand-built model hold up against a production library?"*

Builds, tunes, and compares 8 regression models across two implementation tracks: custom NumPy implementations with gradient descent, and scikit-learn baselines — evaluated on the same holdout set.

---

## 🔬 Pipeline

**1. Data Preprocessing**
- Combined train and test sets for unified transformation
- KNN imputation for numerical missing values (preserves local structure vs. mean/median)
- Mode and constant-fill strategies for categorical missing values
- Cosine transform for cyclical month feature (`MoSold`)
- Log transform on skewed numerical features (|skew| ≥ 0.5)
- One-hot encoding of all categoricals
- Z-score scaling implemented from scratch

**2. Feature Engineering**
- Log-transformed `SalePrice` target to normalize right-skewed distribution
- Pearson correlation heatmap to identify high-signal features (GrLivArea: 0.71, TotalBsmtSF: 0.63, YearBuilt: 0.56)
- Polynomial feature expansion (degree = 2) for non-linear models

**3. Models — From Scratch (Gradient Descent)**

| Model | Key Hyperparameters | Notes |
|---|---|---|
| Linear Regression | lr=0.045, iter=200 | Baseline |
| Polynomial Regression | lr=0.00008, iter=40k, deg=2 | Non-linear fit |
| Ridge (L2) | lr=0.00008, λ=0.00001 | Penalizes large weights |
| Lasso (L1) | lr=0.00008, λ=0.000001 | Sparse weight selection |

**4. Models — scikit-learn Benchmarks**
- `LinearRegression`, `PolynomialFeatures + LinearRegression`, `Ridge(α=10)`, `Lasso(α=0.005)`
- Each model produces a submission CSV with predicted `SalePrice`

**5. Evaluation**
- RMSE on holdout test set
- Adjusted R² (penalizes model complexity)
- Learning curves (loss vs. iteration) for all gradient descent models
- Target vs. Prediction scatter plots per feature

---

## 💡 Key Finding

> Regularized models (Ridge, Lasso) consistently outperform plain linear regression on this high-dimensional dataset. Lasso additionally performs implicit feature selection — zeroing out coefficients for low-signal features, producing a more interpretable model.

---

## 🛠️ Tech Stack

`Python` · `NumPy` · `pandas` · `scikit-learn` · `Matplotlib` · `Seaborn` · `SciPy`

> Core models built from scratch using NumPy — no sklearn.linear_model used in custom implementations.

---

## 📁 Key Files

| File | Description |
|---|---|
| `house-price-prediction.ipynb` | Full pipeline — preprocessing, all 8 models, evaluation |
| `train.csv` / `test.csv` | Raw house price dataset |
| `result-with-best.csv` | Best-known submission baseline for comparison |
| `*-submission.csv` | Model prediction outputs |

> Dataset: [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## 📊 Models at a Glance

| # | Model | Implementation |
|---|---|---|
| 1 | Linear Regression | From scratch (GD) |
| 2 | Polynomial Regression | From scratch (GD) |
| 3 | Ridge — L2 | From scratch (GD) |
| 4 | Lasso — L1 | From scratch (GD) |
| 5 | Linear Regression | scikit-learn |
| 6 | Polynomial Regression | scikit-learn |
| 7 | Ridge | scikit-learn |
| 8 | Lasso | scikit-learn |
