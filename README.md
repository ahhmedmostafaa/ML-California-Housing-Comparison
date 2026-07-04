# 🏠 California Housing — ML Models: From Scratch vs Scikit-learn

A hands-on machine learning project implementing **6 core algorithms from scratch** (pure NumPy, no ML libraries) and benchmarking each one against its **scikit-learn** equivalent on the California Housing dataset — to verify correctness and compare performance.

## Why this project?
Anyone can call `model.fit()`. This project proves the math underneath actually works: every algorithm below is implemented from first principles (matrix algebra, gradient descent, recursive tree splitting) and validated against the industry-standard library implementation.

## Models Implemented

| # | Model | Task | Scratch Technique |
|---|-------|------|--------------------|
| 1 | Linear Regression | Regression | Normal Equation `w = (XᵀX)⁻¹Xᵀy` |
| 2 | Logistic Regression | Classification | One-vs-Rest + Gradient Descent |
| 3 | K-Nearest Neighbors | Regression | Euclidean distance, k=5 |
| 4 | Naïve Bayes | Classification | Gaussian likelihood + log-prior |
| 5 | Decision Tree | Regression | CART, MSE-minimizing splits (depth=5) |
| 6 | Random Forest | Regression | Bootstrap aggregation, 50 trees |

## Results — Scratch vs Sklearn

**Regression (California Housing prices)**

| Model | Impl | RMSE | MAE | R² |
|---|---|---|---|---|
| Linear Regression | Scratch | 0.7456 | 0.5332 | 0.5758 |
| Linear Regression | Sklearn | 0.7456 | 0.5332 | 0.5758 |
| KNN (k=5) | Scratch | 0.6576 | 0.4462 | 0.6700 |
| KNN (k=5) | Sklearn | 0.6576 | 0.4462 | 0.6700 |
| Decision Tree (d=5) | Scratch | 0.7301 | 0.5272 | 0.5932 |
| Decision Tree (d=5) | Sklearn | 0.7242 | 0.5223 | 0.5997 |
| **Random Forest (50)** | **Scratch** | **0.5055** | **0.3398** | **0.8050** |
| Random Forest (50) | Sklearn | 0.5072 | 0.3303 | 0.8037 |

**Classification (Low/Medium/High price bands)**

| Model | Impl | Accuracy |
|---|---|---|
| Logistic Regression | Scratch | 0.7149 |
| Logistic Regression | Sklearn | 0.7345 |
| Naïve Bayes | Scratch | 0.5128 |
| Naïve Bayes | Sklearn | 0.5128 |

**🏆 Best model: Random Forest (R² = 0.805)** — the from-scratch implementation actually edges out sklearn on RMSE and R², confirming the implementation is correct and well-tuned.

## Tech Stack
- Python, NumPy, Pandas
- Scikit-learn (for benchmarking + dataset)
- Matplotlib (visualizations)
- Jupyter Notebook

## Files
- `california_housing_ml_comparison.ipynb` — full implementation, tables, and visualizations
- `California_Housing_ML_Report.pdf` — written report with methodology and analysis

## How to Run
```bash
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook california_housing_ml_comparison.ipynb
```
