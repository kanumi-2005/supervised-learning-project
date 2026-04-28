# Supervised Learning: Regression and Classification

> **Project 1** — CSC14005 - Introduction to Machine Learning (Nhập môn Học máy)
> University of Science, Vietnam National University Ho Chi Minh City (VNUHCM-US)
> Faculty of Information Technology

## Table of Contents

- [Overview](#overview)
- [Course Information](#course-information)
- [Project Structure](#project-structure)
- [Models and Methods](#models-and-methods)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Option A — Full Environment (includes JupyterLab)](#option-a--full-environment-includes-jupyterlab)
  - [Option B — Minimal Environment (BYO Jupyter)](#option-b--minimal-environment-byo-jupyter)
- [Running the Notebooks](#running-the-notebooks)
- [Datasets](#datasets)
- [References](#references)
- [Team](#team)
- [License](#license)

## Overview

This project explores **supervised learning** through two complementary problems:

| Task | Dataset | Goal |
|---|---|---|
| **Regression** | California Housing | Predict median house value from census-block-group features |
| **Classification** | Forest CoverType | Predict forest cover type from cartographic features |

For each task the project covers:

1. **Exploratory Data Analysis (EDA)** — descriptive statistics, distribution analysis, correlation, outlier detection.
2. **Preprocessing** — standardization, train/validation/test split, handling class imbalance.
3. **Model Implementation** — models are implemented from scratch (NumPy) alongside scikit-learn benchmarks.
4. **Evaluation** — k-fold cross-validation, statistical hypothesis testing (t-test, Wilcoxon, McNemar), learning curves, residual plots, ROC/PR curves, calibration plots.
5. **Sensitivity & Robustness Analysis** — train/test ratio sensitivity, Gaussian noise injection, missing-data imputation strategies, convergence analysis.
6. **Advanced Topics** — Bayesian regression, Gaussian Process regression, Kernel methods, Bias-Variance analysis, Robust regression (IRLS), Probit model, Laplace approximation, VC dimension analysis, and more.

## Course Information

| | |
|---|---|
| **Course** | Introduction to Machine Learning (Nhập môn Học máy) |
| **Instructor (Theory)** | Dr. Bùi Tiến Lên |
| **Instructor (Lab)** | MSc. Lê Nhựt Nam |
| **Class** | 23_24 |
| **Group** | 13 |
| **Semester** | Semester 2, 2026 |

## Project Structure

```text
supervised-learning-project/
├── code/
│   ├── Part1_Regression/          # Regression task
│   │   ├── notebook.ipynb         # Main notebook (regression)
│   │   ├── notebook.pdf           # Exported notebook (regression)
│   │   ├── dataset.py             # California Housing loader & splitter
│   │   ├── base/                  # Base classes
│   │   ├── linear_regression/     # OLS, MBGD, WLS, FGLS
│   │   ├── regularization/        # Ridge, Lasso, Elastic Net
│   │   ├── nonlinear_basis/       # Polynomial, RBF, Fourier basis
│   │   ├── features_selection/    # Forward, Backward, Lasso selection
│   │   ├── advanced/              # GPR, Kernel Ridge, BLR, IRLS, Bias-Variance
│   │   ├── evaluation/            # Metrics, cross-validation, stat tests
│   │   ├── eda/                   # Exploratory data analysis utilities
│   │   └── ...
│   ├── Part2_Classification/      # Classification task
│   │   ├── notebook.ipynb         # Main notebook (classification)
│   │   ├── notebook.pdf           # Exported notebook (classification)
│   │   ├── dataset.py             # CoverType loader & splitter
│   │   ├── base/                  # Base classes
│   │   ├── logistic_regression/   # Logistic Regression (OvR, OvO, Softmax)
│   │   ├── lda/                   # LDA, QDA, Fisher Discriminant
│   │   ├── perceptron_logreg/     # Perceptron, regularized LR, class-weighted
│   │   ├── advanced/              # Probit, Laplace, Kernel LR, GNB vs LDA, VC dim
│   │   ├── evaluation/            # Metrics, CV, McNemar, ROC, PR, calibration
│   │   ├── eda/                   # Exploratory data analysis utilities
│   │   └── ...
│   └── pyproject.toml             # Package configuration
├── data/
│   └── README.md                  # Dataset documentation
├── report/
│   ├── report.pdf                 # Full project report (Vietnamese)
│   ├── report.tex                 # Main LaTeX source
│   └── ...                        # Other LaTeX sources & assets
├── requirements-full.txt          # All dependencies + JupyterLab
├── requirements.txt               # All dependencies (no JupyterLab)
├── .editorconfig
├── .gitignore
└── README.md                      # ← You are here
```

## Models and Methods

### Regression

| Category | Models |
|---|---|
| Linear | Ordinary Least Squares (Normal Equation), Mini-Batch Gradient Descent, Weighted Least Squares (FGLS) |
| Regularization | Ridge, Lasso, Elastic Net (with λ selection via k-fold CV, regularization path, warm start) |
| Nonlinear Basis | Polynomial, Radial Basis Function (RBF), Fourier Basis |
| Feature Selection | Forward Stepwise, Backward Elimination, Lasso-based |
| Advanced | Gaussian Process Regression, Kernel Ridge Regression, Bayesian Linear Regression, Robust Regression (IRLS with Huber & Student-t), Bias-Variance decomposition |

### Classification

| Category | Models |
|---|---|
| Discriminative | Logistic Regression (One-vs-Rest, One-vs-One, Softmax), Perceptron |
| Generative | Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA) |
| Regularization | L1/L2 regularized Logistic Regression, Class-weighted loss, Stratified K-Fold CV |
| Advanced | Probit Model, Laplace Approximation, Kernel Logistic Regression, Gaussian Naive Bayes vs. LDA comparison, VC Dimension & Structural Risk Minimization |

## Getting Started

### Prerequisites

- **Python 3.10+** (developed with Python 3.14)
- `pip` package manager
- (Recommended) A virtual environment tool (`venv`, `conda`, etc.)

### Option A — Full Environment (includes JupyterLab)

This option installs **all** dependencies including JupyterLab, so you can run the notebooks directly.

```bash
# 1. Clone the repository
git clone https://github.com/kanumi-2005/supervised-learning-project.git
cd supervised-learning-project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies (includes JupyterLab)
pip install -r requirements-full.txt

# 4. Launch JupyterLab
jupyter lab
```

After JupyterLab opens in your browser, navigate to `code/Part1_Regression/notebook.ipynb` or `code/Part2_Classification/notebook.ipynb`.

### Option B — Minimal Environment (BYO Jupyter)

Use this if you already have Jupyter Lab/Notebook installed externally (e.g., via VS Code, system-wide Jupyter, or another environment).

```bash
# 1. Clone the repository
git clone https://github.com/kanumi-2005/supervised-learning-project.git
cd supervised-learning-project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies (no JupyterLab)
pip install -r requirements.txt

# 4. Register the venv as a Jupyter kernel
python -m ipykernel install --user --name=supervised-learning --display-name "Python (Supervised Learning)"
```

Then open your external Jupyter Lab/Notebook (or VS Code), select the **"Python (Supervised Learning)"** kernel, and run the notebooks.

> **Note:** The `requirements.txt` already includes `ipykernel`, so step 4 only registers it with Jupyter — no extra install is needed.

## Running the Notebooks

1. Open `code/Part1_Regression/notebook.ipynb` for the **regression** task.
2. Open `code/Part2_Classification/notebook.ipynb` for the **classification** task.
3. Run all cells from top to bottom — each notebook is self-contained and reproducible (random seeds are fixed).
4. Datasets are fetched automatically via `scikit-learn` on first run (see [`data/README.md`](data/README.md) for details).

## Datasets

| Dataset | Task | Samples | Features | Source |
|---|---|---|---|---|
| [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) | Regression | 20,640 | 8 | `sklearn.datasets.fetch_california_housing` |
| [Forest CoverType](https://scikit-learn.org/stable/datasets/real_world.html#forest-covertypes) | Classification | 581,012 | 54 | `sklearn.datasets.fetch_covtype` |

For detailed dataset descriptions, see [`data/README.md`](data/README.md).

## References

- Pace, R. K., & Barry, R. (1997). Sparse spatial autoregressions. *Statistics and Probability Letters*, 33(3), 291–297.
- Blackard, J. (1998). Covertype. *UCI Machine Learning Repository*. DOI: [10.24432/C50K5N](https://doi.org/10.24432/C50K5N)

## Team

| Name | Student ID | Role |
|---|---|---|
| Hoàng Ngọc Phú | 23120010 | Team Lead |
| Hoàng Ngọc Quí | 23120077 | Member |
| Nguyễn Duy Bảo | 23120113 | Member |

## License

This project is developed for educational purposes as part of the Introduction to Machine Learning course at VNUHCM-US.
