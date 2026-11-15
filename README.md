# Introduction to Data Science and Engineering (IDSE) Project

**Course:** Introduction to Data Science and Engineering (IDSE)  
**Authors & Team Members:**
- **Hemanthraj M** (Roll No: EC24B1098)
- **Adithya Ajay** (Roll No: EC24B1099)

---

## Overview

> **This repository contains two folders - Game and Model Analysis for IDSE Project.**

This repository encompasses our complete submission for the Introduction to Data Science and Engineering (IDSE) course. It bridges theoretical machine learning concepts with practical engineering implementations through two main components:

1. **[Game](./Game/) — Heart Run: Bias-Variance Dispatch**: An interactive, single-file educational web game designed to teach the bias-variance tradeoff, regularization penalties ($\lambda$), risk aversion ($\alpha$), and 5-fold cross-validation.
2. **[Model Analysis](./Model/) — Student Exam Score Prediction**: An empirical machine learning study comparing unregularized Ordinary Least Squares (OLS) linear regression against regularized regression models (Lasso, Ridge, and ElasticNet) to predict student final exam scores based on study habits, lifestyle, and demographic features.

---

## Repository Structure

```text
IDSE-Project/
│
├── README.md                                          # Overall project documentation (this file)
│
├── Game/                                              # Part 1: Interactive Educational Game
│   ├── Heart Run_ Bias-Variance Dispatch_final.html  # Standalone single-file HTML/JS/CSS game
│   ├── Feedback Form.csv                              # Player survey responses & experiment logs
│   └── README.md                                      # Detailed game documentation & mechanics
│
└── Model/                                             # Part 2: Model Analysis & Regression Study
    ├── IDSE_project.ipynb                             # Jupyter notebook with end-to-end ML pipeline
    ├── IDSE_Report_final.pdf                          # Comprehensive technical research report
    ├── student_performance_dataset.csv                # Dataset (3,000 records, 32 attributes)
    ├── Requirements.txt                               # Python dependencies for reproduction
    └── README.md                                      # Documentation & execution instructions
```

---

## 1. Game: Heart Run — Bias-Variance Dispatch

### Concept & Objective
**Heart Run** places the player in the role of an emergency medical dispatcher. An ambulance carrying a vital donor heart must travel from the **City Trauma Center (0 km)** to the **Rural General Hospital (48 km)**. The baseline trip takes **65 minutes**.

To minimize travel time, the player tunes a polynomial regression model based on historical route data to predict time savings across three distinct routes:
- **Route A (Mountain Pass - 42 km):** Curved route with sharp hairpins. Fast on straights, slow on corners (inflexible linear models miss the curve).
- **Route B (Valley Highway - 48 km):** Multi-lane highway with construction delays and a high-leverage VIP convoy outlier that overfitted models memorize.
- **Route C (Village Bypass - 54 km):** Rural route prone to unpredictable variance from weekly village markets and flash rainstorms.

### Key Machine Learning Concepts Taught
- **Underfitting (High Bias):** Overly simple models (low polynomial degree) fail to capture genuine route patterns.
- **Overfitting (High Variance):** Excessively complex models (high polynomial degree, $\lambda = 0$) memorize random noise and outliers.
- **Regularization ($\lambda$):** Ridge penalty coefficient shrinking weights to maintain smooth, generalizable predictions.
- **Risk Aversion ($\alpha$):** Route selection penalized by cross-validation variance across folds:
  $$\text{Score} = \hat{y}(\text{time saved}) - \alpha \cdot \sigma_{\text{CV}}$$
- **5-Fold Cross-Validation:** Partitioned validation assessing model stability across folds (F1–F5).
- **Confusion Matrix & Metrics:** Out-of-fold classification measuring True/False Positives and Negatives for whether trips achieved $\ge 10$ minutes in time savings.

### Technical Implementation
- **Architecture:** Zero-dependency, client-side application contained entirely within a single `.html` file.
- **Numerical Engine:** Closed-form polynomial regression via the normal equations solved using Gauss-Jordan elimination with partial pivoting in vanilla JavaScript.
- **Visuals:** Responsive SVG animations, interactive trajectory tracking, and live-rendered polynomial curves.
- **Player Feedback:** Field-tested feedback logs preserved in `Feedback Form.csv`.

---

## 2. Model Analysis: Student Performance Regression Study

### Research Objective
This study investigates factors influencing student academic achievement and evaluates whether regularized regression techniques can mitigate overfitting and improve generalization when modeling complex, multi-factorial student performance data.

### Dataset Overview
- **Source:** Kaggle Student Performance Dataset (`student_performance_dataset.csv`).
- **Dimensions:** 3,000 samples and 32 attributes spanning study habits, class attendance, digital distractions, sleep quality, mental stress, and socioeconomic indicators.
- **Target Variable:** `final_exam_score` (continuous scale from 0 to 100).
- **Data Hygiene:** Removed target leakage columns (`performance_category`), handled missing values, and encoded categorical variables into a 43-feature design matrix.

### Methodology & ML Pipeline
1. **Exploratory Data Analysis (EDA):** Correlation heatmaps, feature distributions, and engineered interaction terms.
2. **Feature Preprocessing:** `StandardScaler` feature normalization paired with `PolynomialFeatures`.
3. **Model Selection:**
   - **Baseline:** Ordinary Least Squares (OLS) Linear Regression.
   - **Ridge Regression ($L_2$ regularization):** Shrinks coefficients uniformly to handle multicollinearity.
   - **Lasso Regression ($L_1$ regularization):** Enforces sparsity through automated feature selection.
   - **ElasticNet:** Convex combination of $L_1$ and $L_2$ penalties.
4. **Hyperparameter Optimization:** Exhaustive 5-fold cross-validated grid search (`GridSearchCV`) over polynomial degrees, penalty weights ($\alpha$), and mixing ratios (`l1_ratio`).

### Results Summary
Evaluated on a held-out test set (20% split, fixed seed):

| Model | Adjusted $R^2$ | RMSE | MAE | Rank |
| :--- | :---: | :---: | :---: | :---: |
| Ordinary Least Squares (Baseline) | 0.6911 | 10.282 | 8.230 | 4th |
| Ridge Regression ($L_2$) | 0.6914 | 10.277 | 8.226 | 3rd |
| ElasticNet ($L_1 + L_2$) | 0.6948 | 10.221 | 8.152 | 2nd |
| **Lasso Regression ($L_1$)** | **0.7465** | **9.314** | **7.406** | **1st (Best)** |

**Key Finding:** **Lasso regression** produced the highest explanatory power and lowest prediction error, effectively eliminating non-informative features and identifying attendance, consistent study hours, and stress management as the primary drivers of final exam outcomes.

For complete statistical analyses, residual plots, and pedagogical insights, consult [`Model/IDSE_Report_final.pdf`](./Model/IDSE_Report_final.pdf).

---

## Getting Started

### Running the Educational Game
No installation, web server, or build step is required:
1. Navigate to the `Game/` directory.
2. Open [`Heart Run_ Bias-Variance Dispatch_final.html`](./Game/Heart Run_ Bias-Variance Dispatch_final.html) in Google Chrome, Mozilla Firefox, Microsoft Edge, or Safari.
3. Alternatively, access the live build: [Live Game Demo](https://claude.ai/artifact/XSNKKtdNm624BgUDLBkYvo).

### Running the Model Analysis Notebook
To reproduce the regression experiments and figures:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HemanthrajM06/IDSE-Project.git
   cd IDSE-Project
   ```

2. **Set up a Python virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r Model/Requirements.txt
   ```

4. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook Model/IDSE_project.ipynb
   ```
   Select **Run All Cells** to execute the pipeline from data preprocessing to hyperparameter tuning and model evaluation.

---

## Authors & Acknowledgments

- **Hemanthraj M** — [GitHub Profile](https://github.com/HemanthrajM06) — Roll No: `EC24B1098`
- **Adithya Ajay** — [GitHub Profile](https://github.com/Adithya1523) — Roll No: `EC24B1099`

Developed as part of the coursework for **Introduction to Data Science and Engineering (IDSE)**.
