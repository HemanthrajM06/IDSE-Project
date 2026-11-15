# Predicting Student Final Exam Scores: A Regularised Regression Study

**Subject:** Introduction to Data Science and Engineering (IDSE)

**Group members**

| Name | Roll number |
| --- | --- |
| Hemanthraj M | EC24B1098 |
| Adithya Ajay | EC24B1099 |

## 1. Project summary

This project predicts a student's final exam score (0-100) from study habits, digital distraction, stress, sleep and demographic features. It compares an unregularised baseline (Linear Regression) with three regularised models (Lasso, Ridge, ElasticNet) and identifies which factors are most associated with the score.

**Result:** Lasso performed best on the held-out test set.

| Model | Adjusted R² | RMSE | MAE |
| --- | --- | --- | --- |
| Linear Regression | 0.6911 | 10.282 | 8.230 |
| **Lasso** | **0.7465** | **9.314** | **7.406** |
| Ridge | 0.6914 | 10.277 | 8.226 |
| ElasticNet | 0.6948 | 10.221 | 8.152 |

The full analysis, figures and reasoning are in `IDSE_Regression_Report.pdf`.

## 2. Dataset

- **Source:** Student Performance Dataset from Kaggle. Add the dataset link and author here.
- **Size:** 3,000 rows and 32 columns (19 integer, 7 float, 6 string).
- **Target:** `final_exam_score`.
- **Dropped column:** `performance_category`, because it is derived from the target and would cause data leakage.
- **Missing values:** 60 each in `social_media_hours`, `focus_score` and `sleep_hours`. Rows with missing values are dropped.

Download the CSV from Kaggle and place it in the `data/` folder (see the structure below).

## 3. Method

1. Remove duplicates and rows with missing values.
2. One-hot encode string columns with `pandas.get_dummies` (43 features after encoding).
3. Explore the data: score distribution, performance categories, mental state, study hours, correlation heatmap and engineered features.
4. Split into training and test sets with `train_test_split` (fixed `random_state`).
5. Build one pipeline per model: `StandardScaler`, then `PolynomialFeatures`, then the estimator.
6. Tune `alpha`, polynomial `degree` (and `l1_ratio` for ElasticNet) with `GridSearchCV` on the training data.
7. Evaluate on the test set with adjusted R², RMSE and MAE, and inspect residual plots.

## 4. Repository structure

Adjust the names below to match your files.

```
.
├── README.md
├── requirements.txt
├── IDSE_Regression_Report.pdf
├── data/
│   └── student_performance_dataset.csv
└── notebook.ipynb
```

## 5. How to run

1. Clone the repository and open a terminal in its folder.
2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the dataset CSV in `data/` as shown above.
5. Open `notebook.ipynb` in Jupyter or VS Code and choose **Run All**. The notebook runs from top to bottom with no manual changes. Random seeds are fixed, so the reported figures can be reproduced.

## 6. Requirements

Typical contents of `requirements.txt` (pin the exact versions you used):

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

## 7. Limitations

- The results show association, not causation.
- Several inputs are likely self-reported, so measurement bias is possible.
- Linear models are unbounded, so predictions can fall outside the 0-100 range (this happened with the baseline). Clipping predictions is a simple fix.
- Next steps: feature selection (`SelectKBest`, PCA) and tree-based models (Random Forest, XGBoost or LightGBM).
