# Heart Run: Bias-Variance Dispatch

An interactive, single-file web game that teaches **underfitting vs. overfitting (the bias-variance tradeoff)** and **5-fold cross-validation**. You tune a polynomial regression model that predicts how many minutes each of three routes will save, then dispatch an ambulance carrying a donor heart to the hospital.

Live version: https://claude.ai/artifact/XSNKKtdNm624BgUDLBkYvo

---

## Learning objectives

After playing, you should be able to:

1. Explain **underfitting (high bias)**: a model that is too simple misses real structure, so training and validation error are both high.
2. Explain **overfitting (high variance)**: a model that is too flexible memorises noise, so training error is low but validation error is high.
3. Describe how **model complexity (degree)**, **regularization (λ)** and **risk aversion (α)** each shift the tradeoff.
4. Read **5-fold cross-validation** results (per-fold RMSE, mean ± std) and use them to diagnose a model.
5. Read a **confusion matrix** and its accuracy, precision and recall for out-of-fold predictions.

---

## Story

A donor heart must travel from the **City Trauma Center (0 km)** to **Rural General Hospital (48 km)**. The baseline trip takes **65 minutes**. An animated heart narrator briefs you, and your goal is to tune the model so the ambulance takes the fastest route. There are no levels and no countdown timer. You chase a **Personal Best** and a larger **Time Reduction**.

### Routes

| Route | Distance | Character | Trade-off |
|---|---|---|---|
| A: Mountain Pass | 42 km | Curved, hairpin bends | Fast on straights, slow on hairpins. A straight-line model misses the dip. |
| B: Valley Highway | 48 km | Multi-lane road with a river and bridge | Construction is a steady slowdown. One VIP convoy run is an extreme outlier that overfit models memorise. |
| C: Village Bypass | 54 km | Rural road past farmland | Weekly market spike and heavy rain create high, unpredictable variance. |

Hazard pins on the map show a tooltip with the segment length, mean time reduction (μ) and delay variance (σ), computed from the data.

---

## How to play

1. Pick a route tab (A, B or C) to see its historical runs and the live fitted curve.
2. Move the three sliders:
   - **Model complexity (degree 1 to 10):** how much the curve can bend.
   - **Regularization λ (0.0 to 10.0):** penalty on large coefficients, which smooths the curve.
   - **Risk aversion α (0.0 to 2.0):** penalty on routes whose forecast swings between folds.
3. Tap the ⓘ next to each slider name to read its tradeoffs.
4. Click **Run 5-Fold CV** to step through all five folds and see the diagnosis badge.
5. Choose **Route to drive**: *Model's pick*, or force **A**, **B** or **C**.
6. Click **Dispatch Ambulance**. The ambulance drives the route while a km counter ticks up, and the real travel time is revealed.
7. Submit your run (name, route, time, hyperparameters and matrix metrics) through the end-screen form or by email.

Route score used by the model: `score = ŷ(time saved today) − α · σ_CV`, where σ_CV is the standard deviation of the five fold-models' predictions for that route.

---

## What the numbers mean

- **Train RMSE:** typical error (in minutes) on the runs the model trained on.
- **Val RMSE (fold):** typical error on the currently held-out fold.
- **F1 to F5:** the 60 runs are split into 5 folds of 12 runs (4 per route). For fold *k*, the model trains on the other 48 runs and is tested on the 12 held-out runs. Each number is that fold's RMSE in minutes. F2 holds the heavy-rain runs and F3 holds the VIP convoy outlier.
- **CV RMSE mean ± std:** average and spread of the five fold RMSEs.
- **Diagnosis badge:**
  - *High Variance / Overfit* when CV RMSE is more than 1.6 times the training RMSE.
  - *High Bias / Underfit* when training RMSE is high (above 8.9) and CV RMSE is close to it (ratio below 1.3).
  - *Balanced Sweet Spot* otherwise.

### Confusion matrix

For all 60 runs, an out-of-fold prediction is made by a model that never saw that run, using the current degree and λ. The question asked is: **was the trip fast (saved +10 min or more)?**

- **True positive / true negative:** the model was right.
- **False positive:** predicted fast, actually slow (the ambulance gets stuck).
- **False negative:** predicted slow, actually fast (a good road was ignored).
- Accuracy, precision and recall are shown below the grid.

---

## Implementation

Everything runs client-side in one HTML file. No backend, build step or package install is needed.

- **Data:** 60 synthetic historical runs (20 per route) from a seeded random generator. Each run has a road condition index `x` in [0, 1], minutes saved `y`, a fold id and a tag (hairpin, construction, vip, market, rain and so on). Route outcomes are never hard-coded.
- **Regression:** polynomial least squares via the normal equations, solved with Gauss-Jordan elimination with partial pivoting. Inputs are scaled to [-1, 1]. Ridge (λ) is applied to every coefficient except the intercept, with an internal scale factor of 0.05 so the 0 to 10 slider range stays useful.
- **Cross-validation:** each route is fit separately per fold. Fold RMSE is pooled across routes.
- **Today's conditions:** fixed at condition index 0.40 with heavy rain on the bypass. True minutes saved today are A: +2, B: +14, C: −12, so the best achievable trip is about 51 minutes.
- **Animation:** the ambulance follows the selected SVG `<path>` using `getPointAtLength`, with easing and smoothed heading rotation.
- **Personal best:** stored in the browser's `localStorage` (a failure to read or write it is ignored).

### Suggested settings to explore

| Degree | λ | α | What you should see |
|---|---|---|---|
| 1 | 0 | 0 | Underfit. The model favors Route A and loses time on the hairpins. |
| 2 | 0 | 0 | Still underfit. Can favor the flooded bypass. |
| 4 to 6 | about 2 to 3 | 0 to 0.5 | Balanced. The model picks Route B and saves about 14 minutes. |
| 10 | 0 | 0 | Overfit. Wild curve, very large CV RMSE. |

---

## Submitting results

The end-of-run screen sends: name, degree, λ, α, route, travel time, time reduction, confusion matrix counts (TP, FP, FN, TN), accuracy, precision, recall, training RMSE, CV RMSE and optional comments.

- **Send to Google Form:** opens the linked Google Form in a new tab with every answer pre-filled. The player presses **Submit** in the form. The page cannot submit silently.
- **Email results:** opens the player's mail app with the same data addressed to `hemrohinth@gmail.com`.

### Changing the Google Form

The form link is configured in the `GF` constant in the script:

```js
const GF = { id: '<form id from the /d/e/<id>/viewform link>', e: { name: '<entry id>', degree: '<entry id>', ... } };
```

To re-link a different form, create a short-answer question for each field, use **Get pre-filled link** to find the `entry.NNNN` ids, and update `id` and `e`.

---

## Known limitations

- The "today" conditions and route curves are fixed, so every playthrough faces the same day.
- With high α (about 1 or more), the model can tip toward Route A even at sensible degrees.
- At high degree with λ = 0, the model may still pick Route B by chance. The curve, CV RMSE and badge show the overfit rather than the travel time.
- Pre-filled Google Forms need the player to press Submit. Silent submission from the page is blocked.

---

## Tech

HTML, CSS and vanilla JavaScript, with custom inline SVG for the map, heart narrator, ambulance and hazard pins. Fonts: Inter and JetBrains Mono (Google Fonts).
