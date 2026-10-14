# IT5022 Fundamentals of Machine Learning — Assignment 01

Predicting student **dropout, continued enrolment, or graduation** from administrative and
first-year academic records, and comparing two classical (non-deep-learning) supervised
algorithms — **Logistic Regression** and **Random Forest** — on accuracy and on what each one
can tell a student-support office.

> Group assignment, team of two. The report is the marked deliverable; this repository holds
> the code that validates it.

---

## Problem

The Polytechnic Institute of Portalegre wants to know, early enough to act, which students are
at risk of leaving. For each enrolled undergraduate we predict one of three outcomes assigned at
the end of the course's normal duration:

| Class | Meaning | Share |
|---|---|---|
| **Graduate** | finished on time | 49.9% |
| **Dropout** | left before finishing | 32.1% |
| **Enrolled** | still enrolled past the normal duration (behind schedule) | 17.9% |

This is **supervised classification**: every row carries a known `Target` label, so a learning
signal exists, and the outcome is an unordered label rather than a quantity. The strong class
imbalance is the central modelling challenge — a model that always answers *Graduate* scores
49.9% accuracy and is useless, so we lead on **macro-F1, balanced accuracy and MCC**, not
accuracy.

A second framing, **Dropout vs rest**, is used for the threshold, ROC/PR and resampling analysis
that three classes do not support cleanly. A third, **first-semester-only**, is an honest
*early-warning* model that withholds second-semester columns because they are not known in time
to intervene.

## Dataset

**UCI Machine Learning Repository #697 — _Predict Students' Dropout and Academic Success_.**

- Link: <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>
- DOI: <https://doi.org/10.24432/C5MC89> · Licence: CC BY 4.0
- Size: **4,424 rows × 36 features + 1 target**; **no missing values** (the authors repaired
  anomalies and outliers before release — we claim representation and reduction work, not cleaning).
- Features span demographic, socio-economic, macro-economic and academic groups; nine are
  registry **codes stored as integers** (one-hot encoded), the rest binary flags or continuous.
- Citation: Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predicting Student
  Dropout and Academic Success.* **Data, 7**(11), 146. <https://doi.org/10.3390/data7110146>

See [`data/raw/README.md`](data/raw/README.md) for retrieval, integrity checks and known
discrepancies with the descriptor paper.

## Methodology

A single stratified **80/20 split** (3,539 development / 885 test rows, `random_state=42`) is
fixed in `src/data.py` and reused everywhere. All model selection is cross-validated on the
development set; **the test set is evaluated exactly once, in notebook 05.**

**Preprocessing** (all inside scikit-learn `Pipeline`s, so every statistic is fitted on training
folds only — no leakage):

- **Encoding** — nine nominal code columns one-hot encoded with rare levels (<1%) folded into an
  `infrequent` bucket learned on training data only.
- **Standardisation** — continuous columns centred and scaled; required by Logistic Regression's
  gradient solver, harmless to Random Forest, and kept for both so the two consume an identical matrix.
- **Feature engineering** — four leakage-free, row-wise features (`approval_rate_1st/2nd`,
  `grade_trend`, `total_approved`).
- **Dimensionality reduction, both kinds** — *selection* (ANOVA F / mutual information / chi-square)
  keeps original columns; *extraction* is **standardised PCA** on the continuous block (23 columns →
  13 components, 95% variance), fitted inside the pipeline.

**Four feature variants** (`all`, `reduced`, `pca`, `first_sem`) are run through both algorithms so
the effect of the feature set is separated from the effect of the algorithm.

**Algorithms**

- **Logistic Regression** (notebook 03) — the interpretable model. Tuned `C` (inverse L2) and
  `class_weight` on macro-F1; coefficients read as odds ratios. Imbalance handled with balanced
  class weights, compared against SMOTE on the binary task.
- **Random Forest** (notebook 04) — the higher-capacity model. Tuned depth, leaf size, tree count,
  `max_features` and `class_weight`; feature importances read against the screening and the paper.

## Results

### Cross-validation (development set, 5-fold)

| Model | macro-F1 | balanced acc. | MCC | Enrolled recall | Graduate recall |
|---|---|---|---|---|---|
| Majority baseline | 0.222 | 0.333 | 0.000 | 0.00 | 1.00 |
| Logistic Regression (all) | 0.714 | 0.721 | 0.611 | **0.608** | 0.821 |
| Random Forest (all) | **0.726** | **0.727** | **0.632** | 0.578 | **0.859** |

### Sealed test set (885 rows, touched once — notebook 05)

**Three-class**

| Model | accuracy | macro-F1 | MCC | ROC-AUC (OvR) | Enrolled recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.722 | 0.683 | 0.570 | 0.885 | **0.616** |
| Random Forest | **0.758** | **0.714** | **0.614** | **0.889** | 0.598 |

**Binary — Dropout vs rest**

| Model | accuracy | MCC | ROC-AUC | PR-AUC | Dropout recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.860 | 0.685 | **0.917** | **0.882** | **0.813** |
| Random Forest | **0.877** | **0.717** | 0.916 | 0.871 | 0.806 |

**Reading it.** Random Forest is a little ahead on overall accuracy and macro-F1; Logistic
Regression is stronger on the minority *Enrolled* class and marginally better at *ranking* dropout
risk (ROC/PR). The margins on 885 rows are small, so the recommendation turns on **use, not score**:
Logistic Regression is recommended as the operational early-warning model because it gives a
support office a reason it can act on and defend, with Random Forest kept as the accuracy
benchmark. Standardised PCA matches all-features on score for both models while forfeiting that
interpretability — reported as the required reduction step, not the chosen model. The
early-warning (first-semester) variant costs ~3 points of macro-F1, the honest price of acting a
semester sooner.

## Repository structure

```
ML-assignment-01/
├── data/
│   ├── raw/              data.csv (UCI 697) + source notes
│   └── processed/        CV + final-test results, tuned-param JSON
├── notebooks/
│   ├── 01_data_understanding.ipynb   EDA, class balance, correlations, clustering appendix
│   ├── 02_preprocessing.ipynb        split, encoding, engineering, screening, PCA scree
│   ├── 03_logistic_regression.ipynb  Algorithm 1 — tuning, odds, PCA, binary, early-warning
│   ├── 04_random_forest.ipynb        Algorithm 2 — trees→forest, tuning, importances
│   └── 05_comparison.ipynb           LR vs RF, the single sealed-test evaluation
├── src/
│   ├── data.py           split, column groups, engineered features
│   ├── preprocess.py      ColumnTransformer / Pipeline, PCA, feature variants
│   ├── evaluation.py      shared metrics so both models' tables concatenate
│   └── viz.py             colour-blind-safe palette and figure style
├── reports/
│   ├── figures/          all generated figures
│   └── ACCESSIBILITY.md   figure accessibility audit (WCAG-aligned)
├── requirements.txt
├── members.txt · submission.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook            # run notebooks 01 → 05 in order
```

Notebook 01 downloads and caches `data/raw/data.csv` on first run. Notebooks 03 and 04 each save
their cross-validation table and tuned parameters to `data/processed/`; notebook 05 reads both and
produces the final-test numbers. `random_state=42` throughout makes every result reproducible.

## Individual contribution

- **Lahiru Sandeepa (MS26929378)** — Logistic Regression (notebook 03), the shared evaluation and
  the comparison notebook (05), standardised PCA, the accessible figure styling, and this README.
- **Kusal Dissanayake (MS26929064)** — data understanding (01), preprocessing and feature
  engineering (02), and Random Forest (04).

Both members commit under their own name throughout the project history.

## Accessibility

All figures use the colour-blind-safe Okabe–Ito palette and never rely on hue alone to carry
meaning (confusion matrices are annotated with counts; lines carry distinct markers and legends;
the three outcome classes keep one fixed colour everywhere). See
[`reports/ACCESSIBILITY.md`](reports/ACCESSIBILITY.md).

## Licence

Code released under the MIT Licence. The dataset is © its original authors and distributed by the
UCI Machine Learning Repository under CC BY 4.0.
