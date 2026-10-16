# IT5022 Fundamentals of Machine Learning — Assignment 01

Predicting student **dropout, continued enrolment, or graduation** from administrative and
first-year academic records, and comparing two classical (non-deep-learning) supervised
algorithms — **Linear Regression** and **Random Forest** — on accuracy and on what each one can
tell a student-support office.

> Group assignment, team of two. The report is the marked deliverable; this repository holds the
> code that validates it. The notebook `notebooks/Assignment.ipynb` runs top-to-bottom with all
> outputs embedded.

---

## Problem

The Polytechnic Institute of Portalegre wants to know, early enough to act, which students are at
risk of leaving. For each enrolled undergraduate we predict one of three outcomes assigned at the
end of the course's normal duration:

| Class | Meaning | Share |
|---|---|---|
| **Graduate** | finished on time | 49.9% |
| **Dropout** | left before finishing | 32.1% |
| **Enrolled** | still enrolled past the normal duration (behind schedule) | 17.9% |

This is **supervised classification**: every row carries a known `Target` label, so a learning
signal exists, and the outcome is an unordered label rather than a quantity. The strong class
imbalance is the central modelling challenge — a model that always answers *Graduate* scores
49.9% accuracy and is useless, so we lead on **macro-F1, balanced accuracy and MCC**, not accuracy.

A second framing, **Dropout vs rest**, is used for the threshold, ROC/PR and resampling analysis
that three classes do not support cleanly. A third, **first-semester-only**, is an honest
*early-warning* model that withholds second-semester columns because they are not known in time to
intervene.

## Dataset

**UCI Machine Learning Repository #697 — _Predict Students' Dropout and Academic Success_.**

- Link: <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>
- DOI: <https://doi.org/10.24432/C5MC89> · Licence: CC BY 4.0
- Size: **4,424 rows × 36 features + 1 target**; **no missing values and no duplicates** (the
  authors repaired anomalies and outliers before release — we claim representation and reduction
  work, not cleaning).
- Features span demographic, socio-economic, macro-economic and academic groups; nine are registry
  **codes stored as integers** (one-hot encoded), the rest are binary flags or continuous numbers.
- The dataset is **downloaded automatically** by the notebook via the `ucimlrepo` package
  (`fetch_ucirepo(id=697)`). An offline copy is committed at [`data/students_dropout.csv`](data/students_dropout.csv)
  and is used as an automatic fallback if the download fails.
- Citation: Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predicting Student
  Dropout and Academic Success.* **Data, 7**(11), 146. <https://doi.org/10.3390/data7110146>

## Methodology

A single stratified **80/20 split** (3,539 training / 885 test rows, `random_state=42`). All model
selection — feature screening, hyper-parameter tuning, imbalance strategy and decision threshold —
is cross-validated on the **training set with stratified 5-fold CV**. The **test set is sealed and
evaluated exactly once**, at the end. Every data-dependent transform (one-hot encoding, scaling,
SMOTE) lives inside a scikit-learn `Pipeline` / `ColumnTransformer` so it is re-fit inside each
fold and never leaks information from validation or test data.

**Preprocessing & feature engineering** — one-hot encoding for the nine integer registry codes,
pass-through for binary flags, standardisation for continuous features, plus five engineered
features (first-/second-semester approval rates, grade trend, total approved units). Relevance is
screened with correlation, ANOVA F-test, mutual information and chi-square.

**Algorithms**

| # | Model | Type | Why |
|---|---|---|---|
| 1 | **Linear Regression** | parametric, linear | Interpretable baseline. The ordered outcome (Dropout < Enrolled < Graduate) is modelled as an ordinal *success score* (0/1/2); the prediction is rounded and clipped to a class. Demonstrates the full Lab 3 / Lab 4 toolset (cost function, gradient descent, normal equation, scikit-learn verification, Ridge). |
| 2 | **Random Forest** | non-parametric, bagging ensemble | Captures the non-linear interactions a linear model cannot (e.g. low approval rate *and* unpaid fees). Tuned with grid search: `class_weight=balanced, max_depth=None, max_features=sqrt, min_samples_leaf=2, n_estimators=300`. |

**Evaluation** — accuracy, balanced accuracy, macro-F1, Matthews correlation coefficient (MCC),
per-class recall, and (for probabilistic models) ROC-AUC. Class imbalance is handled with
`class_weight` and SMOTE; the Dropout-vs-rest framing adds threshold analysis, ROC and PR curves.

## Results

### Cross-validation (training set, stratified 5-fold)

| Model | Variant | Accuracy | Balanced acc | Macro-F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|---|
| Random Forest | all features | **0.772** | **0.727** | **0.726** | **0.632** | **0.891** |
| Random Forest | 1st-sem only | 0.748 | 0.697 | 0.698 | 0.591 | 0.871 |
| Linear Regression | all features | 0.705 | 0.686 | 0.672 | 0.561 | n/a |
| Linear Regression | 1st-sem only | 0.639 | 0.630 | 0.615 | 0.479 | n/a |
| Majority-class baseline | — | 0.499 | 0.333 | 0.222 | 0.000 | — |

### Sealed test set (885 students, evaluated once)

| Model | Accuracy | Balanced acc | Macro-F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.758** | **0.719** | **0.714** | **0.614** | **0.889** |
| Linear Regression | 0.690 | 0.682 | 0.663 | 0.547 | n/a |

**Early-warning view (Dropout vs rest, test set):** ROC-AUC **0.916**, PR-AUC **0.871**; at the
default 0.50 threshold the Random Forest catches **81% of dropouts (recall 0.806) at 0.809
precision**.

**Conclusion:** the **Random Forest is the recommended model**; Linear Regression is a stable,
interpretable baseline. The minority *Enrolled* class is hardest for both — which is exactly why we
report macro-averaged metrics rather than accuracy alone.

## Repository structure

```
ML-assignment-01/
├── data/
│   └── students_dropout.csv        # offline copy of UCI #697 (notebook also auto-downloads it)
├── notebooks/
│   ├── Assignment.ipynb            # MAIN submission notebook — 6 sections, fully executed
├── requirements.txt                # Python dependencies
├── members.txt                     # group members: IDs, names, emails
├── submission.txt                  # dataset / GitHub / YouTube links
├── .gitignore
└── README.md                       # this file
```

### What `Assignment.ipynb` contains

1. **Data Loading** — fetch UCI #697 (with CSV fallback), audit, target distribution.
2. **Preprocessing & Feature Screening** — column groups, engineered features, encoding, scaling,
   the leakage-safe pipeline, PCA/feature-selection variants, correlation + ANOVA/MI/chi-square.
3. **Model 1 — Linear Regression** — ordinal-score framing, baseline, gradient descent vs normal
   equation vs scikit-learn, Ridge, coefficient table, regression→classification.
4. **Model 2 — Random Forest** — single-tree overfitting, tuning, feature importances, variants,
   early-warning and binary framings.
5. *(continues within sections 4–5 above)*
6. **Model Comparison & Evaluation (Lab 8 style)** — CV model selection, class-imbalance handling
   (class-weight / SMOTE), **final sealed-test evaluation**, ROC/PR, threshold analysis, conclusion.

## How to run — step by step

### 1. Prerequisites
- **Python 3.10+** and `pip` installed (`python --version`).
- Internet access on first run (the dataset downloads via `ucimlrepo`). If offline, the notebook
  falls back to `data/students_dropout.csv`.

### 2. Get the code
```bash
git clone https://github.com/LahiruSandeepa-LK/ML-assignment-01.git
cd ML-assignment-01
```

### 3. Create an isolated environment (recommended)
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch Jupyter **from the repository root**
```bash
jupyter notebook        # or: jupyter lab
```
Running from the repo root matters: the offline-data fallback path (`data/students_dropout.csv`) is
resolved relative to the project root.

### 6. Run the notebook
- Open **`notebooks/Assignment.ipynb`**.
- Menu: **Kernel → Restart & Run All** (or **Run → Run All Cells**).
- All sections execute in order (data load → preprocessing → Model 1 → Model 2 → comparison).
- **Runtime:** a few minutes on a typical laptop — the Random Forest grid search and the
  cross-validated comparisons are the slow steps.
- **Reproducibility:** every split, model and resampler uses `random_state=42`, so results match
  the tables above.

### 7. (Optional) Run headless from the command line
```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 notebooks/Assignment.ipynb
```

### Troubleshooting
- **`ModuleNotFoundError`** — the environment is not active or deps aren't installed; repeat steps 3–4.
- **Dataset download fails / no internet** — ensure `data/students_dropout.csv` is present; the
  loader uses it automatically.
- **`imbalanced-learn` / SMOTE import error** — `pip install imbalanced-learn` (already in
  `requirements.txt`).

## Individual contribution

- **Lahiru Sandeepa (MS26929378)** — data loading, preprocessing & feature engineering, Model 1
  (Linear Regression), and the model comparison / final evaluation (Section 6).
- **Kusal Dissanayake (MS26929064)** — Model 2 (Random Forest): decision-tree baseline, tuning,
  feature importances, feature-variant and early-warning analysis.

Both members commit under their own name throughout the project history.

## Licence

Code released under the MIT Licence. The dataset is © its original authors and distributed by the
UCI Machine Learning Repository under CC BY 4.0.
