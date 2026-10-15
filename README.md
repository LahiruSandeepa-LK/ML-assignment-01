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

A single stratified **80/20 split** (3,539 development / 885 test rows, `random_state=42`). All model selection is cross-validated on the development set; 

**Algorithms**

## Results

### Cross-validation (development set, 5-fold)



### Sealed test set 



## Repository structure

```
ML-assignment-01/
├── data/
│   ├── student_dropout.csv
├── notebooks/
│   ├── Assignment.ipynb
├── requirements.txt
├── members.txt · submission.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook            # run notebooks 
```


## Individual contribution

- **Lahiru Sandeepa (MS26929378)** — 
- **Kusal Dissanayake (MS26929064)** — 
Both members commit under their own name throughout the project history.

## Licence

Code released under the MIT Licence. The dataset is © its original authors and distributed by the
UCI Machine Learning Repository under CC BY 4.0.
