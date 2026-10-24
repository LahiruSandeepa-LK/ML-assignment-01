# Predicting Student Dropout and Academic Success

IT5022 Fundamentals of Machine Learning, group assignment by Kusal Dissanayake and Lahiru Sandeepa.

We predict whether a student will drop out, stay enrolled or graduate, and compare two supervised
learning algorithms, Linear Regression and a Random Forest. This repository holds the code behind
our report. The full results and discussion are in the report.

The presentation video is at <https://www.youtube.com/watch?v=i4Kr5uCSZ3Q>.

## Dataset

UCI Machine Learning Repository dataset 697, *Predict Students' Dropout and Academic Success*,
available at <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>.

- It has 4,424 students from a higher education institution in Portugal, with 36 features and one target.
- The target classes are Graduate (49.9%), Dropout (32.1%) and Enrolled (17.9%).
- The notebook downloads the data with the `ucimlrepo` package. A copy is kept in `data/students_dropout.csv`.
- The dataset is shared under CC BY 4.0. Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L.
  (2021). *Predict students' dropout and academic success* [Data set]. UCI Machine Learning
  Repository. <https://doi.org/10.24432/C5MC89>

## Notebook contents

`notebooks/Assignment.ipynb` runs from top to bottom and is saved with all outputs.

1. Imports
2. Data loading and audit
3. Preprocessing and feature screening, covering cleaning, feature engineering, encoding, scaling,
   feature selection, PCA and the four feature variants
4. Model 1, Linear Regression
5. Model 2, Random Forest
6. Model comparison and the final evaluation on the test set

## Repository structure

```
ML-assignment-01/
├── data/
│   └── students_dropout.csv     # copy of UCI dataset 697
├── notebooks/
│   └── Assignment.ipynb         # all code, run top to bottom
├── members.txt                  # member IDs and emails
├── submission.txt               # dataset, GitHub and video links
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

1. Install Python 3.10 or newer.
2. Clone the repository.
   ```bash
   git clone https://github.com/LahiruSandeepa-LK/ML-assignment-01.git
   cd ML-assignment-01
   ```
3. Create a virtual environment and install the packages.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # on Windows, .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
4. Start Jupyter, open `notebooks/Assignment.ipynb` and run all cells. The first run needs internet
   access to download the dataset.

Every split and model uses `random_state=42`, so the results match the report.

## Individual contribution

We divided the work according to the project plan agreed at the start.

| Notebook sections | Work | Member |
|---|---|---|
| 1, 2 | Imports, data loading and audit | Lahiru Sandeepa |
| 3.1 to 3.7 | Cleaning, feature engineering, outlier check, split, encoding, scaling and pipeline | Kusal Dissanayake |
| 3.8 to 3.11 | Feature screening, final feature selection, PCA, feature variants, baseline and scoring | Kusal Dissanayake |
| 4 | Model 1, Linear Regression, three classification routes, gradient descent and Ridge | Lahiru Sandeepa |
| 5 | Model 2, Random Forest, decision trees, tuning, importance and early warning | Kusal Dissanayake |
| 6.1 | Model selection comparison | Both |
| 6.2 to 6.5 | Class imbalance, final test evaluation, threshold and ROC/PR analysis, fold-by-fold check | Lahiru Sandeepa |
| 6.6 | Conclusions | Both |
