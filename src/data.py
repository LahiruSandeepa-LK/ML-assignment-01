"""Dataset loading, column groups and the single train/test split.

Every notebook imports the split from here so that all models are trained and
evaluated on exactly the same rows.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Target"

# Deterministic function of Nacionality, so it carries no extra information.
DROP = ["International"]

# National registry codes, no natural ordering.
NOMINAL_CODED = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

BINARY = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
]

CONTINUOUS = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Derived row-wise from existing columns, so they introduce no leakage.
ENGINEERED = [
    "approval_rate_1st",
    "approval_rate_2nd",
    "grade_trend",
    "total_approved",
]

ENGINEERED_FIRST_SEM = ["approval_rate_1st"]

CONTINUOUS_ALL = CONTINUOUS + ENGINEERED

# Columns available at the end of the first semester, for the early-warning variant.
FIRST_SEM_ONLY = [c for c in CONTINUOUS if "2nd sem" not in c] + ENGINEERED_FIRST_SEM


def add_features(df):
    """Add the engineered columns. Each is a function of one row only."""
    out = df.copy()

    for sem in ("1st", "2nd"):
        enrolled = out[f"Curricular units {sem} sem (enrolled)"]
        approved = out[f"Curricular units {sem} sem (approved)"]
        # Students enrolled in nothing get a rate of zero rather than a division error.
        rate = approved.div(enrolled.where(enrolled > 0)).fillna(0.0)
        out[f"approval_rate_{sem}"] = rate.astype("float64")

    out["grade_trend"] = (
        out["Curricular units 2nd sem (grade)"]
        - out["Curricular units 1st sem (grade)"]
    ).astype("float64")

    out["total_approved"] = (
        out["Curricular units 1st sem (approved)"]
        + out["Curricular units 2nd sem (approved)"]
    ).astype("float64")

    bad = [c for c in ENGINEERED if out[c].dtype.kind != "f"]
    assert not bad, f"engineered columns must be float, got {bad}"
    return out


def project_root(start=None):
    """Walk upwards until the directory containing data/raw is found."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "data" / "raw").is_dir():
            return candidate
    raise FileNotFoundError("could not locate the project root")


def load_raw(root=None):
    """Load the full dataset, downloading and caching it if absent."""
    root = Path(root) if root else project_root()
    path = root / "data" / "raw" / "data.csv"

    if path.exists():
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",")
    else:
        from ucimlrepo import fetch_ucirepo

        uci = fetch_ucirepo(id=697)
        df = pd.concat([uci.data.features, uci.data.targets], axis=1)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, sep=";", index=False)

    assert df.shape == (4424, 37), f"expected (4424, 37), got {df.shape}"
    return df


def make_splits(df=None, binary=False, engineered=True):
    """Return X_dev, X_test, y_dev, y_test from the one agreed split.

    binary=True collapses the target to Dropout vs rest.
    """
    df = load_raw() if df is None else df
    if engineered:
        df = add_features(df)
    X = df.drop(columns=[TARGET, *DROP])
    y = df[TARGET]

    if binary:
        y = (y == "Dropout").astype(int).rename("is_dropout")

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_dev, X_test, y_dev, y_test


def cv_splitter(n_splits=5):
    """Cross-validation scheme used for all model selection."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
