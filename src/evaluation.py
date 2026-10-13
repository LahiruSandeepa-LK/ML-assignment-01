"""Shared evaluation helpers.

Both model notebooks import from here so that every metric is computed the same way
and the two results tables can be concatenated directly.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42

RESULTS_COLUMNS = [
    "model",
    "variant",           # all features | reduced | pca | 1st sem only
    "task",              # multiclass | binary
    "balancing",         # none | class_weight | smote
    "n_train",
    "n_test",
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "mcc",
    "roc_auc",           # one-vs-rest for multiclass
    "pr_auc",            # binary only
    "recall_dropout",
    "recall_enrolled",
    "recall_graduate",
    "threshold",         # NaN unless swept
    "fit_seconds",
]

CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]
POSITIVE_LABEL = "Dropout"


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    name,
    variant="all features",
    task="multiclass",
    balancing="none",
    threshold=None,
):
    """Fit a model, score it, and return one row of RESULTS_COLUMNS as a dict.

    task="binary" expects y collapsed to Dropout vs rest and fills pr_auc.
    task="multiclass" fills roc_auc one-vs-rest and leaves pr_auc as NaN.
    threshold applies to the binary task only.
    """
    raise NotImplementedError


def results_table(rows):
    """Stack result dicts into a DataFrame with the agreed column order."""
    return pd.DataFrame(rows, columns=RESULTS_COLUMNS)


def confusion_frame(y_true, y_pred, labels=None):
    """Confusion matrix as a DataFrame, rows true and columns predicted."""
    raise NotImplementedError


def majority_baseline(y_train, y_test):
    """Always-predict-the-largest-class row, for the top of the results table."""
    raise NotImplementedError
