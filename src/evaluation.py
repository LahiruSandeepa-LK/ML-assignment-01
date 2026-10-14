"""Shared evaluation helpers.

Both model notebooks import from here so that every metric is computed the same way
and the two results tables can be concatenated directly.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def _scores(model, X):
    """Probability of the positive class, or None if the model cannot produce one."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


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
    if task not in ("multiclass", "binary"):
        raise ValueError(f"unknown task: {task}")
    if threshold is not None and task != "binary":
        raise ValueError("threshold applies to the binary task only")

    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = round(time.perf_counter() - started, 3)

    scores = _scores(model, X_test)
    row = {
        "model": name,
        "variant": variant,
        "task": task,
        "balancing": balancing,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "threshold": np.nan if threshold is None else threshold,
        "fit_seconds": fit_seconds,
    }

    if task == "binary":
        positive = scores[:, 1] if scores is not None and scores.ndim == 2 else scores
        if threshold is None:
            y_pred = model.predict(X_test)
        else:
            y_pred = (positive >= threshold).astype(int)

        row["roc_auc"] = roc_auc_score(y_test, positive) if positive is not None else np.nan
        row["pr_auc"] = (
            average_precision_score(y_test, positive) if positive is not None else np.nan
        )
        row["recall_dropout"] = recall_score(y_test, y_pred, pos_label=1)
        row["recall_enrolled"] = np.nan
        row["recall_graduate"] = np.nan
    else:
        y_pred = model.predict(X_test)

        if scores is not None and scores.ndim == 2:
            row["roc_auc"] = roc_auc_score(
                y_test, scores, multi_class="ovr", average="macro", labels=model.classes_
            )
        else:
            row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan

        per_class = recall_score(y_test, y_pred, average=None, labels=CLASS_ORDER,
                                 zero_division=0)
        for label, value in zip(CLASS_ORDER, per_class):
            row[f"recall_{label.lower()}"] = value

    row["accuracy"] = accuracy_score(y_test, y_pred)
    row["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
    row["precision_macro"] = precision_score(y_test, y_pred, average="macro", zero_division=0)
    row["recall_macro"] = recall_score(y_test, y_pred, average="macro", zero_division=0)
    row["f1_macro"] = f1_score(y_test, y_pred, average="macro", zero_division=0)
    row["mcc"] = matthews_corrcoef(y_test, y_pred)

    return {column: row[column] for column in RESULTS_COLUMNS}


def evaluate_cv(model, X, y, *, name, variant="all features", task="multiclass",
                balancing="none", threshold=None, cv=None):
    """Same metrics as evaluate_model, from out-of-fold predictions on one set.

    Used while the final test set is still sealed, so model selection and reporting
    both run on the development data without ever fitting on the rows being scored.
    """
    from sklearn.model_selection import cross_val_predict

    started = time.perf_counter()
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    elapsed = round(time.perf_counter() - started, 3)

    classes = np.unique(y)
    if task == "binary":
        positive = proba[:, 1]
        cut = 0.5 if threshold is None else threshold
        y_pred = (positive >= cut).astype(int)
        roc = roc_auc_score(y, positive)
        pr = average_precision_score(y, positive)
        per_class = {
            "recall_dropout": recall_score(y, y_pred, pos_label=1, zero_division=0),
            "recall_enrolled": np.nan,
            "recall_graduate": np.nan,
        }
    else:
        y_pred = classes[proba.argmax(axis=1)]
        roc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
        pr = np.nan
        scores = recall_score(y, y_pred, average=None, labels=CLASS_ORDER, zero_division=0)
        per_class = {
            f"recall_{label.lower()}": value
            for label, value in zip(CLASS_ORDER, scores)
        }

    row = {
        "model": name,
        "variant": variant,
        "task": task,
        "balancing": balancing,
        "n_train": len(y),
        "n_test": len(y),
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, y_pred),
        "roc_auc": roc,
        "pr_auc": pr,
        "threshold": np.nan if threshold is None else threshold,
        "fit_seconds": elapsed,
        **per_class,
    }
    return {column: row[column] for column in RESULTS_COLUMNS}


def results_table(rows):
    """Stack result dicts into a DataFrame with the agreed column order."""
    return pd.DataFrame(rows, columns=RESULTS_COLUMNS)


def confusion_frame(y_true, y_pred, labels=None):
    """Confusion matrix as a DataFrame, rows true and columns predicted."""
    if labels is None:
        labels = CLASS_ORDER if set(np.unique(y_true)) <= set(CLASS_ORDER) else None
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    index = labels if labels is not None else sorted(np.unique(y_true))
    return pd.DataFrame(
        matrix,
        index=pd.Index(index, name="true"),
        columns=pd.Index(index, name="predicted"),
    )


def majority_baseline(y_train, y_test, task="multiclass"):
    """Always-predict-the-largest-class row, for the top of the results table."""
    majority = pd.Series(y_train).value_counts().idxmax()
    y_pred = np.full(len(y_test), majority)

    row = {
        "model": "Majority baseline",
        "variant": "none",
        "task": task,
        "balancing": "none",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "threshold": np.nan,
        "fit_seconds": 0.0,
    }

    if task == "binary":
        row["recall_dropout"] = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        row["recall_enrolled"] = np.nan
        row["recall_graduate"] = np.nan
    else:
        per_class = recall_score(y_test, y_pred, average=None, labels=CLASS_ORDER,
                                 zero_division=0)
        for label, value in zip(CLASS_ORDER, per_class):
            row[f"recall_{label.lower()}"] = value

    return {column: row[column] for column in RESULTS_COLUMNS}
