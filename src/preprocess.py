"""Preprocessing pipelines shared by both model notebooks.

Everything is assembled as scikit-learn transformers so that each one is fitted on
training folds only, never on the full dataset.
"""

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import BINARY, CONTINUOUS_ALL, FIRST_SEM_ONLY, NOMINAL_CODED, RANDOM_STATE

# Levels below this share of rows are folded into a single "infrequent" category.
MIN_FREQUENCY = 0.01


# 13 components retain 95% of the variance in the continuous block (notebook 02).
PCA_COMPONENTS = 13


def continuous_block(pca_components=None):
    """Scaler for the continuous columns, with PCA appended when requested."""
    steps = [("scale", StandardScaler())]
    if pca_components:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)))
    return Pipeline(steps)


def build_preprocessor(continuous=None, pca_components=None):
    """ColumnTransformer over the nominal, binary and continuous blocks."""
    continuous = CONTINUOUS_ALL if continuous is None else continuous
    encoder = OneHotEncoder(
        min_frequency=MIN_FREQUENCY,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )
    return ColumnTransformer(
        [
            ("nominal", encoder, NOMINAL_CODED),
            ("binary", "passthrough", BINARY),
            ("continuous", continuous_block(pca_components), continuous),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model, variant="all", k=40, pca_components=13):
    """Preprocessing plus an estimator, for one of the four feature variants.

    all        every feature, scaled and encoded
    reduced    ANOVA F-test selection of the k strongest transformed features
    pca        continuous block replaced by its principal components
    first_sem  second-semester columns withheld, for the early-warning model
    """
    if variant == "all":
        pre = build_preprocessor()
    elif variant == "reduced":
        pre = build_preprocessor()
    elif variant == "pca":
        pre = build_preprocessor(pca_components=pca_components)
    elif variant == "first_sem":
        pre = build_preprocessor(continuous=FIRST_SEM_ONLY)
    else:
        raise ValueError(f"unknown variant: {variant}")

    steps = [("prep", pre)]
    if variant == "reduced":
        steps.append(("select", SelectKBest(score_func=f_classif, k=k)))
    steps.append(("model", model))
    return Pipeline(steps)
