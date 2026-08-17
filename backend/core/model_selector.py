"""
backend/core/model_selector.py

Select appropriate ML models based on dataset characteristics.

Responsibilities:
1. Analyze dataset characteristics
2. Decide which models should be trained
3. Filter unsuitable models
4. Log model selection decisions
"""

from typing import Any

import pandas as pd

from backend.utils.logger import logger
from backend.core.model_zoo import get_models


# ============================================================
# Dataset Analysis
# ============================================================

def analyze_dataset(
    X: pd.DataFrame,
    y,
    task: str,
) -> dict[str, Any]:
    """
    Analyze the dataset and return characteristics used
    for model selection.
    """

    n_rows = len(X)
    n_features = X.shape[1]

    if isinstance(X, pd.DataFrame):

        numeric_columns = X.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            exclude="number"
        ).columns.tolist()

    else:

        # Data has already gone through preprocessing.
        # At this stage X is a numerical matrix.
        numeric_columns = list(
            range(X.shape[1])
        )

        categorical_columns = []

    missing_values = int(
        pd.isna(X).sum().sum()
    )

    n_classes = None

    if task == "classification":

        if isinstance(y, pd.Series):

            n_classes = y.nunique()

        else:

            n_classes = len(
                set(y)
            )

    analysis = {
        "n_rows": n_rows,
        "n_features": n_features,
        "n_numeric_features": len(
            numeric_columns
        ),
        "n_categorical_features": len(
            categorical_columns
        ),
        "missing_values": missing_values,
        "n_classes": n_classes,
    }

    logger.info(
        "Dataset analysis:"
        f" rows={n_rows},"
        f" features={n_features},"
        f" numeric={len(numeric_columns)},"
        f" categorical={len(categorical_columns)},"
        f" missing={missing_values},"
        f" classes={n_classes}"
    )

    return analysis


# ============================================================
# Model Selection
# ============================================================

def select_models(
    X: pd.DataFrame,
    y,
    task: str,
) -> dict[str, dict[str, Any]]:
    """
    Select models appropriate for the dataset.

    The model registry remains the source of truth for
    model metadata.
    """

    models = get_models(task)

    analysis = analyze_dataset(
        X=X,
        y=y,
        task=task,
    )

    selected_models = {}

    n_rows = analysis["n_rows"]
    n_features = analysis["n_features"]
    n_categorical = analysis[
        "n_categorical_features"
    ]

    for model_name, model_info in models.items():

        reasons = []

        # ----------------------------------------------------
        # Multiclass support
        # ----------------------------------------------------

        if task == "classification":

            n_classes = analysis["n_classes"]

            if (
                n_classes > 2
                and not model_info.get(
                    "supports_multiclass",
                    False,
                )
            ):
                reasons.append(
                    f"does not support {n_classes}-class classification"
                )

        # ----------------------------------------------------
        # Very large datasets
        # ----------------------------------------------------

        if n_rows >= 100_000:

            if model_name in {
                "KNN",
                "SVM",
            }:
                reasons.append(
                    "dataset is too large for this model"
                )

        # ----------------------------------------------------
        # Very high-dimensional datasets
        # ----------------------------------------------------

        if n_features >= 500:

            if model_name == "KNN":
                reasons.append(
                    "high-dimensional dataset"
                )

        # ----------------------------------------------------
        # Categorical data
        # ----------------------------------------------------

        if n_categorical > 0:

            # Distance/kernel models are more sensitive
            # to categorical features.
            if model_name in {
                "KNN",
                "SVM",
            }:
                reasons.append(
                    "categorical features present"
                )

        # ----------------------------------------------------
        # Decide
        # ----------------------------------------------------

        if reasons:

            logger.info(
                f"Skipping {model_name}: "
                + "; ".join(reasons)
            )

        else:

            selected_models[
                model_name
            ] = model_info

            logger.info(
                f"Selected {model_name}"
            )

    logger.info(
        f"Selected {len(selected_models)} "
        f"/ {len(models)} models"
    )

    return selected_models


# ============================================================
# Public API
# ============================================================

def get_selected_models(
    X: pd.DataFrame,
    y,
    task: str,
) -> dict[str, dict[str, Any]]:
    """
    Public API for model selection.
    """

    if X is None:
        raise ValueError(
            "X cannot be None."
        )

    if y is None:
        raise ValueError(
            "y cannot be None."
        )

    return select_models(
        X=X,
        y=y,
        task=task,
    )


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    from sklearn.datasets import load_iris

    data = load_iris()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names,
    )

    y = pd.Series(
        data.target
    )

    task = "classification"

    print(
        "\n========== MODEL SELECTION ==========\n"
    )

    selected = get_selected_models(
        X=X,
        y=y,
        task=task,
    )

    print(
        "\nSelected Models:\n"
    )

    for name in selected:
        print(
            f" ✓ {name}"
        )
