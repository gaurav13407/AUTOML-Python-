"""
backend/core/tuner.py

Hyperparameter tuning using Optuna.
"""

import optuna
import numpy as np

from pandas.core.common import random_state
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from backend.utils.logger import logger


# ============================================================
# SVM Objective
# ============================================================

def svm_objective(
    trial,
    X_train,
    y_train,
):
    """
    Optuna objective for SVM.

    Optuna will try different hyperparameters and
    maximize weighted F1 cross-validation score.
    """

    C = trial.suggest_float(
        "C",
        0.01,
        100.0,
        log=True,
    )

    gamma = trial.suggest_float(
        "gamma",
        1e-4,
        10.0,
        log=True,
    )

    kernel = trial.suggest_categorical(
        "kernel",
        [
            "linear",
            "rbf",
            "poly",
        ],
    )

    model = SVC(
        C=C,
        gamma=gamma,
        kernel=kernel,
        probability=True,
        random_state=42,
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1_weighted",
    )

    return np.mean(scores)


# ============================================================
# Tune SVM
# ============================================================

def tune_svm(
    X_train,
    y_train,
    n_trials: int = 20,
):
    """
    Tune SVM hyperparameters using Optuna.

    Returns:
        best_params
        best_score
        study
    """

    logger.info(
        f"Starting SVM hyperparameter tuning "
        f"({n_trials} trials)..."
    )

    study = optuna.create_study(
        direction="maximize",
        study_name="svm_tuning",
    )

    study.optimize(
        lambda trial: svm_objective(
            trial,
            X_train,
            y_train,
        ),
        n_trials=n_trials,
    )

    logger.info(
        f"SVM tuning completed | "
        f"Best CV F1: {study.best_value:.6f}"
    )

    logger.info(
        f"Best parameters: {study.best_params}"
    )

    return {
        "model_name": "SVM",
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study,
    }


def build_tuned_svm(best_params):
    #==========Build an SVM Optuna's best parameters

    return SVC(
            C=best_params["C"],
            gamma=best_params["gamma"],
            kernel=best_params["kernel"],
            probability=True,
            random_state=42,
            )


def tune_model(
    model_name: str,
    X_train,
    y_train,
    n_trials: int = 20,
):
    """
    Tune a selected model using Optuna.

    Currently supported:
        - SVM
    """

    if model_name == "SVM":

        tuning_result = tune_svm(
            X_train=X_train,
            y_train=y_train,
            n_trials=n_trials,
        )

        tuned_model = build_tuned_svm(
            tuning_result["best_params"]
        )

        return {
            "model_name": model_name,
            "model": tuned_model,
            "best_params": tuning_result["best_params"],
            "best_score": tuning_result["best_score"],
        }

    raise ValueError(
        f"Hyperparameter tuning not implemented "
        f"for model: {model_name}"
    )
# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    result = tune_model(
        model_name="SVM",
        X_train=X_train,
        y_train=y_train,
        n_trials=20,
    )

    print("\n========== TUNING RESULT ==========\n")

    print("Model      :", result["model_name"])
    print("Best Score :", result["best_score"])
    print("Parameters :", result["best_params"])
    print("Model      :", result["model"])
