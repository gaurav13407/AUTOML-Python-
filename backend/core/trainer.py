"""
backend/core/trainer.py

Train all machine learning models from the model zoo.
"""

import time
from typing import Any

from pandas.core.common import random_state
from sklearn.model_selection import StratifiedKFold, cross_validate,StratifiedGroupKFold,KFold
import numpy as np
from sklearn.utils import shuffle 
from backend.utils.logger import logger
from backend.core.model_selector import get_selected_models
from backend.core.model_zoo import (
    detect_task,
    get_models,
)


def create_result(
        model_name:str,
        task:str,
        model,
        y_pred,
        y_prob,
        training_time:float,
        status:str="success",
        error:str | None=None,
        cv_scores=None,
        cv_mean=None,
        cv_std=None,
        cv_metric=None,
        ):
    # Standard result object returned after training a model 

    return{
            "model_name":model_name,
            "task":task,
            "model":model,
            "y_pred":y_pred,
            "y_prob":y_prob,
            "training_time":round(training_time,4),
            "status":status,
            "error":error,
            "cv_scores":"cv_score",
            "cv_mean":cv_mean,
            "cv_std":cv_std,
            "cv_metric":cv_metric
            }


def cross_validate_model(
        model,
        X_train,
        y_train,
        task:str,
        cv:int=5,
        ):
    #========Perform cross_validation on the training data.===============
    logger.info(
            f"Running {cv}-fold cross_validation...."
            )

    if task=="classification":
        splitter=StratifiedKFold(
                n_splits=cv,
                shuffle=True,
                random_state=42,
                )
        scoring="f1_weighted"
    elif task == "regression":
        splitter=KFold(
                n_splits=cv,
                shuffle=True,
                random_state=42,
                )
        scoring="neg_root_mean_squared_error"
    else:
        raise ValueError(
                f"Unsupported task:{task}"
                )

    cv_result=cross_validate(
            model,
            X_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
    scores=cv_result["test_score"]

    if task=="regression":
        scores=-scores

    result={
            "cv_scores":scores.tolist(),
            "cv_mean":float(np.mean(scores)),
            "cv_std":float(np.std(scores)),
            "cv_folds":cv,
            "cv_metric":scoring,
            }
    logger.info(
            f"CV Mean:{result['cv_mean']:4f}"
            f"CV Std:{result['cv_std']:4f}"
            )
    return result


def train_single_model(
    model_name: str,
    model_info: dict[str, Any],
    task: str,
    X_train,
    y_train,
    X_test,
):
    """
    Train one model and return predictions.
    """

    logger.info(f"Training {model_name}...")

    model = model_info["model"]

    cv_result=None

    start = time.perf_counter()

    try:
        cv_result=cross_validate_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                task=task,
                cv=5,
            )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_prob=None 

        if model_info.get("supports_probability",False):
            y_prob=model.predict_proba(X_test)

        training_time = time.perf_counter() - start

        logger.info(
            f"Finished {model_name} ({training_time:.3f}s)"
        )

        return create_result(
            model_name=model_name,
            task=task,
            model=model,
            y_pred=y_pred,
            y_prob=y_prob,
            training_time=training_time,

            cv_scores=cv_result["cv_scores"],
            cv_mean=cv_result["cv_mean"],
            cv_std=cv_result["cv_std"],
            cv_metric=cv_result["cv_metric"],
        )

    except Exception as e:

        logger.exception(f"{model_name} failed.")

        return create_result(
            model_name=model_name,
            task=task,
            model=None,
            y_pred=None,
            y_prob=None,
            training_time=0,
            status="failed",
            error=str(e),
        )


def train_all_models(
    task: str,
    X_train,
    y_train,
    X_test,
):
    """
    Train every model available for the detected task.
    """

    models = get_selected_models(
            X=X_train,
            y=y_train,
            task=task,
            )

    logger.info(f"Found {len(models)} models for {task}")

    results = []

    for model_name, model_info in models.items():

        result = train_single_model(
            model_name=model_name,
            model_info=model_info,
            task=task,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        )

        results.append(result)

    return results


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()

    X = data.data
    y = data.target

    task = detect_task(y)

    results = train_all_models(
        task=task,
        X_train=X,
        y_train=y,
        X_test=X,
    )

    print("\n========== TRAINING SUMMARY ==========\n")

    for result in results:

        status = "✓" if result["status"] == "success" else "✗"

        print(
            f"{status} "
            f"{result['model_name']:<25}"
            f"{result['training_time']:.3f} sec"
        )

    success = sum(
        r["status"] == "success"
        for r in results
    )

    print(
        f"\nSuccessfully trained {success}/{len(results)} models."
    ) 
