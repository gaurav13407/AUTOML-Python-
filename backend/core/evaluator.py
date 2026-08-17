
"""
backend/core/evaluator.py

Evaluate trained AutoML models and generate a model leaderboard.

Responsibilities:
1. Calculate metrics for classification/regression.
2. Handle prediction probabilities when available.
3. Rank successful models.
4. Return a standardized leaderboard.
"""

from math import log
from typing import Any

import numpy as np 
import pandas as pd 

from sklearn import metrics
from sklearn.base import MetaEstimatorMixin
from sklearn.externals.array_api_compat.numpy import average
from sklearn.metrics import(
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        )

from backend.utils.logger import logger

#---------Classification------------- 

def evaluate_classification(
        y_true,
        y_pred,
        y_prob=None,
        ):
    metrics={}

    metrics["accuracy"]=accuracy_score(
            y_true,
            y_pred,
            )

    metrics["precision"]=precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
            )
    metrics["recall"]=recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
            )
    metrics["f1"]=f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
            )

    #-----------Probability-based metrics---------------- 
    metrics["roc_auc"]=None 
    metrics["log_loss"]=None 

    if y_prob is not None:
        try:
            if y_prob.ndim==2 and y_prob.shape[1]>2:
                metrics["roc_auc"]=roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="weighted",
                        )
            else:
                metrics["roc_auc"]=roc_auc_score(
                        y_true,
                        y_prob[:,1],
                        )
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC:{e}")
        try:
            metrics["log_loss"]=log_loss(
                    y_true,
                    y_prob,
                    )
        except Exception as e:
            logger.warning(f"Could not calculate Log Loss :{e}")

    return metrics


#-------------Regression--------------------------

def evalaute_regression(
        y_true,
        y_pred,
        ):
    mse=mean_squared_error(
            y_true,
            y_pred,
            )
    metrics={
            "mae":mean_absolute_error(
                y_true,
                y_pred,
                ),

            "mse":mse,
            "rmse":np.sqrt(mse),
            "r2":r2_score(
                y_true,
                y_pred,
                ),
            }
    return metrics


def evaluate_single_model(
    result: dict[str, Any],
    y_test,
):
    """
    Evaluate one trained model.
    """

    model_name = result["model_name"]
    task = result["task"]

    # --------------------------------------------------------
    # Check training status
    # --------------------------------------------------------

    if result["status"] != "success":

        logger.warning(
            f"Skipping evaluation for failed model: "
            f"{model_name}"
        )

        return {
            **result,
            "metrics": {},
        }

    y_pred = result["y_pred"]

    y_prob = result.get(
        "y_prob",
        None,
    )

    logger.info(
        f"Evaluating {model_name}..."
    )

    try:

        if task == "classification":

            metrics = evaluate_classification(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
            )

        elif task == "regression":

            metrics = evaluate_regression(
                y_true=y_test,
                y_pred=y_pred,
            )

        else:

            raise ValueError(
                f"Unknown task: {task}"
            )

        logger.info(
            f"Finished evaluation: {model_name}"
        )

        return {
            **result,
            "metrics": metrics,
        }

    except Exception as e:

        logger.exception(
            f"Evaluation failed for {model_name}"
        )

        return {
            **result,
            "metrics": {},
            "status": "evaluation_failed",
            "error": str(e),
        }

#============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_all_models(
    results: list[dict[str, Any]],
    y_test,
):
    """
    Evaluate every trained model.
    """

    evaluated_results = []

    for result in results:

        evaluated_result = evaluate_single_model(
            result=result,
            y_test=y_test,
        )

        evaluated_results.append(
            evaluated_result
        )

    return evaluated_results


# ============================================================
# LEADERBOARD
# ============================================================

def create_leaderboard(
    results: list[dict[str, Any]],
    task: str,
):
    """
    Create a ranked leaderboard from evaluated models.
    """

    rows = []

    for result in results:

        # Only include successfully evaluated models
        if result["status"] != "success":
            continue

        metrics = result.get("metrics", {})

        # Make sure metrics actually exist
        if not metrics:
            logger.warning(
                f"No metrics found for {result['model_name']}"
            )
            continue

        row = {
            "model": result["model_name"],
            "training_time": result["training_time"],

            #Cross-validate results 

            "cv_mean":result.get("cv_mean"),
            "cv_std":result.get("cv_std"),
            "cv_metric":result.get("cv_metric"),
        }

        row.update(metrics)

        rows.append(row)

    # --------------------------------------------------------
    # No valid results
    # --------------------------------------------------------

    if not rows:

        logger.warning(
            "No evaluated models available for leaderboard."
        )

        return pd.DataFrame()

    leaderboard = pd.DataFrame(rows)

    logger.info(
        f"Leaderboard columns: {list(leaderboard.columns)}"
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if task == "classification":

        ranking_metric = "f1"

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    elif task == "regression":

        ranking_metric = "r2"

    else:

        raise ValueError(
            f"Unknown task: {task}"
        )

    # --------------------------------------------------------
    # Make sure ranking metric exists
    # --------------------------------------------------------

    if ranking_metric not in leaderboard.columns:

        raise ValueError(
            f"Ranking metric '{ranking_metric}' "
            f"was not produced by the evaluator. "
            f"Available metrics: "
            f"{list(leaderboard.columns)}"
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    leaderboard = leaderboard.sort_values(
        by=ranking_metric,
        ascending=False,
    )

    leaderboard = leaderboard.reset_index(
        drop=True
    )

    leaderboard.insert(
        0,
        "rank",
        range(
            1,
            len(leaderboard) + 1,
        ),
    )

    return leaderboard


def get_best_model(
        evaluated_result:list[dict[str,Any]],
        leaderboard:pd.DataFrame,
        task:str,
        ):
    if leaderboard.empty:
        logger.warning("CAnnot select best model:leaderboard is empty.")
        return None

    successful=[
            result 
            for result in evaluated_result
            if result["status"]=="success"
            ]
    if not successful:
        logger.warning(
                "cannot select best model:no successful models."
                )
        return None

    if task == "classification":
        ranked=leaderboard.sort_values(
                by=["cv_mean","cv_std","f1"],
                ascending=[False,True,False],
                )
    elif task=="regression":
        ranked=leaderboard.sort_values(
                by=["cv_mean","cv_std","rmse"],
                ascending=[True,True,True],
                )
    else:
        raise ValueError(
                f"Unknown Task:{task}"
                )

    best_model_name=leaderboard.iloc[0]["model"]

    for result in evaluated_result:
        if(
                result["status"]=="success"
                and result["model_name"]==best_model_name
                ):
            logger.info(
                    f"Best Model selected:{best_model_name}"
                    )
            return{
                    "model_name":best_model_name,
                    "model":result["model"],
                    "task":result["task"],
                    "metrics":result.get("metrics",{}),
                    "training_time":result["training_time"],
                    "y_pred":result.get("y_pred"),
                    "y_prob":result.get("y_prob"),
                    "cv_scores":result.get("cv_scores",[]),
                    "cv_mean":result.get("cv_mean"),
                    "cv_std":result.get("cv_std"),
                    "cv_metric":result.get("cv_metric")
                    }
    logger.warning(
                    f"Best Model '{best_model_name}' was not found"
                    "in evaluated results"
                    )
    return None 
# ============================================================
# PUBLIC API
# ============================================================

def evaluate_models(
    results: list[dict[str, Any]],
    y_test,
    task: str,
):
    """
    Main evaluator entry point.

    Returns:
        evaluated_results
        leaderboard
    """

    logger.info(
        f"Evaluating {len(results)} models..."
    )

    evaluated_results = evaluate_all_models(
        results=results,
        y_test=y_test,
    )

    leaderboard = create_leaderboard(
        results=evaluated_results,
        task=task,
    )

    best_model=get_best_model(
            evaluated_results,
            leaderboard,
            task,
            )
    print("\n---------------Best Model--------------------")
    if best_model is not None:
        print("Model        :", best_model["model_name"])
        print("Task         :", best_model["task"])
        print("Training Time:", best_model["training_time"])
        print("Metrics      :", best_model["metrics"])

    else:

        print("No successful model found.")

    logger.info(
        "Evaluation completed."
    )

    return (
        evaluated_results,
        leaderboard,
        best_model
    )


if __name__ == "__main__":

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    from backend.core.model_zoo import detect_task
    from backend.core.trainer import train_all_models

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = load_iris()

    X = data.data
    y = data.target

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------------
    # Detect task
    # --------------------------------------------------------

    task = detect_task(y_train)

    print(
        f"\nDetected Task: {task}"
    )

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    results = train_all_models(
        task=task,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluated_results, leaderboard,best_model = evaluate_models(
        results=results,
        y_test=y_test,
        task=task,
    )

    # --------------------------------------------------------
    # Display leaderboard
    # --------------------------------------------------------

    print(
        "\n========== MODEL LEADERBOARD ==========\n"
    )

    print(
        leaderboard.to_string(
            index=False
        )
    )
