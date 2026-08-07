"""
backend/core/trainer.py

Train all machine learning models from the model zoo.
"""

import time
from typing import Any

from backend.utils.logger import logger
from backend.core.model_zoo import (
    detect_task,
    get_models,
)


def create_result(
        model_name:str,
        task:str,
        model,
        y_pred,
        training_time:float,
        status:str="success",
        error:str | None=None,
        ):
    # Standard result object returned after training a model 

    return{
            "model_name":model_name,
            "task":task,
            "model":model,
            "y_pred":y_pred,
            "training_time":round(training_time,4),
            "status":status,
            "error":error,
            }

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

    start = time.perf_counter()

    try:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        training_time = time.perf_counter() - start

        logger.info(
            f"Finished {model_name} ({training_time:.3f}s)"
        )

        return create_result(
            model_name=model_name,
            task=task,
            model=model,
            y_pred=y_pred,
            training_time=training_time,
        )

    except Exception as e:

        logger.exception(f"{model_name} failed.")

        return create_result(
            model_name=model_name,
            task=task,
            model=None,
            y_pred=None,
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

    models = get_models(task)

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
