"""
backend/core/pipeline.py

Main AutoML orchestration pipeline.

Flow:
    Dataset
        ↓
    Preprocessing
        ↓
    Task Detection
        ↓
    Model Training
        ↓
    Model Evaluation
        ↓
    Best Model

This module coordinates the AutoML engine.
It does not contain model-specific logic.
"""

from typing import Any

from joblib.parallel import FallbackToBackend

from backend.utils.logger import logger

from backend.core.preprocessing import (
    preprocessing_pipeline,
)

from backend.core.model_zoo import (
    detect_task,
)

from backend.core.trainer import (
    train_all_models,
)

from backend.core.evaluator import (
    evaluate_models,
)
from backend.core.tuner import tune_model
from backend.core.model_manager import save_model

def run_automl(
    filepath: str,
    target_col: str,
    test_size: float = 0.2,
    scale_numeric: bool = True,
) -> dict[str, Any]:
    """
    Run the complete AutoML pipeline.

    """

    logger.info("=" * 70)
    logger.info("STARTING AUTOML PIPELINE")
    logger.info("=" * 70)

    # ==========================================================
    # STEP 1 — PREPROCESSING
    # ==========================================================

    logger.info(
        "STEP 1/4: Starting preprocessing"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        target_encoder,
        metadata,
    ) = preprocessing_pipeline(
        filepath=filepath,
        target_col=target_col,
        test_size=test_size,
        scale_numeric=scale_numeric,
    )

    logger.info(
        f"Preprocessing completed | "
        f"Train={X_train.shape} | "
        f"Test={X_test.shape}"
    )

    # ==========================================================
    # STEP 2 — TASK DETECTION
    # ==========================================================

    logger.info(
        "STEP 2/4: Detecting ML task"
    )

    task = detect_task(y_train)

    logger.info(
        f"Detected task: {task}"
    )

    # ==========================================================
    # STEP 3 — MODEL TRAINING
    # ==========================================================

    logger.info(
        "STEP 3/4: Training models"
    )

    training_results = train_all_models(
        task=task,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
    )

    successful_models = sum(
        result["status"] == "success"
        for result in training_results
    )

    logger.info(
        f"Training completed | "
        f"{successful_models}/"
        f"{len(training_results)} models successful"
    )

    # ==========================================================
    # STEP 4 — MODEL EVALUATION
    # ==========================================================

    logger.info(
        "STEP 4/4: Evaluating models"
    )

    (
        evaluated_results,
        leaderboard,
        best_model,
    ) = evaluate_models(
        results=training_results,
        y_test=y_test,
        task=task,
    )

    #=========================================================
    #Step 5: HyperParameter Tuning
    #=====================================
    logger.info("STEP 5/6:HyperParameter Tuning")

    tuned_result=None 
    if best_model is not None:
        try:
            tuned_result=tune_model(
                    model_name=best_model["model_name"],
                    X_train=X_train,
                    y_train=y_train,
                    n_trials=20,
                    )
            logger.info(f"Tunning completed for {best_model['model_name']}")
            logger.info(f"best tunning score:"
                        f"{tuned_result['best_score']:.6f}")

            logger.info(f"Best parameters:"
                        f"{tuned_result['best_params']}")
        except ValueError as e:
            logger.warning(
                    f"No tuner available for"
                    f"{best_model['model_name']}:{e}"
                    )
        except Exception:
            logger.exception(
                    "HyperParameter tunning failed."
                    )

    #==========================================================
    # Step 6:ORIGINAL VS TUNED MODEL 
    #=======================================================

    final_model=best_model 

    if tuned_result is not None:
        logger.info("Evaluating tuned model on test set....")

        tuned_model=tuned_result["model"]
        tuned_model.fit(
                X_train,
                y_train,
                )
        tuned_pred=tuned_model.predict(
                X_test
                )
        tuned_prob=None 

        if hasattr(tuned_model,"predict_proba"):
            tuned_prob=tuned_model.predict_proba(X_test)
        from sklearn.metrics import f1_score

        tuned_f1=f1_score(
                y_test,
                tuned_pred,
                average="weighted",
                )
        original_f1=best_model["metrics"].get("f1",0,)

        logger.info(
        f"Original {best_model['model_name']} "
        f"F1: {original_f1:.6f}"
            )

        logger.info(
                f"Tuned {best_model['model_name']}"
                f"F1:{tuned_f1:.6f}"
                )
        if tuned_f1>original_f1:
            logger.info(
                    "Tuned model is better."
                    "Using tuned model."
                    )
            final_model={
                    **best_model,
                    "model":tuned_model,
                    "y_pred":tuned_pred,
                    "y_prob":tuned_prob,
                    "metrics":{
                        **best_model["metrics"],
                        "f1":tuned_f1,
                        },
                    "tuned":True,
                    "best_params":tuned_result[
                        "best_params"
                        ],
                    }
        else:
            logger.info(
                    "Oringal model is better."
                    "Keeping Oringal model."
                    )
            final_model["tuned"]=False


    #=========================================================
    #==============================SAVE BEST MODEL==========
    #====================================================== 

    saved_model_path=None 
    if best_model is not None:
        saved_model_path=save_model(
                model=best_model["model"],
                preprocessor=preprocessor,
                target_encoder=target_encoder,
                task=task,
                metadata=metadata,
                metrics=best_model["metrics"],
                model_name=best_model["model_name"]
                )
        logger.info(
                f"Best Model saved:{saved_model_path}"
                )
    else:
        logger.warning("No best model available to save.")

    # ==========================================================
    # FINAL RESULT
    # ==========================================================

    if best_model is not None:

        logger.info(
            f"Best model: "
            f"{best_model['model_name']}"
        )

    else:

        logger.warning(
            "No successful model was selected."
        )

    logger.info("=" * 70)
    logger.info("AUTOML PIPELINE COMPLETED")
    logger.info("=" * 70)

    return {
        "task": task,

        "metadata": metadata,

        "preprocessor": preprocessor,

        "target_encoder": target_encoder,

        "training_results": training_results,

        "evaluated_results": evaluated_results,

        "leaderboard": leaderboard,

        "best_model": best_model,

        "saved_model_path":saved_model_path,

        "X_test": X_test,

        "y_test": y_test,
    }


# ==============================================================
# TEST / CLI
# ==============================================================

if __name__ == "__main__":

    result = run_automl(
        filepath="data/Iris.csv",
        target_col="Species",
    )

    # ----------------------------------------------------------
    # Task
    # ----------------------------------------------------------

    print(
        "\n========== AUTOML RESULT ==========\n"
    )

    print(
        "Detected Task:",
        result["task"],
    )

    # ----------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------

    print(
        "\n========== PREPROCESSING METADATA ==========\n"
    )

    for key, value in result["metadata"].items():

        print(
            f"{key}: {value}"
        )

    # ----------------------------------------------------------
    # Leaderboard
    # ----------------------------------------------------------

    print(
        "\n========== MODEL LEADERBOARD ==========\n"
    )

    leaderboard = result["leaderboard"]

    if leaderboard.empty:

        print(
            "No successful models."
        )

    else:

        print(
            leaderboard.to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # Best Model
    # ----------------------------------------------------------

    print(
        "\n========== BEST MODEL ==========\n"
    )

    best_model = result["best_model"]

    if best_model is not None:

        print(
            "Model         :",
            best_model["model_name"],
        )

        print(
            "Task          :",
            best_model["task"],
        )

        print(
            "Training Time :",
            best_model["training_time"],
        )

        print(
            "Metrics       :",
            best_model["metrics"],
        )

    else:

        print(
            "No successful model found."
        )


