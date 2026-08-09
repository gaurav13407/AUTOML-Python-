
"""
backend/core/model_manager.py

Responsible for saving and loading trained AutoML artifacts.

An artifact contains:
    - trained model
    - preprocessing transformer
    - target encoder
    - metadata
    - task information
    - evaluation metrics
"""

from pathlib import Path 
from datetime import datetime 
from typing import Any 

import joblib
from numpy import save
from pandas._libs.tslibs import timestamps 

from backend.utils.logger import PROJECT_ROOT, logger

#--------------------Paths------------------------

PROJECT_ROOT=Path(__file__).resolve().parents[2]

MODEL_DIR=PROJECT_ROOT/"models"/"saved"

MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
        )




#============================Save Model=============================

def save_model(
        model,
        preprocessor,
        target_encoder,
        task:str,
        metadata:dict[str,Any],
        metrics:dict[str,Any],
        model_name:str,
        filepath:str | None=None,
        )->str:
    #============Save the complete AUTOML model artifact.=================
    if model is None:
        raise ValueError(
                "Cannot save model:model is None."
                )

    if filepath is None:
        timestamps=datetime.now().strftime(
                 "%Y%m%d_%H%M%S"
                )
        safe_model_name=(
                model_name.lower().replace(" ","_").replace("/","_")
                )
        filename=(f"{safe_model_name}_{timestamps}.joblib")

        save_path=MODEL_DIR/filename 

    else:

        save_path=Path(filepath)
        if not save_path.is_absolute():
            save_path=PROJECT_ROOT/save_path

        save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
                )


    #=============Build Artificat=================

     
    artifact = {
            "model": model,
            "preprocessor": preprocessor,
            "target_encoder": target_encoder,
            "task": task,
            "metadata": metadata,
            "metrics": metrics,
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
        }

    try:
        joblib.dump(
                artifact,
                save_path,
                )
        logger.info(f"Model Artificat saved:{save_path}")

    except Exception:
        logger.exception(f"Failed to save model:{save_path}")

        raise 

    return str(save_path)


#============================Load Model========================

def load_model(
    filepath: str,
) -> dict[str, Any]:
    """
    Load a complete AutoML model artifact.

    Returns the same dictionary that was saved.
    """

    path = Path(filepath)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():

        raise FileNotFoundError(
            f"Model artifact not found: {path}"
        )

    try:

        artifact = joblib.load(path)

        logger.info(
            f"Model artifact loaded: {path}"
        )

    except Exception:

        logger.exception(
            f"Failed to load model: {path}"
        )

        raise

    required_keys = {
        "model",
        "preprocessor",
        "target_encoder",
        "task",
        "metadata",
        "metrics",
        "model_name",
    }

    missing_keys = (
        required_keys
        - artifact.keys()
    )

    if missing_keys:

        raise ValueError(
            "Invalid model artifact. "
            f"Missing keys: {missing_keys}"
        )

    return artifact


#=========List saved Model=====================
def list_saved_models() -> list[str]:
    """
    Return all saved model artifacts.
    """

    models = sorted(
        MODEL_DIR.glob("*.joblib")
    )

    logger.info(
        f"Found {len(models)} saved model(s)."
    )

    return [
        str(model)
            for model in models
        ]


#=========Test==============


if __name__ == "__main__":

    print(
        "\n========== SAVED MODELS ==========\n"
    )

    models = list_saved_models()

    if not models:

        print(
            "No saved models found."
        )

    else:

        for model in models:

            print(
                model
            )
