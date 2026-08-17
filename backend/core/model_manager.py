
"""
backend/core/model_manager.py

Responsible for saving and loading trained AutoML artifacts.

An artifact contains:
    - trained model
    - preprocessing transform
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
import pandas as pd 

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
            "feature_names": preprocessor.feature_names_in_.tolist(), 
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


def predict(
        filepath:str,
        X:pd.DataFrame,
        return_lables:bool=True,
        ):
    """
    Predict using a saved AUTOML Artificat 
    """
    artifact=load_model(filepath)
    model=artifact["model"]
    preprocessor=artifact["preprocessor"]
    target_encoder=artifact["target_encoder"]

    expected_feature=artifact["feature_names"]

    #=============VAlidate columns============
    missing=[
            col for col in expected_feature
            if col not in X.columns
            ]
    if missing:
        raise ValueError(
                f"Missing required columns:{missing}"
                )
    X=X[expected_feature]

    #================Preprocess==============
    X_processed=preprocessor.transform(X)
    predictions=model.predict(X_processed)
    if(
        return_lables
        and target_encoder is not None
        ):
        predictions=target_encoder.inverse_transform(
                predictions
                )
    logger.info(
            f"Generated{len(predictions)}predictions"
            f"Using {artifact['model_name']}"
            )
    return predictions




def predict_proba(
        filepath:str,
        X:pd.DataFrame,
        ):
    #===============Return prediction probabilites for classificaton models.==================
    artifact=load_model("model")

    model=artifact["model"]
    preprocessor=artifact["preprocessor"]

    if not hasattr(model,"predict_proba"):
        raise ValueError(
                f"{artifact['model_name']}"
                "does not support probabilites."
                )
    X=X[artifact["feature_names"]]

    X_processed=preprocessor.transform(X)
    return model.predict_proba(X_processed)
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

    import pandas as pd

    models = list_saved_models()

    if not models:

        print("No saved models.")
        exit()

    model_path = models[-1]

    print("\nUsing model:")
    print(model_path)

    artifact = load_model(model_path)

    print("\nModel:", artifact["model_name"])
    print("Task :", artifact["task"])

    # Iris example
    sample = pd.DataFrame(
        [
            {
                "SepalLengthCm": 5.1,
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2,
            },
            {
                "SepalLengthCm": 6.5,
                "SepalWidthCm": 3.0,
                "PetalLengthCm": 5.5,
                "PetalWidthCm": 2.0,
            },
        ]
    )

    preds = predict(
        model_path,
        sample,
    )

    print("\nPredictions:")

    for p in preds:
        print(" •", p)
