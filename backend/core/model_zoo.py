
"""
backend/core/model_zoo.py

Model registry for the AutoML Engine.

Responsibilities:
1. Detect ML task (classification/regression)
2. Provide a registry of supported models
3. Store metadata about each model
"""

from pandas.api.types import (
        is_numeric_dtype,
        )

#---Classification-------
from pandas.core.common import random_state
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 

from sqlalchemy import true
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


#--------Rregression----------

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR 

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


#----------Detect Task------------
def detect_task(y):
    # Automatically determine whether the problem is classification,regression

    if not is_numeric_dtype(y):
        return "classification"
    if y.nunique()<=20:
        return "classification"

    return "regression"

#-------------Classification Models ---------------

def get_classification_models():
    return{
            "Logistic Regression":{
                "model":LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    ),

                "requires_scaling":True,
                "supports_feature_importance":False,
                "supports_probability":True,
                },

            "Decision Tree":{
                "model":DecisionTreeClassifier(
                    random_state=42,
                    ),
                "requires_scaling":False,
                "supports_feature_importance":True,
                "supports_probability":True,
                },
            "Random Forest":{
            "model":RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                ),
                "requires_scaling":False,
                "supports_feature_importance":True,
                "supports_probability":True,
            },
            "KNN":{
                "model":KNeighborsClassifier(),
                "requires_scaling":True,
                "supports_feature_importance":False,
                "supports_probability":True,
                },
            "SVM":{
                "model":SVC(
                    probability=True,
                    random_state=42,
                    ),
                "requires_scaling":True,
                "supports_feature_importance":False,
                "supports_probability":True,
                },
            "XGBoost":{
                "model":XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                    ),
                "requires_scaling":False,
                "supports_feature_importance":True,
                "supports_probability":True,
                },
            "lightGBM":{
                    "model":LGBMClassifier(
                        random_state=42,
                        verbose=-1,
                        ),
                    "requires_scaling":False,
                    "supports_feature_importance":True,
                    "supports_probability":True,
                    },
}

#-------Regresssion Models----------------

def get_regression_models():

    return {

        "Linear Regression": {

            "model": LinearRegression(),

            "requires_scaling": True,

            "supports_feature_importance": False,

        },

        "Decision Tree": {

            "model": DecisionTreeRegressor(
                random_state=42,
            ),

            "requires_scaling": False,

            "supports_feature_importance": True,

        },

        "Random Forest": {

            "model": RandomForestRegressor(
                random_state=42,
                n_estimators=100,
            ),

            "requires_scaling": False,

            "supports_feature_importance": True,

        },

        "KNN": {

            "model": KNeighborsRegressor(),

            "requires_scaling": True,

            "supports_feature_importance": False,

        },

        "SVR": {

            "model": SVR(),

            "requires_scaling": True,

            "supports_feature_importance": False,

        },

        "XGBoost": {

            "model": XGBRegressor(
                random_state=42,
            ),

            "requires_scaling": False,

            "supports_feature_importance": True,

        },

        "LightGBM": {

            "model": LGBMRegressor(
                random_state=42,
                verbose=-1,
            ),

            "requires_scaling": False,

            "supports_feature_importance": True,

        },

    }


#-----------------Public API-------------------------------

def get_models(task:str):
    if task=="classification":
        return get_classification_models()
    if task=="regression":
        return get_regression_models()

    raise ValueError(f"Unknown task:{task}")



if __name__ == "__main__":

    import pandas as pd

    print("=" * 50)

    y = pd.Series(
        [
            "Cat",
            "Dog",
            "Dog",
            "Cat",
        ]
    )

    task = detect_task(y)

    print(f"Detected Task : {task}")

    print("\nAvailable Models\n")

    models = get_models(task)

    for name, info in models.items():

        print(f"{name}")

        print(
            f"   Scaling: {info['requires_scaling']}"
        )

        print(
            f"   Feature Importance: {info['supports_feature_importance']}"
        )

        print()
