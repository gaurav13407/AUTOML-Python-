"""
backend/core/preprocessing.py

Generic preprocessing pipeline for tabular datasets.
Prevents data leakage by fitting transformers only on training data.
"""

from typing import Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, _target_encoder
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import LabelEncoder

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded {filepath} -> {df.shape}")
    return df


def basic_info(df: pd.DataFrame) -> None:
    print("\n========== DATA INFO ==========")
    df.info()

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== SAMPLE ==========")
    print(df.head())


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = None

    if (
        y.nunique() > 1
        and y.value_counts().min() >= 2
    ):
        stratify = y

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

def remove_identifier_columns(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """
    Remove identifier columns such as:
    Id, CustomerID, OrderID, EmployeeID, etc.

    Detection:
    1. Common ID column names.
    2. Integer column with all unique values.
    """

    df = df.copy()

    id_keywords = {
        "id",
        "customerid",
        "customer_id",
        "userid",
        "user_id",
        "employeeid",
        "employee_id",
        "orderid",
        "order_id",
        "invoiceid",
        "invoice_id",
        "transactionid",
        "transaction_id",
    }

    cols_to_drop = []

    for col in df.columns:

        if col == target_col:
            continue

        # Rule 1: Column name
        if col.lower() in id_keywords:
            cols_to_drop.append(col)
            continue

        # Rule 2: Integer column with unique values
        if (
            is_integer_dtype(df[col])
            and df[col].is_unique
            and df[col].min()==1
            and df[col].max()==len(df)
        ):
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"Removing identifier columns: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    return df


def encode_target(
        df:pd.DataFrame,
        target_col:str,
        ):
    # encode column if it is categorical 
    df=df.copy()
    encoder=None 
    if(
        df[target_col].dtype=="object"
        or str(df[target_col].dtype)=="category"
            ):
        encoder=LabelEncoder()
        df[target_col]=encoder.fit_transform(df[target_col])
        print(f"Target encoded:{list(encoder.classes_)}")
    return df,encoder 




def analyze_dataset(
    df: pd.DataFrame,
    target_col: str,
    cardinality_threshold: int = 20,
) -> pd.DataFrame:
    """
    Analyze dataset and automatically:
    - Remove constant columns
    - Convert date columns into numerical features
    - Remove high-cardinality categorical columns
    """

    df = df.copy()

    # -----------------------------
    # Remove constant columns
    # -----------------------------
    constant_cols = [
        col for col in df.columns
        if col != target_col and df[col].nunique(dropna=False) <= 1
    ]

    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        df.drop(columns=constant_cols, inplace=True)

    df=remove_identifier_columns(
        df,
        target_col,
        )

    # -----------------------------
    # Detect date columns
    # -----------------------------
    object_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in object_cols:

        if col == target_col:
            continue

        converted=pd.to_datetime(
                df[col],
                errors="coerce",
                )

        if converted.notna().mean()>0.9:
            converted = pd.to_datetime(df[col], errors="raise")

            print(f"Detected date column: {col}")

            df[f"{col}_year"] = converted.dt.year
            df[f"{col}_month"] = converted.dt.month
            df[f"{col}_day"] = converted.dt.day
            df[f"{col}_dayofweek"] = converted.dt.dayofweek

            df.drop(columns=[col], inplace=True)




    # -----------------------------
    # Detect high-cardinality columns
    # -----------------------------
    object_cols = df.select_dtypes(include=["object", "string"]).columns

    high_cardinality = []

    for col in object_cols:

        if col == target_col:
            continue

        unique_values = df[col].nunique()

        if unique_values > cardinality_threshold:
            high_cardinality.append(col)

    if high_cardinality:
        print(f"Dropping high-cardinality columns: {high_cardinality}")
        df.drop(columns=high_cardinality, inplace=True)

    return df


def build_preprocessor(
    X_train: pd.DataFrame,
    scale_numeric: bool = True,
) -> ColumnTransformer:

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    categorical_cols = (
        X_train.select_dtypes(include=["object", "category", "string"])
        .columns.tolist()
    )


    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def preprocessing_pipeline(
    filepath: str,
    target_col: str,
    test_size: float = 0.2,
    scale_numeric: bool = True,
):
    df = load_data(filepath)

    basic_info(df)
    df = analyze_dataset(df, target_col)

    df,target_encoder=encode_target(df,target_col)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col,
        test_size,
    )

    preprocessor = build_preprocessor(
        X_train,
        scale_numeric,
    )

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        target_encoder,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the preprocessing pipeline"
    )

    parser.add_argument(
        "filepath",
        help="Path to the CSV file"
    )

    parser.add_argument(
        "target",
        help="Target column name"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size"
    )

    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable feature scaling"
    )

    args = parser.parse_args()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        target_encoder,

    ) = preprocessing_pipeline(
        filepath=args.filepath,
        target_col=args.target,
        test_size=args.test_size,
        scale_numeric=not args.no_scale,
    )
    print("Target encoded:", target_encoder is not None)
    print("\n========== RESULTS ==========")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\nPreprocessor Built Successfully")
