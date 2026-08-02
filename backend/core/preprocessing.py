"""
core/preprocessing.py
 
Generic preprocessing pipeline for tabular datasets.
Handles: missing values, categorical encoding, numeric scaling,
and train/test split — works on any CSV, not just one dataset.
"""

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_data(filepath:str)-> pd.DataFrame:
    df=pd.read_csv(filepath)
    print(f"Loaded {filepath} -> shape:{df.shape}")
    return df 

def basic_info(df:pd.DataFrame)->None:
    print("\n---Basic Info---")
    print(df.info())
    print("\n---Missing Values---")
    print(df.isnull().sum()[df.isnull().sum()>0])
    print("\n---Sample Rows---")
    print(df.head())


def handle_missing_values(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    for col in df.columns:
        if df[col].isnull().sum()==0:
            continue
        if df[col].dtype in [np.float64,np.int64]:
            df[col]=df[col].fillna(df[col].median())
        else:
            df[col]=df[col].fillna(df[col].mode()[0])
    return df 



def encoded_categorical(df:pd.DataFrame,target_col:str)->pd.DataFrame:
    df=df.copy()
    cat_cols=df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    for col in cat_cols:
        le=LabelEncoder()
        df[col]=le.fit_transform(df[col].astype(str))

    return df 
