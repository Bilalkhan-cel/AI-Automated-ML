import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler , MinMaxScaler,OneHotEncoder,LabelEncoder 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def Data_cleaning(daf, target, feature, test_size=0.2):

    y = daf[target]
    x = daf[feature]




    

   


    cat_cols = x.select_dtypes(include="object").columns
    num_cols = x.select_dtypes(include="number").columns


    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42
    )


    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )


    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),

            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )


    x_train = preprocessor.fit_transform(x_train)

    x_test = preprocessor.transform(x_test)


    print("X_train:", x_train.shape)
    print("X_test:", x_test.shape)
    print("Y_train:", y_train.shape)
    print("Y_test:", y_test.shape)


    return x_train, x_test, y_train, y_test, preprocessor


# def get_columns(df):
#     col=df.columns.to_list()
#     return col
# def get_num_columns(df):
#     col=df.select_dtypes(include="number").columns.to_list()
#     return col
# def get_cat_columns(df):
#     col=df.select_dtypes(include="object").columns.tolist()
#     return col

# def apply_onehot(df):
#     cat_col=get_cat_columns(df)
#     oh=OneHotEncoder(sparse_output=False).set_output(transform="pandas")
#     df_encoded=oh.fit_transform(df[cat_col])
#     return df_encoded

# def apply_standard_scaling(df):
#     num_col=get_num_columns(df)
#     oh=StandardScaler().set_output(transform="pandas")
#     df_encoded=oh.fit_transform(df[num_col])
#     return df_encoded


import math





def sanitize_records(records):
    clean = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                clean_row[k] = None
            else:
                clean_row[k] = v
        clean.append(clean_row)
    return clean

    