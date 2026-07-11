import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler , MinMaxScaler,OneHotEncoder,LabelEncoder


def get_columns(df):
    col=df.columns.to_list()
    return col
def get_num_columns(df):
    col=df.select_dtypes(include="number").columns.to_list()
    return col
def get_cat_columns(df):
    col=df.select_dtypes(include="object").columns.tolist()
    return col

def apply_onehot(df):
    cat_col=get_cat_columns(df)
    oh=OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    df_encoded=oh.fit_transform(df[cat_col])
    return df_encoded

def apply_standard_scaling(df):
    num_col=get_num_columns(df)
    oh=StandardScaler().set_output(transform="pandas")
    df_encoded=oh.fit_transform(df[num_col])
    return df_encoded


    
    