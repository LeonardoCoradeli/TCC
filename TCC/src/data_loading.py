import pandas as pd
import geohash
import numpy as np
from sklearn.compose      import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline     import Pipeline
from sklearn.model_selection import train_test_split
import category_encoders as ce

def load_and_preprocess(
    df: pd.DataFrame,
    label_col: str,
    drop_cols: list[str] = None
) -> tuple[np.ndarray, pd.Series, ColumnTransformer]:
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    y = df[label_col].copy()
    X = df.drop(columns=[label_col])

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    return X, y,cat_cols



def load_fraud_data(path: str, sample_size: int = 40000):
    df = pd.read_csv(path)
    if len(df) > sample_size:
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df['is_fraud'],
            random_state=42
        )
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['trans_timestamp'] = df['trans_date_trans_time'].astype('int64') // 10**9
    df['lat_long_hash'] = df.apply(
        lambda r: geohash.encode(r['lat'], r['long'], precision=5), axis=1
    )
    df['merch_lat_long_hash'] = df.apply(
        lambda r: geohash.encode(r['merch_lat'], r['merch_long'], precision=5), axis=1
    )

    drop_cols = [
        'id', 'trans_date_trans_time', 'cc_num', 'merchant',
        'first', 'last', 'street', 'city', 'state', 'zip', 'dob',
        'lat', 'long', 'merch_lat', 'merch_long',
        'trans_num', 'unix_time'
    ]
    return load_and_preprocess(
        df=df,
        label_col='is_fraud',
        drop_cols=drop_cols
    )


def load_heart_disease(path: str, sample_size: int = 40000):
    df = pd.read_csv(path)
    if len(df) > sample_size:
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df['HeartDisease'],
            random_state=42
        )
    df['HeartDisease'] = df['HeartDisease'].replace({'Yes': 1, 'No': 0})
    drop_cols = ['MentalHealth']
    return load_and_preprocess(
        df=df,
        label_col='HeartDisease',
        drop_cols=drop_cols
    )

def load_ecommerce(path: str, sample_size: int = 40000):
    df = pd.read_csv(path)
    if len(df) > sample_size:
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df['Churn'],
            random_state=42
        )

    drop_cols = ['Customer ID', 'Customer Name']
    return load_and_preprocess(
        df=df,
        label_col='Churn',
        drop_cols=drop_cols
    )