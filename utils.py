import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.dropna()

    # Encode categorical
    le = LabelEncoder()
    df['location'] = le.fit_transform(df['location'])

    X = df.drop('price', axis=1)
    y = df['price']

    return X, y, le