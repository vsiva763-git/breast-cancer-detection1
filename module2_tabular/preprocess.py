import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess(csv_path='data/wisconsin.csv', scaler_path='models/scaler.pkl'):
    df = pd.read_csv(csv_path)

    # Drop irrelevant columns
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    if 'Unnamed: 32' in df.columns:
        df = df.drop(['Unnamed: 32'], axis=1)

    # Encode target: M = 1 (Malignant), B = 0 (Benign)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    feature_names = list(X.columns)
    print(f"Features: {len(feature_names)}")
    print(f"Samples:  {len(df)}")
    print(f"Malignant: {y.sum()} | Benign: {(y==0).sum()}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler for dashboard use
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_names, scaler
