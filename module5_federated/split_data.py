import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def split_into_hospitals(csv_path='data/wisconsin.csv',
                          output_dir='data/federated'):
    """
    Split Wisconsin dataset into 3 hospital partitions.
    Each hospital has different data sizes to simulate real world.
    Hospital A: 40% of data
    Hospital B: 35% of data
    Hospital C: 25% of data
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Clean same way as module 2
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    if 'Unnamed: 32' in df.columns:
        df = df.drop(['Unnamed: 32'], axis=1)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    total = len(df)
    split_a = int(total * 0.40)
    split_b = int(total * 0.35)

    hospital_a = df.iloc[:split_a]
    hospital_b = df.iloc[split_a:split_a + split_b]
    hospital_c = df.iloc[split_a + split_b:]

    hospital_a.to_csv(f'{output_dir}/hospital_A.csv', index=False)
    hospital_b.to_csv(f'{output_dir}/hospital_B.csv', index=False)
    hospital_c.to_csv(f'{output_dir}/hospital_C.csv', index=False)

    print("Hospital data split complete:")
    print(f"  Hospital A : {len(hospital_a)} samples (40%)")
    print(f"  Hospital B : {len(hospital_b)} samples (35%)")
    print(f"  Hospital C : {len(hospital_c)} samples (25%)")
    print(f"  Total      : {total} samples")
    print(f"  Saved to   : {output_dir}/")

    return hospital_a, hospital_b, hospital_c


def load_hospital_data(hospital_file):
    """Load and preprocess a single hospital CSV."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(hospital_file)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    X = df.drop('diagnosis', axis=1).values
    y = df['diagnosis'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
