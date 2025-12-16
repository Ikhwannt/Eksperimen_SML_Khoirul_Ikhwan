import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


def load_dataset():
    """
    Load dataset Breast Cancer dari scikit-learn
    """
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values
    Dataset ini tidak memiliki missing value,
    namun fungsi disiapkan untuk robustness
    """
    return df.dropna()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus data duplikat jika ada
    """
    return df.drop_duplicates()


def detect_outliers_iqr(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Deteksi outlier menggunakan metode IQR
    Outlier tidak dihapus (khusus kasus medis)
    """
    Q1 = df[feature_columns].quantile(0.25)
    Q3 = df[feature_columns].quantile(0.75)
    IQR = Q3 - Q1

    outlier_mask = (
        (df[feature_columns] < (Q1 - 1.5 * IQR)) |
        (df[feature_columns] > (Q3 + 1.5 * IQR))
    )

    # Hanya logging (tidak menghapus)
    outlier_count = outlier_mask.sum().sum()
    print(f"[INFO] Total outlier terdeteksi: {outlier_count}")

    return df


def feature_scaling(df: pd.DataFrame, scaler_path: str = None):
    """
    Standarisasi fitur numerik menggunakan StandardScaler
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Simpan scaler jika path disediakan
    if scaler_path:
        joblib.dump(scaler, scaler_path)

    return X_scaled, y

def preprocess_pipeline(
    save_scaler: bool = True,
    scaler_path: str = "standard_scaler.pkl"
):
    """
    Pipeline preprocessing utama
    Mengembalikan data siap dilatih
    """

    # 1. Load data
    df = load_dataset()

    # 2. Missing value handling
    df = handle_missing_values(df)

    # 3. Remove duplicates
    df = remove_duplicates(df)

    # 4. Outlier detection
    feature_columns = df.drop("target", axis=1).columns.tolist()
    df = detect_outliers_iqr(df, feature_columns)

    # 5. Feature scaling
    X_scaled, y = feature_scaling(
        df,
        scaler_path=scaler_path if save_scaler else None
    )

    print("[INFO] Preprocessing selesai. Data siap untuk training.")

    return X_scaled, y


if __name__ == "__main__":
    X, y = preprocess_pipeline()
    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
