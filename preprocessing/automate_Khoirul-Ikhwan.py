import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# LOAD DATASET
def load_dataset(
    raw_path: str = "BreastCancer_raw.csv"
) -> pd.DataFrame:
    """
    Load dataset Breast Cancer dari file CSV (raw)
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"[ERROR] Dataset tidak ditemukan di {raw_path}")

    df = pd.read_csv(raw_path)
    return df
    
# HANDLE MISSING VALUES
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values
    """
    return df.dropna()
    
# REMOVE DUPLICATES
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus data duplikat jika ada
    """
    return df.drop_duplicates()

# OUTLIER DETECTION (IQR)
def detect_outliers_iqr(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Deteksi outlier menggunakan metode IQR
    Outlier tidak dihapus (kasus medis)
    """
    Q1 = df[feature_columns].quantile(0.25)
    Q3 = df[feature_columns].quantile(0.75)
    IQR = Q3 - Q1

    outlier_mask = (
        (df[feature_columns] < (Q1 - 1.5 * IQR)) |
        (df[feature_columns] > (Q3 + 1.5 * IQR))
    )

    total_outliers = outlier_mask.sum().sum()
    print(f"[INFO] Total outlier terdeteksi: {total_outliers}")

    return df

# FEATURE SCALING
def feature_scaling(df: pd.DataFrame, scaler_path: str = None):
    """
    Standarisasi fitur numerik menggunakan StandardScaler
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return X_scaled, y


# SAVE PROCESSED DATA
def save_processed_data(X, y, output_path):
    """
    Simpan dataset hasil preprocessing ke CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = X.copy()
    df["target"] = y.values
    df.to_csv(output_path, index=False)


# MAIN PREPROCESSING PIPELINE
def preprocess_pipeline(
    save_scaler: bool = True,
    scaler_path: str = "preprocessing/breast_cancer_preprocessing/standard_scaler.pkl"
):
    """
    Pipeline preprocessing utama
    Mengembalikan data siap dilatih
    """

    print("[INFO] Load dataset...")
    df = load_dataset()

    print("[INFO] Handle missing values...")
    df = handle_missing_values(df)

    print("[INFO] Remove duplicates...")
    df = remove_duplicates(df)

    print("[INFO] Detect outliers...")
    feature_columns = df.drop("target", axis=1).columns.tolist()
    df = detect_outliers_iqr(df, feature_columns)

    print("[INFO] Feature scaling...")
    X_scaled, y = feature_scaling(
        df,
        scaler_path=scaler_path if save_scaler else None
    )

    print("[INFO] Preprocessing selesai. Data siap untuk training.")
    return X_scaled, y

# EXECUTION
if __name__ == "__main__":
    X, y = preprocess_pipeline(
        save_scaler=True,
        scaler_path="preprocessing/breast_cancer_preprocessing/standard_scaler.pkl"
    )

    save_processed_data(
        X,
        y,
        output_path="preprocessed/breast_cancer_preprocessed.csv"
    )

    print("[INFO] Dataset hasil preprocessing berhasil disimpan.")
