# automate_Muhammad-Firdaus.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats

def preprocess_wine_data(input_csv: str, output_csv: str) -> None:
    """
    Membaca dataset wine mentah, melakukan preprocessing lengkap, dan menyimpan hasilnya.
    """
    # Load
    df = pd.read_csv(input_csv)

    # Label klasifikasi
    df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    df.drop('quality', axis=1, inplace=True)

    # Hapus duplikat
    df.drop_duplicates(inplace=True)

    # Hapus outlier
    z_scores = stats.zscore(df.drop('quality_label', axis=1))
    df = df[(abs(z_scores) < 3).all(axis=1)]

    # Standarisasi
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df.drop('quality_label', axis=1))
    df_scaled = pd.DataFrame(features_scaled, columns=df.drop('quality_label', axis=1).columns)

    # Tambahkan label
    df_scaled['quality_label'] = df['quality_label'].reset_index(drop=True)

    # Simpan hasil
    df_scaled.to_csv(output_csv, index=False)
    print(f"[INFO] Preprocessing selesai. Hasil disimpan di: {output_csv}")


if __name__ == "__main__":
    preprocess_wine_data(
        input_csv="namadataset_raw/winequality-red.csv",
        output_csv="namadataset_preprocessing/wine_preprocessed.csv"
    )
