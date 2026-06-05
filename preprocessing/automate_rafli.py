import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(raw_path, output_path):
    print("Mulai preprocessing data...")
    # Load
    df = pd.read_csv(raw_path)
    
    # Drop kolom
    df_clean = df.drop(columns=['customer_id'], errors='ignore')
    
    # Encoding
    le = LabelEncoder()
    df_clean['gender'] = le.fit_transform(df_clean['gender'])
    df_clean = pd.get_dummies(df_clean, columns=['country'], drop_first=True)
    
    # Scaling
    scaler = StandardScaler()
    num_cols = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
    
    # Hanya lakukan scaling jika kolom tersebut ada
    cols_to_scale = [col for col in num_cols if col in df_clean.columns]
    if cols_to_scale:
        df_clean[cols_to_scale] = scaler.fit_transform(df_clean[cols_to_scale])
    
    # Buat direktori output jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Simpan
    df_clean.to_csv(output_path, index=False)
    print(f"Data berhasil disimpan di {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_file = os.path.join(current_dir, "..", "dataset_raw", "Churn_Modelling.csv")
    output_file = os.path.join(current_dir, "..", "dataset_preprocessing", "dataset_clean.csv")

    preprocess_data(raw_file, output_file)

# Ini komentar untuk mengetest action 