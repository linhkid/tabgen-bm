import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, na_values='?')
    df.dropna(axis=1, how='all', inplace=True)
    return df

def is_numerical(col):
    return pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)

def process_numerical_columns(df):
    df_copy = df.copy()

    for col in df.columns:
        if is_numerical(df[col]):
            try:
                col_data = pd.to_numeric(df[col], errors='coerce')
                col_data = col_data.fillna(col_data.median())

                unique_vals = col_data.nunique(dropna=True)
                if unique_vals <= 1:
                    print(f"[!] Skipping column '{col}' — constant or all NaN")
                    continue

                df_copy[col] = col_data

            except Exception as e:
                print(f"Error processing column '{col}': {e}")
                continue

    return df_copy

def encode_categorical_columns(df):
    df_copy = df.copy()
    for col in df.columns:
        if not is_numerical(df_copy[col]):
            try:
                df_copy[col] = df_copy[col].astype(str).str.strip()
                df_copy[col] = df_copy[col].replace(['missing'], 'Missing')
                df_copy[col] = df_copy[col].fillna('Missing').astype(str)

                # Sort values, ensuring 'Missing' comes first
                unique_vals = sorted(df_copy[col].unique(), key=lambda x: (x != 'Missing', x))
                
                le = LabelEncoder()
                le.classes_ = np.array(unique_vals)
                df_copy[col] = le.transform(df_copy[col])
                
            except Exception as e:
                print(f"Skipping categorical column '{col}' due to error: {e}")
    return df_copy

def preprocess(file_path, output_path):
    print(f"Preprocessing: {file_path}")
    df = load_and_clean_data(file_path)

    # Separate target column (assumed to be last)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Process features
    X = process_numerical_columns(X)
    X = encode_categorical_columns(X)
    X = X.astype(int)

    # Encode target
    y = y.fillna('Missing').astype(str)
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    y_encoded = pd.Series(y_encoded, name=y.name)

    # Combine and save
    df_processed = pd.concat([X, y_encoded], axis=1)
    df_processed.columns = [str(i) for i in range(df_processed.shape[1])]
    df_processed.to_csv(output_path, index=False)
    print(f"Saved preprocessed discrete dataset to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessor for Discrete Integer Datasets')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    args = parser.parse_args()

    preprocess(args.input, args.output)
