import pandas as pd
import numpy as np
import json
import os
import argparse
import random
from sklearn.model_selection import train_test_split

def safe_train_test_split(x_df, y_df, cat_idx, test_size=0.2, seed_start=42):
    total_rows = x_df.shape[0]
    cat_columns = cat_idx
    seed = seed_start

    while True:
        x_train, x_test, y_train, y_test = train_test_split(
            x_df, y_df, test_size=test_size, random_state=seed, stratify=y_df
        )

        flag = 0
        for col in cat_columns:
            train_vals = set(x_train.iloc[:, col])
            test_vals = set(x_test.iloc[:, col])
            if not test_vals.issubset(train_vals):
                flag = 1
                break

        if flag == 0:
            print(f"Split successful with seed {seed}")
            return x_train, x_test, y_train, y_test
        else:
            seed += 1

def convert_csv_to_npy(dataset_name, data_dir="Data", test_size=0.2, random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)

    # Just use the provided data_dir directly, don't append dataset_name again
    # as data_dir already includes the dataset name and seed
    dataset_path = data_dir
    
    # Try to find info.json in multiple locations
    # First try the data directory itself
    info_path = os.path.join(dataset_path, "info.json")
    if not os.path.exists(info_path):
        # If not found, try parent directory
        parent_dir = os.path.dirname(dataset_path)
        info_path = os.path.join(parent_dir, "info.json")
        if not os.path.exists(info_path):
            # If still not found, try standard Data/dataset_name/info.json
            info_path = os.path.join("Data", dataset_name, "info.json")
            if not os.path.exists(info_path):
                # If still not found, create a default info.json
                print(f"Warning: info.json not found. Creating a default version at {dataset_path}")
                x_df = pd.read_csv(os.path.join(dataset_path, "x_train.csv"))
                
                # Detect numerical and categorical columns
                num_idx = []
                cat_idx = []
                for i, col in enumerate(x_df.columns):
                    if pd.api.types.is_numeric_dtype(x_df[col]) and x_df[col].nunique() > 10:
                        num_idx.append(i)
                    else:
                        cat_idx.append(i)
                
                # Create default info
                info = {
                    "num_col_idx": num_idx,
                    "cat_col_idx": cat_idx
                }
                
                # Save the default info
                info_path = os.path.join(dataset_path, "info.json")
                with open(info_path, "w") as f:
                    json.dump(info, f, indent=2)
                print(f"Created default info.json with num_cols: {num_idx}, cat_cols: {cat_idx}")

    with open(info_path, "r") as f:
        info = json.load(f)

    num_idx = info.get("num_col_idx", [])
    cat_idx = info.get("cat_col_idx", [])

    x_df = pd.read_csv(os.path.join(dataset_path, "x_train.csv"))
    y_df = pd.read_csv(os.path.join(dataset_path, "y_train.csv"))

    if len(x_df) != len(y_df):
        raise ValueError("x_train.csv and y_train.csv must have the same number of rows.")

    # 🚀 Use safe split logic here
    x_train, x_test, y_train, y_test = safe_train_test_split(
        x_df, y_df, cat_idx, test_size=test_size, seed_start=random_state
    )

    # Numerical features
    X_num_train = x_train.iloc[:, num_idx].to_numpy() if num_idx else np.empty((len(x_train), 0))
    X_num_test = x_test.iloc[:, num_idx].to_numpy() if num_idx else np.empty((len(x_test), 0))

    # Categorical features
    X_cat_train = x_train.iloc[:, cat_idx].astype(int).to_numpy() if cat_idx else np.empty((len(x_train), 0), dtype=int)
    X_cat_test = x_test.iloc[:, cat_idx].astype(int).to_numpy() if cat_idx else np.empty((len(x_test), 0), dtype=int)

    # Save
    np.save(os.path.join(dataset_path, "X_num_train.npy"), X_num_train)
    np.save(os.path.join(dataset_path, "X_num_test.npy"), X_num_test)
    np.save(os.path.join(dataset_path, "X_cat_train.npy"), X_cat_train)
    np.save(os.path.join(dataset_path, "X_cat_test.npy"), X_cat_test)
    np.save(os.path.join(dataset_path, "y_train.npy"), y_train.to_numpy().ravel())
    np.save(os.path.join(dataset_path, "y_test.npy"), y_test.to_numpy().ravel())

    print("Saved .npy files:")
    print(f"- X_num_train.npy (shape={X_num_train.shape})")
    print(f"- X_num_test.npy  (shape={X_num_test.shape})")
    print(f"- X_cat_train.npy (shape={X_cat_train.shape})")
    print(f"- X_cat_test.npy  (shape={X_cat_test.shape})")
    print(f"- y_train.npy     (shape={y_train.shape})")
    print(f"- y_test.npy      (shape={y_test.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., adult)')
    parser.add_argument('--data_dir', type=str, default='Data', help='Base directory containing dataset folders')
    parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of data for test split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    convert_csv_to_npy(args.dataset, args.data_dir, args.test_size, args.seed)
