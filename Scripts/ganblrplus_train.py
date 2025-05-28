import os
import sys

sys.path.append(os.path.abspath("."))
import argparse
import pandas as pd
import numpy as np
import random
import json
from ganblr.model.ganblrpp import GANBLRPP


def main():
    parser = argparse.ArgumentParser(description="Train GANBLR++ model and generate synthetic data")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., adult)')
    parser.add_argument('--size_category', type=str, required=True, choices=['small', 'medium', 'large'],
                        help='Dataset size category (small/medium/large)')
    parser.add_argument('--k', type=int, default=2, help='k parameter for KDB (default: 2)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (default: based on size_category)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--data_dir', type=str, default=None, help='Custom data directory to use instead of default')
    args = parser.parse_args()

    # Set epochs based on size category if not specified
    if args.epochs is None:
        args.epochs = 200 if args.size_category == 'small' else 100

    model_name = "ganblrplus"
    dataset_name = args.dataset
    
    # Use custom data directory if provided
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = f"Data/{dataset_name}"
        
    save_dir = os.path.join("Synthetic", dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")

    # Load info.json to determine numerical columns
    info_path = os.path.join(data_dir, "info.json")
    try:
        with open(info_path, "r") as f:
            info = json.load(f)
        # Get numerical column indices
        numerical_columns = info.get("num_col_idx", [])
        print(f"Identified numerical columns: {numerical_columns}")
    except (FileNotFoundError, json.JSONDecodeError):
        # If info.json doesn't exist or is invalid, infer numerical columns from data
        print("No info.json found, attempting to infer numerical columns from data")
        numerical_columns = []

    # === Set Random Seed for Reproducibility ===
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    # === Load Training Data ===
    X = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path).values.ravel()
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    # If numerical_columns is empty, try to infer them from the data types
    if not numerical_columns:
        for i, col in enumerate(X.columns):
            if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 10:
                numerical_columns.append(i)
        print(f"Inferred numerical columns: {numerical_columns}")

    # If still no numerical columns are found, create a dummy numerical column
    # This allows GANBLRPP to run without error even on purely categorical datasets
    if not numerical_columns:
        print("No numerical columns detected. Adding a dummy numerical column for GANBLRPP.")
        # Create a copy of the dataframe to avoid modifying the original
        X = X.copy()
        # Add a dummy numerical column with constant value
        X['_dummy_num'] = 0.0
        # Set this as the only numerical column (at the last index)
        numerical_columns = [X.shape[1] - 1]

    # Initialize GANBLRPP model with numerical columns
    model = GANBLRPP(numerical_columns=numerical_columns, random_state=seed)

    print(f"Training GANBLR++ model with k={args.k}, epochs={args.epochs}")
    # Convert to numpy arrays for training
    X_arr = X.values
    # Train the model
    model.fit(X_arr, y, k=args.k, epochs=args.epochs, batch_size=args.batch_size)

    # Generate synthetic data
    print("Generating synthetic data...")
    syn_data = model.sample(X.shape[0])

    # Get original column names (before any dummy column was added)
    original_x_cols = pd.read_csv(x_train_path).columns.tolist()
    y_col = pd.read_csv(y_train_path).columns[0]

    # Split features and target
    x_synth_data = syn_data[:, :-1]

    # Remove the dummy column if it was added
    if '_dummy_num' in X.columns:
        # The dummy column is the last one
        x_synth_data = x_synth_data[:, :-1]

    # Create DataFrame with original column names
    x_synth = pd.DataFrame(x_synth_data, columns=original_x_cols)
    y_synth = pd.DataFrame(syn_data[:, -1], columns=[y_col])

    # Save synthetic data
    x_synth.to_csv(os.path.join(save_dir, "x_synth.csv"), index=False)
    y_synth.to_csv(os.path.join(save_dir, "y_synth.csv"), index=False)

    # Save combined data for reference
    combined_df = pd.concat([x_synth, y_synth], axis=1)
    combined_df.to_csv(os.path.join(save_dir, "synthetic_data.csv"), index=False)

    print(f"\nSynthetic data saved to: {save_dir}")


if __name__ == "__main__":
    main()