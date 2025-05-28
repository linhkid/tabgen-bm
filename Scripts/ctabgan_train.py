#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath("."))
import pandas as pd
import numpy as np
import argparse
import torch
import random
import json
from ctabgan.model.ctabgan import CTABGAN

def improve_reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_ctabgan(args):
    # Setup paths
    dataset_path = os.path.join(args.real_data_dir, args.dataset_name)
    x_path = os.path.join(dataset_path, "x_train.csv")
    y_path = os.path.join(dataset_path, "y_train.csv")
    save_path = os.path.join(args.synthetic_data_dir, args.dataset_name, "ctabgan")
    os.makedirs(save_path, exist_ok=True)

    # Load info.json
    info_path = os.path.join(args.real_data_dir, args.dataset_name, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    # Determine column indices
    cat_idx = info.get("cat_col_idx", [])
    num_idx = info.get("num_col_idx", [])    

    # Load data
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    df = pd.concat([X, y], axis=1)
    y_col = y.columns[0]

    # Convert indices to column names
    feature_columns = X.columns.tolist()
    categorical_columns = [feature_columns[i] for i in cat_idx]
    categorical_columns.append(y_col)
    integer_columns = [feature_columns[i] for i in num_idx]
    
    # Set reproducibility and device
    improve_reproducibility(args.seed)
    device = "cuda" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(args.gpu_id)

    # Determine epochs based on dataset size (300 for small, 150 for others)
    epochs = 300 if args.size_category == "small" else 150

    print(f"\n--- Running CTAB-GAN for dataset {args.dataset_name} ---")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")

    # Use a temporary CSV path to avoid NoneType issue
    temp_csv_path = os.path.join(save_path, "temp_data.csv")
    df.to_csv(temp_csv_path, index=False)
    
    # Create CTABGAN model
    model = CTABGAN(
        raw_csv_path=temp_csv_path,
        test_ratio=0.0,
        categorical_columns=categorical_columns,
        log_columns=[],
        mixed_columns={},
        integer_columns=integer_columns,
        problem_type={"Classification": y_col},
        epochs=epochs
    )
    
    # Train the model
    model.fit()
    
    # Generate synthetic samples
    synth_df = model.generate_samples()
    
    # Split features and target
    x_synth = synth_df.drop(columns=[y_col])
    y_synth = synth_df[[y_col]]

    # Save to CSV
    x_synth.to_csv(os.path.join(save_path, "x_synth.csv"), index=False)
    y_synth.to_csv(os.path.join(save_path, "y_synth.csv"), index=False)

    # Save merged file for reference
    synth_df.to_csv(os.path.join(save_path, "synthetic_data.csv"), index=False)
    
    # Clean up temporary file
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

    print(f"Synthetic data saved at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset folder name under real_data_dir")
    parser.add_argument("--real_data_dir", type=str, default="Data", help="Path to real data directory")
    parser.add_argument("--synthetic_data_dir", type=str, default="Synthetic", help="Path to save synthetic outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID to use (-1 for CPU)")
    parser.add_argument(
        "--size_category", type=str, required=True,
        choices=["small", "medium", "large"],
        help="Dataset size category: small → 300 epochs, medium/large → 150"
    )

    args = parser.parse_args()
    run_ctabgan(args)