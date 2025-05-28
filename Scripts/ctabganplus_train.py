import os
import sys
sys.path.append(os.path.abspath("."))
import pandas as pd
import numpy as np
import argparse
import torch
import random
import zero
import json
from ctabganplus.model.ctabgan import CTABGAN

def improve_reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    zero.improve_reproducibility(seed)

def run_ctabganplus(args):
    # Setup paths
    dataset_path = os.path.join(args.real_data_dir, args.dataset_name)
    x_path = os.path.join(dataset_path, "x_train.csv")
    y_path = os.path.join(dataset_path, "y_train.csv")
    save_path = os.path.join(args.synthetic_data_dir, args.dataset_name, "ctabgan_plus")
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
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"


    # Determine epochs based on paper (300 for small, 150 for others)
    epochs = 300 if args.size_category == "small" else 150

    print(f"\n--- Running CTAB-GAN+ for dataset {args.dataset_name} ---")

    model = CTABGAN(
        df=df,
        test_ratio=0.0,
        categorical_columns=categorical_columns,
        log_columns=[],
        mixed_columns={},
        general_columns=[],
        non_categorical_columns=[],
        integer_columns=integer_columns,
        problem_type={"output": y_col},
        class_dim=(256, 256, 256, 256),
        random_dim=100,
        num_channels=64,
        l2scale=1e-5,
        batch_size=500,
        epochs=epochs,
        lr=2e-4,
        device=device
    )


    model.fit()
    synth_df = model.generate_samples(num_samples=len(X), seed=args.seed)
    x_synth = synth_df.drop(columns=[y_col])
    y_synth = synth_df[[y_col]]

    # Save to CSV
    x_synth.to_csv(os.path.join(save_path, "x_synth.csv"), index=False)
    y_synth.to_csv(os.path.join(save_path, "y_synth.csv"), index=False)

    print(f"Synthetic data saved at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset folder name under real_data_dir")
    parser.add_argument("--real_data_dir", type=str, default="Data", help="Path to real data directory")
    parser.add_argument("--synthetic_data_dir", type=str, default="Synthetic", help="Path to save synthetic outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Training device")
    parser.add_argument(
        "--size_category", type=str, required=True,
        choices=["small", "medium", "large"],
        help="Dataset size category: small → 300 epochs, medium/large → 150"
    )

    args = parser.parse_args()
    run_ctabganplus(args)
