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
import warnings

warnings.filterwarnings("ignore")

# Import CTGAN from the original implementation
try:
    from ctgan import CTGAN
except ImportError:
    print("CTGAN not found, attempting to install...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "ctgan"])
    from ctgan import CTGAN


def improve_reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_ctgan(args):
    # Setup paths
    if args.data_dir:
        # If a specific data directory is provided, use it directly
        dataset_path = args.data_dir
    else:
        # Otherwise, construct from real_data_dir and dataset_name
        dataset_path = os.path.join(args.real_data_dir, args.dataset_name)
    
    x_path = os.path.join(dataset_path, "x_train.csv")
    y_path = os.path.join(dataset_path, "y_train.csv")
    
    # Create a seed-specific save directory
    save_path = os.path.join(args.synthetic_data_dir, args.dataset_name, "ctgan", f"seed{args.seed}")
    os.makedirs(save_path, exist_ok=True)

    # Load info.json
    if args.data_dir:
        # Try to find info.json in the provided data directory first
        info_path = os.path.join(args.data_dir, "info.json")
        if not os.path.exists(info_path):
            # If not found, look in the parent directory
            parent_dir = os.path.dirname(args.data_dir)
            info_path = os.path.join(parent_dir, "info.json")
    else:
        info_path = os.path.join(args.real_data_dir, args.dataset_name, "info.json")
    
    with open(info_path, "r") as f:
        info = json.load(f)

    # Determine column indices and types
    cat_idx = info.get("cat_col_idx", [])
    num_idx = info.get("num_col_idx", [])

    # Load data
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    df = pd.concat([X, y], axis=1)
    y_col = y.columns[0]

    # Convert indices to column names
    feature_columns = X.columns.tolist()

    # Create metadata for CTGAN
    discrete_columns = [feature_columns[i] for i in cat_idx]
    if y_col not in discrete_columns:
        discrete_columns.append(y_col)

    # Set reproducibility and determine batch size
    improve_reproducibility(args.seed)

    # Determine batch size based on dataset size
    if args.size_category == "small":
        batch_size = 500
        epochs = 300
    elif args.size_category == "medium":
        batch_size = 1000
        epochs = 150
    else:  # large
        batch_size = 2000
        epochs = 100

    # Determine device
    device = "cuda" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print(f"\n--- Running CTGAN for dataset {args.dataset_name} ---")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Discrete columns: {discrete_columns}")

    # Create temporary file to track progress
    with open(os.path.join(save_path, "training_started.txt"), "w") as f:
        f.write("Training started")

    try:
        # Initialize and train CTGAN model
        model = CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=128,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            generator_lr=2e-4,
            discriminator_lr=2e-4,
            discriminator_steps=1,
            verbose=True,
            cuda=(device == "cuda")
        )

        # Convert column names to strings for CTGAN
        df.columns = df.columns.astype(str)

        # Train model
        model.fit(df, discrete_columns=discrete_columns)

        # Generate synthetic data
        num_samples = len(df)
        print(f"Generating {num_samples} synthetic samples...")
        synth_df = model.sample(num_samples)

        # Split features and target
        x_synth = synth_df.drop(columns=[y_col])
        y_synth = synth_df[[y_col]]

        # Save to CSV
        x_synth.to_csv(os.path.join(save_path, "x_synth.csv"), index=False)
        y_synth.to_csv(os.path.join(save_path, "y_synth.csv"), index=False)

        # Save merged file for reference
        synth_df.to_csv(os.path.join(save_path, "synthetic_data.csv"), index=False)

        print(f"Synthetic data saved at: {save_path}")

        # Remove the progress tracking file
        if os.path.exists(os.path.join(save_path, "training_started.txt")):
            os.remove(os.path.join(save_path, "training_started.txt"))

    except Exception as e:
        print(f"Error training CTGAN model: {str(e)}")
        # If training fails, create empty output files to avoid breaking the pipeline
        pd.DataFrame(columns=X.columns).to_csv(os.path.join(save_path, "x_synth.csv"), index=False)
        pd.DataFrame(columns=y.columns).to_csv(os.path.join(save_path, "y_synth.csv"), index=False)

        # Save the error message
        with open(os.path.join(save_path, "error.txt"), "w") as f:
            f.write(str(e))

        print(f"Created empty output files at: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset folder name under real_data_dir")
    parser.add_argument("--real_data_dir", type=str, default="Data", help="Path to real data directory")
    parser.add_argument("--data_dir", type=str, default=None, help="Custom data directory to use directly (overrides real_data_dir)")
    parser.add_argument("--synthetic_data_dir", type=str, default="Synthetic", help="Path to save synthetic outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID to use (-1 for CPU)")
    parser.add_argument(
        "--size_category", type=str, required=True,
        choices=["small", "medium", "large"],
        help="Dataset size category: small → 300 epochs, medium/large → 150/100"
    )

    args = parser.parse_args()
    run_ctgan(args)