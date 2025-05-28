import os
import sys

sys.path.append(os.path.abspath("."))
import argparse
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from ganblr.model.ganblr import GANBLR


def main():
    parser = argparse.ArgumentParser(description="Train GANBLR and generate synthetic data")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., adult)')
    parser.add_argument('--size_category', type=str, required=True, choices=['small', 'medium', 'large'],
                        help='Dataset size category (small/medium/large)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    args.epochs = 150 if args.size_category == 'large' else 100

    model_name = "ganblr"
    dataset_name = args.dataset
    data_dir = f"Data/{dataset_name}"
    save_dir = os.path.join("Synthetic", dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")

    # === Set Random Seed for Reproducibility ===
    seed = args.seed
    print(f"Using random seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    # === Load Training Data ===
    X = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path).values.ravel()
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    model = GANBLR()
    model.fit(X, y, k=0, epochs=args.epochs, batch_size=64)

    syn_data = model.sample(X.shape[0])
    df_synth = pd.DataFrame(syn_data)
    x_synth = df_synth.iloc[:, :-1]
    y_synth = df_synth.iloc[:, -1]

    x_synth.to_csv(os.path.join(save_dir, "x_synth.csv"), index=False)
    y_synth.to_csv(os.path.join(save_dir, "y_synth.csv"), index=False, header=True)
    print(f"\n Synthetic data saved to: {save_dir}")


if __name__ == "__main__":
    main()