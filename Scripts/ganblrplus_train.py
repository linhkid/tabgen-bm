import os
import sys
sys.path.append(os.path.abspath("."))
import argparse
import pandas as pd
import numpy as np
import random
from ganblr.model.rlig import RLiG

def main():
    parser = argparse.ArgumentParser(description="Train GANBLR++ (RLiG) and generate synthetic data")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., adult)')
    parser.add_argument('--size_category', type=str, required=True, choices=['small', 'medium', 'large'], help='Dataset size category (small/medium/large)')
    parser.add_argument('--k', type=int, default=2, help='k parameter for KDB (default: 2)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of RL episodes (default: 5)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (default: based on size_category)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set epochs based on size category if not specified
    if args.epochs is None:
        args.epochs = 200 if args.size_category == 'small' else 100
    
    model_name = "ganblrplus"
    dataset_name = args.dataset
    data_dir = f"Data/{dataset_name}"
    save_dir = os.path.join("Synthetic", dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")

    # === Set Random Seed for Reproducibility ===
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    # === Load Training Data ===
    X = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path).values.ravel()
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    # Initialize RLiG model (GANBLR++)
    model = RLiG()
    
    print(f"Training GANBLR++ model with k={args.k}, episodes={args.episodes}, epochs={args.epochs}")
    # Train the model
    model.fit(X, y, 
              k=args.k, 
              episodes=args.episodes, 
              epochs=args.epochs, 
              batch_size=args.batch_size)

    # Generate synthetic data
    print("Generating synthetic data...")
    syn_data = model.sample(X.shape[0])
    df_synth = pd.DataFrame(syn_data)
    
    # Split features and target
    x_synth = df_synth.iloc[:, :-1]
    y_synth = df_synth.iloc[:, -1]

    # Save synthetic data
    x_synth.to_csv(os.path.join(save_dir, "x_synth.csv"), index=False)
    y_synth.to_csv(os.path.join(save_dir, "y_synth.csv"), index=False, header=True)
    print(f"\nSynthetic data saved to: {save_dir}")

if __name__ == "__main__":
    main()