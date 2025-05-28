import os
import sys
sys.path.append(os.path.abspath("."))
import argparse
import pandas as pd
import numpy as np
import random
from ganblr.model.rlig import RLiG

def main():
    parser = argparse.ArgumentParser(description="Train RLiG (GANBLR++) and generate synthetic data")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., adult)')
    parser.add_argument('--size_category', type=str, required=True, choices=['small', 'medium', 'large'], help='Dataset size category (small/medium/large)')
    parser.add_argument('--k', type=int, default=1, help='Parameter k for RLiG model (default: 1)')
    parser.add_argument('--episodes', type=int, default=2, help='Number of episodes for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs per episode (default: based on size)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Number of warmup epochs (default: 1)')
    parser.add_argument('--n', type=int, default=3, help='The n-steps in generative states (default: 3)')
    parser.add_argument('--gan', type=int, default=1, help='Whether to use GAN structure (1=yes, 0=no) (default: 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    # Set default epochs based on size if not provided
    if args.epochs is None:
        args.epochs = 150 if args.size_category == 'large' else 100
    
    model_name = "rlig"  # or "ganblr++"
    dataset_name = args.dataset
    data_dir = f"Data/{dataset_name}"
    save_dir = os.path.join("Synthetic", dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")

    # === Set Random Seed for Reproducibility ===
    np.random.seed(args.seed)
    random.seed(args.seed)

    # === Load Training Data ===
    X = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path)
    
    # Convert y to Series if it's a DataFrame with one column
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape if hasattr(y, 'shape') else 'Series'}")
    print(f"Training with parameters: k={args.k}, episodes={args.episodes}, epochs={args.epochs}, batch_size={args.batch_size}")

    # Initialize and train the model
    model = RLiG()
    try:
        model.fit(
            x=X, 
            y=y, 
            k=args.k,
            batch_size=args.batch_size, 
            episodes=args.episodes,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            verbose=1,
            gan=args.gan,
            n=args.n
        )
        
        # Generate synthetic data
        print("Generating synthetic data...")
        syn_data = model.sample(X.shape[0])
        df_synth = pd.DataFrame(syn_data)
        
        # Split synthetic data into features and target
        x_synth = df_synth.iloc[:, :-1]
        y_synth = df_synth.iloc[:, -1]

        # Save synthetic data
        x_synth.to_csv(os.path.join(save_dir, "x_synth.csv"), index=False)
        y_synth.to_csv(os.path.join(save_dir, "y_synth.csv"), index=False, header=True)
        print(f"Synthetic data saved to: {save_dir}")
        
        # Optional: Evaluate the model
        print("Evaluating model performance...")
        accuracy = model.evaluate(X, y, model='lr')
        print(f"TSTR Accuracy with Logistic Regression: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()