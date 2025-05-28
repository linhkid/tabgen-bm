import os
import random
import numpy as np
import pandas as pd
import torch
import argparse
import sys
sys.path.append(os.path.abspath("."))

from great import GReaT
from transformers import TrainingArguments
from great.great_trainer import GReaTTrainer
from great.great_dataset import GReaTDataset, GReaTDataCollator

def run_great(dataset_name, real_data_dir="Data", synthetic_dir="Synthetic", seed=42):
    model_name = "great"
    data_path = os.path.join(real_data_dir, dataset_name)
    save_path = os.path.join(synthetic_dir, dataset_name, model_name)
    os.makedirs(save_path, exist_ok=True)

    x_train_path = os.path.join(data_path, "x_train.csv")
    y_train_path = os.path.join(data_path, "y_train.csv")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path).values.ravel()
    df_train = X.copy()
    df_train["label"] = y
    print(f"Loaded '{dataset_name}': X={X.shape}, y={y.shape}")

    model_pt_path = os.path.join(save_path, "model.pt")

    # === Load or Train ===
    if os.path.exists(model_pt_path):
        print("Loading pre-trained GReaT model...")
        model = GReaT.load_from_dir(save_path)
    else:
        print("Training new GReaT model...")
        model = GReaT(
            llm="distilgpt2",
            experiment_dir=save_path,
            epochs=1,
            batch_size=32,
            report_to=[],
        )
        # Prepare dataset
        model._update_column_information(df_train)
        model._update_conditional_information(df_train, "label")
        great_ds = GReaTDataset.from_pandas(df_train)
        great_ds.set_tokenizer(model.tokenizer)

        training_args = TrainingArguments(
            output_dir=save_path,
            num_train_epochs=1,
            per_device_train_batch_size=32,
            report_to=[],
            save_strategy="no",
            logging_strategy="no",
        )
        trainer = GReaTTrainer(
            model.model,
            training_args,
            train_dataset=great_ds,
            tokenizer=model.tokenizer,
            data_collator=GReaTDataCollator(model.tokenizer),
        )
        
        # === Manual Early Stopping Loop ===
        patience_limit = 5
        best_loss = float("inf")
        patience = 0

        for epoch in range(1):  # max epochs
            output = trainer.train()
            train_loss = output.training_loss
            print(f"Epoch {epoch+1} | Loss: {train_loss:.4f}")

            if train_loss < best_loss or epoch == 0:
                best_loss = train_loss
                patience = 0
                print("New best model found. Saving...")
                trainer.save_model(save_path)
                model.save(save_path)
            else:
                patience += 1
                print(f"No improvement. Patience {patience}/{patience_limit}")
                if patience >= patience_limit:
                    print("Early stopping triggered.")
                    break
        model = GReaT.load_from_dir(save_path)
        #model.model.load_state_dict(torch.load(os.path.join(save_path, "model.pt")))
        print(f"Model saved to: {save_path}")

    # === Sampling ===
    print(f"Sampling using device: {device}")
    print("Sampling synthetic data...")
    df_synth = model.sample(
        n_samples=len(df_train),
        guided_sampling=False,
        temperature=0.7,
        device=device,
        k=100
    ).reset_index(drop=True)

    if "label" not in df_synth.columns:
        raise ValueError("Missing 'label' column in synthetic output.")

    x_synth = df_synth.drop(columns=["label"])
    y_synth = df_synth["label"]

    x_synth.to_csv(os.path.join(save_path, "x_synth.csv"), index=False)
    y_synth.to_csv(os.path.join(save_path, "y_synth.csv"), index=False)

    print(f"GReaT synthetic data saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--real_data_dir', default='Data', help='Base directory containing dataset folders')
    parser.add_argument('--data_dir', type=str, default=None, help='Custom data directory to use directly (overrides real_data_dir)')
    parser.add_argument('--synthetic_dir', default='Synthetic')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Use specific data_dir if provided, otherwise use the real_data_dir/dataset path
    data_dir = args.data_dir if args.data_dir else os.path.join(args.real_data_dir, args.dataset)

    run_great(
        dataset_name=args.dataset,
        real_data_dir=os.path.dirname(data_dir),  # Get the parent directory
        synthetic_dir=args.synthetic_dir,
        seed=args.seed
    )
