import os
import argparse
import pandas as pd
import numpy as np
import torch
import random
import sys

sys.path.append(os.path.abspath("."))

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_train, get_input_generate, split_num_cat_target, recover_data
from tabsyn.diffusion_utils import sample

from scipy.spatial.distance import cdist


def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:, col])
        dist = cdist(X_synth[:, col][:, np.newaxis], uniq[:, np.newaxis])
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth


def run_tabsyn(dataset_name, seed=42, data_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_name = "tabsyn"
    # Use custom data directory if provided
    if data_dir is None:
        data_dir = os.path.join("Data", dataset_name)
        
    # Create a seed-specific save directory
    save_dir = os.path.join("Synthetic", dataset_name, model_name, f"seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    # Load VAE latent embeddings
    args_obj = argparse.Namespace(dataname=dataset_name, device=device)
    train_z, _, _, ckpt_dir, _ = get_input_train(args_obj)

    z = train_z
    in_dim = z.shape[1]
    mean = z.mean(0)
    z = (z - mean) / 2

    loader = torch.utils.data.DataLoader(z, batch_size=4096, shuffle=True, num_workers=4)

    # === Initialize Diffusion Model ===
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

    # === Training ===
    best_loss = float('inf')
    patience = 0
    for epoch in range(10000 + 1):
        model.train()
        total_loss, count = 0.0, 0
        for batch in loader:
            x = batch.float().to(device)
            loss = model(x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
            count += len(x)

        avg_loss = total_loss / count
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            best_state = model.state_dict()
        else:
            patience += 1
            if patience >= 500:
                break

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: loss = {avg_loss:.6f}")

    print(f"Best loss: {best_loss:.6f}")
    model.load_state_dict(best_state)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))

    model.eval()

    # === Sampling ===
    num_samples = len(train_z)
    sample_dim = in_dim
    z_synth = sample(model.denoise_fn_D, num_samples, sample_dim, device=device)
    z_synth = z_synth * 2 + mean.to(device)
    z_synth = z_synth.float().cpu().numpy()

    # === Decode ===
    _, _, _, _, info_gen, num_inv, cat_inv = get_input_generate(args_obj)
    syn_num, syn_cat, syn_target = split_num_cat_target(z_synth, info_gen, num_inv, cat_inv, device)
    # === Optional: Round numerical columns to closest real bin values ===
    num_col_idx = info_gen['num_col_idx']
    X_real = pd.read_csv(os.path.join(data_dir, "x_train.csv")).values
    syn_num = round_columns(X_real[:, num_col_idx], syn_num, list(range(len(num_col_idx))))

    syn_df = recover_data(syn_num, syn_cat, syn_target, info_gen)

    idx_name_map = info_gen.get('idx_name_mapping', None)

    if idx_name_map is not None:
        idx_name_map = {int(k): v for k, v in idx_name_map.items()}
        syn_df.rename(columns=idx_name_map, inplace=True)
        x_synth = syn_df.drop(columns=[v for k, v in idx_name_map.items() if k in info_gen['target_col_idx']])
        y_synth = syn_df[[v for k, v in idx_name_map.items() if k in info_gen['target_col_idx']]]
    else:
        # fallback: assume default column naming
        target_cols = info_gen['target_col_idx']
        x_synth = syn_df.drop(syn_df.columns[target_cols], axis=1)
        y_synth = syn_df.iloc[:, target_cols]

    x_synth.to_csv(os.path.join(save_dir, "x_synth.csv"), index=False)
    y_synth.to_csv(os.path.join(save_dir, "y_synth.csv"), index=False)

    print(f"\n Synthetic data saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--data_dir', type=str, default=None, help='Custom data directory to use instead of default')
    args = parser.parse_args()

    run_tabsyn(args.dataset, args.seed, args.data_dir)