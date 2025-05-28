import os
import sys
sys.path.append(os.path.abspath("."))
import torch
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder
from tabddpm.model.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from tabddpm.utils_train import get_model, make_dataset, update_ema
from copy import deepcopy
from tabddpm.lib.data import Transformations, prepare_fast_dataloader
import zero


def run_tabddpm(dataset_name, real_data_dir="Data", synthetic_dir="Synthetic", seed=42):
    # === Setup Paths ===
    model_name = "tabddpm"
    data_path = os.path.join(real_data_dir, dataset_name)
    save_path = os.path.join(synthetic_dir, dataset_name, model_name)
    os.makedirs(save_path, exist_ok=True)

    x_train_path = os.path.join(data_path, "x_train.csv")
    y_train_path = os.path.join(data_path, "y_train.csv")

    # === Set Seed for Reproducibility ===
    np.random.seed(seed)
    random.seed(seed)
    zero.improve_reproducibility(seed)

    # === Load and Format Data ===
    X = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path).values.ravel()
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    print(f"Loaded dataset '{dataset_name}': X={X.shape}, y={y.shape}")

    # === Save formatted training set in temp folder for make_dataset ===
    temp_dir = os.path.join(synthetic_dir, f"_temp_{dataset_name}_{model_name}")
    os.makedirs(temp_dir, exist_ok=True)

    # Save in .npy format (required by TabDDPM's make_dataset)
    np.save(os.path.join(temp_dir, "X_cat_train.npy"), X.to_numpy())
    np.save(os.path.join(temp_dir, "y_train.npy"), y)


    # === TabDDPM Config (Paper Defaults) ===
    T_dict = {
        'normalization': None,
        'cat_encoding': 'pre_encoded',
        'seed': seed,
    }

    model_params = {
        'num_classes': len(np.unique(y)),
        'is_y_cond': True,
        'rtdl_params': {
            'd_layers': [256, 256, 256, 256],
            'dropout': 0.0
        }
    }

    # === Prepare Dataset & Model ===
    T = Transformations(**T_dict)
    dataset = make_dataset(temp_dir, T, num_classes=model_params['num_classes'], is_y_cond=model_params['is_y_cond'], change_val=False, dataset_name=dataset_name)


    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features

    model_params['d_in'] = int(d_in)

    steps = 10000
    lr = 0.001
    weight_decay = 1e-05
    batch_size = 256
    num_timesteps = 1000
    gaussian_loss_type = 'mse'
    scheduler = 'cosine'
    model_type = 'mlp'

    model = get_model(model_type, model_params, num_numerical_features, category_sizes=dataset.get_category_sizes('train'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    
    train_loader = prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    # === Training Loop ===
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)

    for step in range(steps):
        x, out_dict = next(train_loader)
        x = x.to(device)
        out_dict = {'y': out_dict.long().to(device)}
        optimizer.zero_grad()
        loss_multi, loss_gauss = diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        optimizer.step()
        update_ema(ema_model.parameters(), model.parameters())
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{steps} | MLoss: {loss_multi.item():.4f} | GLoss: {loss_gauss.item():.4f}")

    # === Sampling from Trained EMA Model ===
    diffusion._denoise_fn.load_state_dict(ema_model.state_dict())
    diffusion.eval()
    class_dist = torch.bincount(torch.tensor(y)) / len(y)
    x_synth, y_synth = diffusion.sample_all(num_samples=len(y), batch_size=batch_size, y_dist=class_dist.to(device))

    # === Save Results ===
    pd.DataFrame(x_synth.cpu().numpy()).to_csv(os.path.join(save_path, "x_synth.csv"), index=False)
    pd.DataFrame(y_synth.cpu().numpy(), columns=["label"]).to_csv(os.path.join(save_path, "y_synth.csv"), index=False)

    print(f"\nSynthetic data saved to: {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TabDDPM and generate synthetic data")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., adult)')
    parser.add_argument('--real_data_dir', type=str, default='Data', help='Directory containing x_train.csv and y_train.csv')
    parser.add_argument('--synthetic_dir', type=str, default='Synthetic', help='Directory to save synthetic output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    run_tabddpm(
        dataset_name=args.dataset,
        real_data_dir=args.real_data_dir,
        synthetic_dir=args.synthetic_dir,
        seed=args.seed
    )
