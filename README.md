# Tabular Data Generation Benchmark

This repository contains code for benchmarking various tabular data generation models.

## Dataset Structure

The repository organizes datasets by size category:
- Small datasets: `datasets_DM_small/`
- Medium datasets: `datasets_DM_medium/`
- Large datasets: `datasets_DM_big/`

## Available Models

The following models are implemented:
- GANBLR - Generative Adversarial Network with Bayesian Label Representation
- GANBLR++ - Enhanced GANBLR with Reinforcement Learning capabilities
- CTABGAN+ - Conditional Tabular GAN Plus
- TabDDPM - Tabular Denoising Diffusion Probabilistic Model
- TabSyn - Tabular Data Synthesizer
- GREAT - Generative Relational Autoencoder for Tabular data
- RLIG - Representation Learning with Information Gain (KDB-based)

## Setup

1. Create the conda environments (this only needs to be done once):

```bash
# For TensorFlow-based models
conda env create -f envs/env_tf.yml

# For PyTorch-based models
conda env create -f envs/env_torch.yml
```

## Using the Benchmark Runner

This repository includes a `main.py` script that automates the entire workflow. It will:
1. Convert ARFF files to CSV
2. Preprocess the data
3. Split datasets into train/test
4. Train selected models
5. Run evaluation

### Examples:

```bash
# Create the conda environments (only needs to be done once)
python main.py --create_envs

# Run a single model on a single dataset (for testing)
python main.py --single_run --dataset car --model ganblr --size small

# Run specific models on specific datasets
python main.py --datasets adult magic --models ganblr tabddpm

# Run all models on all datasets
python main.py

# Run with GPU selection
python main.py --gpu 1

# Run only the new RLIG model
python main.py --single_run --dataset adult --model rlig --size medium
```

## Manual Workflow

If you prefer to run the steps manually, follow these instructions:

### 1. Convert ARFF to CSV

```python
# Include this in your script
import re
import pandas as pd
from scipy.io import arff

# Load ARFF File
data, meta = arff.loadarff('adult.arff')
df = pd.DataFrame(data)

# Decode Byte Strings
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Clean: remove \, ", and ' from strings
df = df.applymap(lambda x: re.sub(r'[\\\'\"]', '', x) if isinstance(x, str) else x)

# Function for encoding pre-discretized interval bins
def extract_lower_bound(interval):
    match = re.match(r"\(?(-?[\d\.inf]+)-", interval)
    if match:
        val = match.group(1)
        return float('-inf') if val == '-inf' else float(val)
    return float('inf')  # fallback if format doesn't match

def encode_bins_numerically(series):
    unique_bins = series.dropna().unique()
    sorted_bins = sorted(unique_bins, key=extract_lower_bound)
    bin_to_id = {bin_val: i for i, bin_val in enumerate(sorted_bins)}
    return series.map(bin_to_id), bin_to_id

# Identify and encode binned columns
binned_cols = [ ]  
for col in binned_cols:
    df[col], mapping = encode_bins_numerically(df[col])
    print(f"\nMapping for {col}:")
    for k, v in mapping.items():
        print(f"{k} → {v}")

# Save final CSV
df.to_csv("adult.csv", index=False)
```

### 2. Preprocessing

```bash
# Encode discrete values
python Scripts/preprocess_encode_only.py --input Raw/adult.csv --output Discrete/adult_discrete.csv

# Split dataset
python Scripts/split_dataset.py --input_csv Discrete/adult_discrete.csv --output_dir Data/adult --seed 42
```

### 3. Training Models

#### GANBLR
```bash
# Activate TF environment
conda activate tabgen-tf

# Train model
python Scripts/ganblr_train.py --dataset adult --size_category medium

# Evaluate
python Scripts/tstr_evaluation.py --synthetic_dir Synthetic/adult/ganblr --real_test_dir Data/adult
```

#### GANBLR++
```bash
# Activate TF environment
conda activate tabgen-tf

# Train model
python Scripts/ganblrplus_train.py --dataset adult --size_category medium --k 2 --episodes 5

# Evaluate
python Scripts/tstr_evaluation.py --synthetic_dir Synthetic/adult/ganblrplus --real_test_dir Data/adult
```

#### CTABGAN+
```bash
# Activate PyTorch environment
conda activate tabgen-torch

# Train model
python Scripts/ctabganplus_train.py --dataset_name adult --size_category medium

# Evaluate
python Scripts/tstr_evaluation.py --synthetic_dir Synthetic/adult/ctabgan_plus --real_test_dir Data/adult
```

#### TabDDPM
```bash
# Activate PyTorch environment
conda activate tabgen-torch

# Train model
python Scripts/tabddpm_train.py --dataset adult

# Evaluate
python Scripts/tstr_evaluation.py --synthetic_dir Synthetic/adult/tabddpm --real_test_dir Data/adult
```

#### TabSyn
```bash
# Activate PyTorch environment
conda activate tabgen-torch

# Create NPY files
python Scripts/create_npy.py --dataset adult

# Train VAE
python tabsyn/vae/main.py --dataname adult --gpu 0

# Train diffusion model
python Scripts/tabsyn_train.py --dataset adult

# Evaluate
python Scripts/tstr_evaluation.py --synthetic_dir Synthetic/adult/tabsyn --real_test_dir Data/adult
```

#### GREAT
```bash
# Activate PyTorch environment
conda activate tabgen-torch

# Train model
python Scripts/great_train.py --dataset adult

# Evaluate
python Scripts/tstr_evaluation.py --synthetic_dir Synthetic/adult/great --real_test_dir Data/adult
```

#### RLIG (KDB-based)
```bash
# Activate TF environment
conda activate tabgen-tf

# Train model
python Scripts/rlig_train.py --dataset adult --size_category medium

# Evaluate
python Scripts/tstr_evaluation.py --synthetic_dir Synthetic/adult/rlig --real_test_dir Data/adult
```