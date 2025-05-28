#!/usr/bin/env python
"""
Tabular Data Generation Benchmark Runner

This script automates the entire process of benchmarking tabular data generation models:
1. Converting ARFF files to CSV
2. Preprocessing the data
3. Splitting datasets into train/test
4. Training and evaluating multiple tabular generation models
5. Running train/synthetic/test (TSTR) evaluation

Usage examples:
    # Run all models on all datasets
    python main.py

    # Run specific models on specific datasets
    python main.py --datasets adult magic --models ganblr tabddpm

    # Run a single model on a single dataset (for testing)
    python main.py --single_run --dataset adult --model ganblr --size medium --gpu 0

    # Run on all datasets with GPU selection
    python main.py --gpu 1
"""

import os
import argparse
import subprocess
import re
import pandas as pd
import numpy as np
from scipy.io import arff
import glob


def arff_to_csv(arff_file, output_csv):
    """Convert ARFF file to CSV with proper encoding of discrete values"""
    print(f"Converting {arff_file} to CSV...")

    # Load ARFF File
    data, meta = arff.loadarff(arff_file)
    df = pd.DataFrame(data)

    # Decode Byte Strings
    df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Clean: remove \, ", and ' from strings
    df = df.applymap(lambda x: re.sub(r'[\\\'\"]', '', x) if isinstance(x, str) else x)

    # Define function for encoding pre-discretized interval bins
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
    # This is a heuristic approach - looking for columns with interval patterns
    binned_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_vals = df[col].dropna().sample(min(5, df[col].nunique())).astype(str)
            if any(re.match(r'.*\(.*-.*\].*', str(v)) for v in sample_vals):
                binned_cols.append(col)

    # Apply encoding to binned columns
    for col in binned_cols:
        df[col], mapping = encode_bins_numerically(df[col])
        print(f"Mapped {col} from intervals to discrete values")

    # Save final CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned dataset to '{output_csv}'")
    return output_csv


def run_command(cmd):
    """Run a shell command and print output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    return result.returncode


def process_dataset(dataset_name, dataset_path, size_category, models, base_dir, gpu_id=0, seeds=None):
    """Process a single dataset through the entire pipeline"""
    print(f"\n{'=' * 80}\nProcessing dataset: {dataset_name} ({size_category})\n{'=' * 80}")

    # Default seeds if not provided
    if seeds is None:
        seeds = [42, 456, 1710]

    # Create directory structure with absolute paths
    raw_dir = os.path.join(base_dir, "Raw")
    discrete_dir = os.path.join(base_dir, "Discrete")
    data_dir = os.path.join(base_dir, f"Data/{dataset_name}")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(discrete_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Convert ARFF to CSV
    csv_path = os.path.join(raw_dir, f"{dataset_name}.csv")
    arff_to_csv(dataset_path, csv_path)

    # Step 2: Preprocess to discrete values (using system Python)
    discrete_csv = os.path.join(discrete_dir, f"{dataset_name}_discrete.csv")
    script_path = os.path.join(base_dir, "Scripts/preprocess_encode_only.py")
    run_command(f"python {script_path} --input {csv_path} --output {discrete_csv}")

    # Step 3: Split dataset into train/test for each seed
    # Split the dataset for each seed to ensure fair comparison
    print(f"Creating {len(seeds)} different train/test splits for seeds: {seeds}")
    for seed in seeds:
        seed_dir = os.path.join(data_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        script_path = os.path.join(base_dir, "Scripts/split_dataset.py")
        run_command(f"python {script_path} --input_csv {discrete_csv} --output_dir {seed_dir} --seed {seed}")
        print(f"Created train/test split with seed {seed} in {seed_dir}")

    # Step 4: Train models and evaluate with multiple seeds
    for model in models:
        print(f"\n{'-' * 40}\nTraining {model} on {dataset_name} with {len(seeds)} different seeds\n{'-' * 40}")

        # Create a directory for this model
        model_dir = os.path.join(base_dir, f"Synthetic/{dataset_name}/{model}")
        os.makedirs(model_dir, exist_ok=True)

        # Run the model with each seed
        for seed in seeds:
            print(f"\nRunning {model} with seed {seed}")
            # Use the seed-specific data directory for this run
            seed_data_dir = os.path.join(data_dir, f"seed{seed}")
            synthetic_dir = os.path.join(model_dir, f"seed{seed}")
            os.makedirs(synthetic_dir, exist_ok=True)

            # Note: We're using system Python for everything until conda environments are properly set up
            if model == "ganblr":
                # Originally used TF environment
                script_path = os.path.join(base_dir, "Scripts/ganblr_train.py")
                run_command(
                    f"python {script_path} --dataset {dataset_name} --size_category {size_category} --seed {seed} --data_dir {seed_data_dir}")

                eval_script = os.path.join(base_dir, "Scripts/tstr_evaluation.py")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir}")

            elif model == "ganblrplus":
                # GANBLR++ model
                script_path = os.path.join(base_dir, "Scripts/ganblrplus_train.py")
                run_command(
                    f"python {script_path} --dataset {dataset_name} --size_category {size_category} --seed {seed} --data_dir {seed_data_dir}")

                eval_script = os.path.join(base_dir, "Scripts/tstr_evaluation.py")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir}")

            elif model == "ctgan":
                # CTGAN implementation (original CTGAN)
                script_path = os.path.join(base_dir, "Scripts/ctgan_train.py")
                run_command(
                    f"python {script_path} --dataset_name {dataset_name} --data_dir {seed_data_dir} --size_category {size_category} --gpu_id {gpu_id} --seed {seed}")

                eval_script = os.path.join(base_dir, "Scripts/tstr_evaluation.py")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir}")

            elif model == "ctabgan":
                # Originally used PyTorch environment
                script_path = os.path.join(base_dir, "Scripts/ctabgan_train.py")
                run_command(
                    f"python {script_path} --dataset_name {dataset_name} --data_dir {seed_data_dir} --size_category {size_category} --gpu_id {gpu_id} --seed {seed}")

                eval_script = os.path.join(base_dir, "Scripts/tstr_evaluation.py")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir}")

            elif model == "ctabgan_plus":
                # Originally used PyTorch environment
                script_path = os.path.join(base_dir, "Scripts/simple_ctabganplus_train.py")
                run_command(
                    f"python {script_path} --dataset {dataset_name} --size_category {size_category} --gpu {gpu_id} --seed {seed} --data_dir {seed_data_dir}")

                eval_script = os.path.join(base_dir, "Scripts/tstr_evaluation.py")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir}")

            elif model == "tabddpm":
                # Originally used PyTorch environment
                script_path = os.path.join(base_dir, "Scripts/tabddpm_train.py")
                run_command(f"python {script_path} --dataset {dataset_name} --seed {seed} --data_dir {seed_data_dir}")

                eval_script = os.path.join(base_dir, "Scripts/tstr_evaluation.py")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir}")

            elif model == "tabsyn":
                # Originally used PyTorch environment
                script_path = os.path.join(base_dir, "Scripts/create_npy.py")
                run_command(f"python {script_path} --dataset {dataset_name} --data_dir {seed_data_dir}")

                vae_script = os.path.join(base_dir, "tabsyn/vae/main.py")
                run_command(f"python {vae_script} --dataname {dataset_name} --gpu {gpu_id} --seed {seed} --data_dir {seed_data_dir}")

                train_script = os.path.join(base_dir, "Scripts/tabsyn_train.py")
                run_command(f"python {train_script} --dataset {dataset_name} --seed {seed} --data_dir {seed_data_dir}")

                eval_script = os.path.join(base_dir, "Scripts/tstr_evaluation.py")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir}")

            elif model == "great":
                # Originally used PyTorch environment
                script_path = os.path.join(base_dir, "Scripts/great_train.py")
                run_command(f"python {script_path} --dataset {dataset_name} --seed {seed} --data_dir {seed_data_dir}")

                eval_script = os.path.join(base_dir, "Scripts/tstr_evaluation.py")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir}")

            elif model == "rlig":
                # RLIG is based on KDB module in ganblr
                script_path = os.path.join(base_dir, "Scripts/rlig_train.py")
                run_command(
                    f"python {script_path} --dataset {dataset_name} --size_category {size_category} --seed {seed} --data_dir {seed_data_dir}")

                eval_script = os.path.join(base_dir, "Scripts/tstr_evaluation.py")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir}")

            print(f"Completed {model} for {dataset_name} with seed {seed}")

        # After running with all seeds, compute the average of the results
        if len(seeds) > 1:
            print(f"\nComputing average results for {model} across {len(seeds)} seeds")
            avg_results_dir = os.path.join(model_dir, "avg_results")
            os.makedirs(avg_results_dir, exist_ok=True)

            # Create a script to average the results across seeds
            avg_script = os.path.join(base_dir, "Scripts/average_results.py")

            # If the script doesn't exist, create it
            if not os.path.exists(avg_script):
                with open(avg_script, "w") as f:
                    f.write("""#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Average results across multiple seeds")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing seed results')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for average results')
    args = parser.parse_args()

    # Find all seed directories
    seed_dirs = glob.glob(os.path.join(args.model_dir, "seed*"))

    if not seed_dirs:
        print(f"No seed directories found in {args.model_dir}")
        return

    print(f"Found {len(seed_dirs)} seed directories: {seed_dirs}")

    # Find all CSV files in the first seed directory
    first_seed_dir = seed_dirs[0]
    csv_files = glob.glob(os.path.join(first_seed_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {first_seed_dir}")
        return

    # Process each CSV file
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        print(f"Processing {file_name}")

        # Collect dataframes from all seeds
        all_dfs = []
        for seed_dir in seed_dirs:
            seed_file = os.path.join(seed_dir, file_name)
            if os.path.exists(seed_file):
                try:
                    df = pd.read_csv(seed_file)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {seed_file}: {e}")

        # If we have dataframes, average them
        if all_dfs:
            # Stack all dataframes
            combined_df = pd.concat(all_dfs)

            # Group by index and average
            avg_df = combined_df.groupby(combined_df.index).mean()

            # Save the result
            output_file = os.path.join(args.output_dir, file_name)
            avg_df.to_csv(output_file, index=False)
            print(f"Saved average results to {output_file}")
        else:
            print(f"No valid dataframes found for {file_name}")

if __name__ == "__main__":
    main()
""")

            # Run the averaging script
            run_command(f"python {avg_script} --model_dir {model_dir} --output_dir {avg_results_dir}")

            # Use the first seed's data directory for evaluation on the averaged results
            first_seed_data_dir = os.path.join(data_dir, f"seed{seeds[0]}")
            eval_script = os.path.join(base_dir, "Scripts/tstr_evaluation.py")
            run_command(f"python {eval_script} --synthetic_dir {avg_results_dir} --real_test_dir {first_seed_data_dir}")

            print(f"Completed averaging results for {model} on {dataset_name}")

        print(f"Completed all runs for {model} on {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Tabular Data Generation Benchmark Runner")
    parser.add_argument('--datasets', nargs='+', help='List of specific datasets to run (default: all)')
    parser.add_argument('--models', nargs='+',
                        choices=['ganblr', 'ganblrplus', 'ctabgan', 'ctgan', 'ctabgan_plus', 'tabddpm', 'tabsyn',
                                 'great', 'rlig', 'all'],
                        default=['all'], help='Models to run')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--single_run', action='store_true',
                        help='Run just one model on one dataset (useful for testing)')
    parser.add_argument('--dataset', type=str, help='Dataset name for single run')
    parser.add_argument('--model', type=str,
                        choices=['ganblr', 'ganblrplus', 'ctabgan', 'ctgan', 'ctabgan_plus', 'tabddpm', 'tabsyn',
                                 'great', 'rlig'],
                        help='Model name for single run')
    parser.add_argument('--size', type=str, choices=['small', 'medium', 'large'], default='medium',
                        help='Size category for single run')
    parser.add_argument('--create_envs', action='store_true', help='Create conda environments (only needed once)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44],
                        help='List of random seeds to use (default: 42, 43, 44)')
    args = parser.parse_args()

    # Handle single run case
    if args.single_run:
        if not args.dataset or not args.model:
            print("Error: --dataset and --model must be specified with --single_run")
            return

        # For single run, we'll create a streamlined version
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Map dataset to its path
        dataset_paths = {
            # Small datasets
            'car': os.path.join(base_dir, 'datasets_DM_small', 'car.arff'),
            'satellite': os.path.join(base_dir, 'datasets_DM_small', 'satellite.arff'),
            'segment': os.path.join(base_dir, 'datasets_DM_small', 'segment.arff'),
            'sick': os.path.join(base_dir, 'datasets_DM_small', 'sick.arff'),
            'sign': os.path.join(base_dir, 'datasets_DM_small', 'sign.arff'),
            'hungarian': os.path.join(base_dir, 'datasets_DM_small', 'hungarian.arff'),
            # Medium datasets
            'adult': os.path.join(base_dir, 'datasets_DM_medium', 'adult.arff'),
            'magic': os.path.join(base_dir, 'datasets_DM_medium', 'magic.arff'),
            'shuttle': os.path.join(base_dir, 'datasets_DM_medium', 'shuttle.arff'),
            'nursery': os.path.join(base_dir, 'datasets_DM_medium', 'nursery.arff'),
            'chess': os.path.join(base_dir, 'datasets_DM_medium', 'chess.arff'),
            # Large datasets
            'census-income': os.path.join(base_dir, 'datasets_DM_big', 'census-income.arff'),
            'covtype-mod': os.path.join(base_dir, 'datasets_DM_big', 'covtype-mod.arff'),
            'localization': os.path.join(base_dir, 'datasets_DM_big', 'localization.arff'),
            'poker-hand': os.path.join(base_dir, 'datasets_DM_big', 'poker-hand.arff'),
        }

        if args.dataset not in dataset_paths:
            print(f"Error: Unknown dataset '{args.dataset}'")
            return

        # Process just this single dataset with the selected model
        process_dataset(args.dataset, dataset_paths[args.dataset], args.size, [args.model], base_dir, args.gpu,
                        args.seeds)
        return

    # Determine which models to run
    models_to_run = []
    if 'all' in args.models:
        models_to_run = ['ganblr', 'ganblrplus', 'ctabgan', 'ctgan', 'ctabgan_plus', 'tabddpm', 'tabsyn', 'great',
                         'rlig']
    else:
        models_to_run = args.models

    print(f"Will run the following models: {', '.join(models_to_run)}")

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up dataset groups with absolute paths
    dataset_groups = {
        'small': [
            ('car', os.path.join(base_dir, 'datasets_DM_small', 'car.arff')),
            ('satellite', os.path.join(base_dir, 'datasets_DM_small', 'satellite.arff')),
            ('segment', os.path.join(base_dir, 'datasets_DM_small', 'segment.arff')),
            ('sick', os.path.join(base_dir, 'datasets_DM_small', 'sick.arff')),
            ('sign', os.path.join(base_dir, 'datasets_DM_small', 'sign.arff')),
            ('hungarian', os.path.join(base_dir, 'datasets_DM_small', 'hungarian.arff')),
        ],
        'medium': [
            ('adult', os.path.join(base_dir, 'datasets_DM_medium', 'adult.arff')),
            ('magic', os.path.join(base_dir, 'datasets_DM_medium', 'magic.arff')),
            ('shuttle', os.path.join(base_dir, 'datasets_DM_medium', 'shuttle.arff')),
            ('nursery', os.path.join(base_dir, 'datasets_DM_medium', 'nursery.arff')),
            ('chess', os.path.join(base_dir, 'datasets_DM_medium', 'chess.arff')),
        ],
        'large': [
            ('census-income', os.path.join(base_dir, 'datasets_DM_big', 'census-income.arff')),
            ('covtype-mod', os.path.join(base_dir, 'datasets_DM_big', 'covtype-mod.arff')),
            ('localization', os.path.join(base_dir, 'datasets_DM_big', 'localization.arff')),
            ('poker-hand', os.path.join(base_dir, 'datasets_DM_big', 'poker-hand.arff')),
        ]
    }

    # Filter datasets if specified
    if args.datasets:
        filtered_groups = {}
        for size, datasets in dataset_groups.items():
            filtered_datasets = [d for d in datasets if d[0] in args.datasets]
            if filtered_datasets:
                filtered_groups[size] = filtered_datasets
        dataset_groups = filtered_groups

    # Create environments if requested
    if args.create_envs:
        print("Creating conda environments (this may take a while)...")
        tf_env_path = os.path.join(base_dir, "envs/env_tf.yml")
        torch_env_path = os.path.join(base_dir, "envs/env_torch.yml")
        run_command(f"conda env list | grep -q tabgen-tf || conda env create -f {tf_env_path}")
        run_command(f"conda env list | grep -q tabgen-torch || conda env create -f {torch_env_path}")

    # Process each dataset
    for size_category, datasets in dataset_groups.items():
        for dataset_name, dataset_path in datasets:
            process_dataset(dataset_name, dataset_path, size_category, models_to_run, base_dir, args.gpu, args.seeds)

    print("\nAll datasets and models have been processed.")


if __name__ == "__main__":
    main()