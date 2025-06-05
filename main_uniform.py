#!/usr/bin/env python
"""
Tabular Data Generation Benchmark Runner with Uniform Preprocessing

This script applies a standardized preprocessing approach across all models:
1. Handles missing values consistently
2. Optionally discretizes numerical features using uniform binning 
3. Applies the same train/test split strategy for all models
4. Processes datasets in a uniform way for fair comparison

Usage examples:
    # Run all models on all datasets with uniform preprocessing
    python main_uniform.py

    # Run specific models on specific datasets
    python main_uniform.py --datasets adult magic --models ganblr tabddpm

    # Run a single model on a single dataset (for testing)
    python main_uniform.py --single_run --dataset adult --model ganblr --size medium --gpu 0

    # Run on all datasets with GPU selection
    python main_uniform.py --gpu 1
    
    # Run with or without discretization
    python main_uniform.py --discretize True
"""

import os
import argparse
import subprocess
import re
import pandas as pd
import numpy as np
from scipy.io import arff
import glob
import json
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split, KFold

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from ucimlrepo import fetch_ucirepo
    UCI_AVAILABLE = True
except ImportError:
    UCI_AVAILABLE = False
    print("ucimlrepo package not found. UCI datasets will not be available.")
    
# Import RLiG model if available
try:
    from ganblr.model.rlig import RLiG
    RLIG_AVAILABLE = True
except ImportError:
    RLIG_AVAILABLE = False
    print("RLiG model not found. RLiG functionality will be disabled.")


def train_rlig(X_train, y_train, episodes=2, epochs=5):
    """Train a RLiG model"""
    if not RLIG_AVAILABLE:
        return None

    try:
        # Initialize and train RLiG model
        rlig_model = RLiG()

        # Ensure the data is properly formatted
        if isinstance(y_train, pd.DataFrame):
            y_series = y_train.iloc[:, 0] if y_train.shape[1] == 1 else y_train
        else:
            y_series = y_train

        print(f"Training RLiG with {episodes} episodes and {epochs} epochs")
        rlig_model.fit(X_train, y_series, episodes=2, gan=1, k=1, epochs=20, n=1)
        return rlig_model
    except Exception as e:
        print(f"Error training RLiG model: {e}")
        return None
        
def sample_rlig(rlig_model, X_train, y_train, n_samples):
    """Generate synthetic samples using a trained RLiG model"""
    if rlig_model is None:
        print("Cannot generate samples: RLiG model is None")
        return None
    
    try:
        # Generate synthetic data
        rlig_synthetic = rlig_model.sample(n_samples)
        
        # Convert to DataFrame if it's a numpy array
        if isinstance(rlig_synthetic, np.ndarray):
            # Create column names based on training data
            if isinstance(X_train, pd.DataFrame):
                feature_columns = list(X_train.columns)
            else:
                feature_columns = [f"feature_{i}" for i in range(X_train.shape[1])]
                
            # Add target column name
            if isinstance(y_train, pd.DataFrame):
                target_column = y_train.columns[0]
            elif hasattr(y_train, 'name') and y_train.name is not None:
                target_column = y_train.name
            else:
                target_column = 'target'
                
            columns = feature_columns + [target_column]
            rlig_synthetic = pd.DataFrame(rlig_synthetic, columns=columns)
        
        return rlig_synthetic
    except Exception as e:
        print(f"Error generating samples with RLiG model: {e}")
        return None

def label_encode_cols(df, cols):
    """
    Label encodes specified columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing columns to encode
    cols : list
        List of column names to encode
        
    Returns:
    --------
    DataFrame, dict
        Transformed DataFrame and dictionary of encoders
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    return df_encoded, encoders


def read_arff_file(file_path):
    """
    Read an ARFF file and return as DataFrame with metadata.
    
    Parameters:
    -----------
    file_path : str
        Path to the ARFF file
        
    Returns:
    --------
    DataFrame, dict
        Parsed data and metadata
    """
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Decode any byte strings (common in ARFF files)
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    return df, meta


def preprocess_data(X, y, name, discretize=True, numeric_cols=True, model_name=None, cv_fold=None, n_folds=None):
    """
    Preprocess data: optionally discretize continuous variables and encode categoricals
    
    This version can selectively apply discretization using quantile binning with 7 bins,
    which better preserves the distribution of the original data. This is especially useful
    for certain models, while others may perform better with non-discretized data.
    
    Parameters:
    -----------
    X : DataFrame
        Features to preprocess
    y : DataFrame or Series
        Target variable
    name: Name of dataset
    discretize : bool, default=True
        Whether to apply discretization to continuous features
    model_name : str, optional
        Name of the model being trained, used for model-specific preprocessing decisions
    cv_fold : int, optional
        Current fold number when doing k-fold cross-validation (0-indexed)
    n_folds : int, optional
        Total number of folds when doing k-fold cross-validation
    """
    # First, handle missing values
    # Check if there are any missing values
    if X.isnull().any().any():
        print("Handling missing values in the dataset...")

        # For categorical columns, fill with the most frequent value
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna(X[col].mode()[0])

        # For numeric columns, fill with the median
        for col in X.select_dtypes(include=['number']).columns:
            X[col] = X[col].fillna(X[col].median())

        print("Missing values have been imputed")

    # Identify column types after imputation
    continuous_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Store the original column counts for reference
    continuous_cols_count = len(continuous_cols)
    categorical_cols_count = len(categorical_cols)
    
    print("Continuous columns: ", continuous_cols)
    print("Categorical columns: ", categorical_cols)
    print(f"Number of continuous columns: {continuous_cols_count}")
    print(f"Number of categorical columns: {categorical_cols_count}")

    # Apply discretization based on the flag
    apply_discretization = discretize

    # Log the discretization status for the current model
    if model_name:
        if apply_discretization:
            print(f"Note: Using discretized features for {model_name}")
        else:
            print(f"Note: Using non-discretized features for {model_name}")

    # Create transformation pipeline with optional discretization
    transformers = []
    if len(continuous_cols) > 0:
        if apply_discretization:
            # Add discretization step to the pipeline
            continuous_transformer = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('discretizer', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'))
            ])
            print("Using discretization with uniform binning (5 bins)")
        else:
            # Only standardize without discretization
            continuous_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            print("Using standardization without discretization")

        transformers.append(('num', continuous_transformer, continuous_cols))

    # Handle categorical columns
    if len(categorical_cols) > 0:
        X, encoders = label_encode_cols(X, categorical_cols)

    # Apply transformations
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    X_transformed = preprocessor.fit_transform(X)
    
    # Create a DataFrame with numeric column names to avoid feature name mismatch issues
    # This ensures compatibility with models that expect numeric column indices
    X_transformed_df = pd.DataFrame(X_transformed, columns=[str(i) for i in range(X_transformed.shape[1])])

    # Clean the target values for 'adult' dataset
    if name == 'adult':
        y = y.transform(lambda col: col.astype(str).str.replace('.', '', regex=False))

    # Handle target variable
    if y.isnull().any().any():
        print("Handling missing values in target variable...")
        if y.dtypes[0] == 'object':
            y = y.fillna(y.mode()[0])
        else:
            y = y.fillna(y.median())

    if y.dtypes[0] == 'object':
        label_encoder = LabelEncoder()
        y_transformed = pd.DataFrame(label_encoder.fit_transform(y.values.ravel()), columns=y.columns)
    else:
        y_transformed = y

    # Split data based on whether we're using cross-validation or traditional train-test split
    if cv_fold is not None and n_folds is not None:
        from sklearn.model_selection import KFold
        print(f"Using {n_folds}-fold cross-validation (fold {cv_fold + 1}/{n_folds})")

        # Create fold indices
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Convert to arrays for indexing
        X_array = X_transformed_df.values
        y_array = y_transformed.values

        # Get the train/test indices for this fold
        train_indices = []
        test_indices = []

        for i, (train_idx, test_idx) in enumerate(kf.split(X_array)):
            if i == cv_fold:
                train_indices = train_idx
                test_indices = test_idx
                break

        # Split the data using the indices
        X_train = pd.DataFrame(X_array[train_indices], columns=[str(i) for i in range(X_array.shape[1])])
        X_test = pd.DataFrame(X_array[test_indices], columns=[str(i) for i in range(X_array.shape[1])])
        y_train = pd.DataFrame(y_array[train_indices], columns=y_transformed.columns)
        y_test = pd.DataFrame(y_array[test_indices], columns=y_transformed.columns)

        return X_train, X_test, y_train, y_test
    else:
        # Traditional split
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed_df, y_transformed, test_size=0.2, random_state=42, stratify=y_transformed
        )
        
        # Ensure numeric column names for compatibility
        if not all(col.isdigit() for col in X_train.columns):
            X_train.columns = [str(i) for i in range(X_train.shape[1])]
            X_test.columns = [str(i) for i in range(X_test.shape[1])]
            
        return X_train, X_test, y_train, y_test


def load_dataset(name, dataset_info):
    """Load dataset from UCI repository or local file"""
    if isinstance(dataset_info, int) and UCI_AVAILABLE:
        try:
            data = fetch_ucirepo(id=dataset_info)
            X = data.data.features
            # Change the name of columns to avoid "-" to parsing error
            X.columns = [col.replace('-', '_') for col in X.columns]
            y = data.data.targets
            # Change the name of y dataframe to avoid duplicate "class" keyword
            y.columns = ["target"]

            # Special handling for Credit dataset which is known to have NaN values
            if name == "Credit":
                print(f"Special handling for {name} dataset")
                # Check for missing values
                missing_X = X.isnull().sum().sum()
                missing_y = y.isnull().sum().sum()
                print(f"Missing values detected: {missing_X} in features, {missing_y} in target")

                # Drop rows with NaN values if there are not too many
                if missing_X + missing_y > 0 and missing_X + missing_y < len(X) * 0.1:  # If less than 10% are missing
                    print(f"Dropping {missing_X + missing_y} rows with missing values")
                    # Combine X and y for dropping rows with any NaN
                    combined = pd.concat([X, y], axis=1)
                    combined_clean = combined.dropna()

                    # Split back to X and y
                    X = combined_clean.iloc[:, :-1]
                    y = combined_clean.iloc[:, -1:].copy()
                    y.columns = ["target"]

                    print(f"After dropping rows: X shape = {X.shape}, y shape = {y.shape}")

            return X, y
        except Exception as e:
            print(f"Error loading UCI dataset {name} (id={dataset_info}): {e}")
            return None, None
    elif isinstance(dataset_info, str):
        try:
            if dataset_info.endswith(".csv"):
                df = pd.read_csv(dataset_info)

                # Check for NaN values
                if df.isnull().any().any():
                    print(f"Dataset {name} has missing values. Handling...")
                    # For Credit Card dataset specifically
                    if "Credit" in name or "credit" in name or "UCI_Credit_Card.csv" in dataset_info:
                        print(f"Special handling for Credit dataset")
                        # Drop rows with NaN if there are not too many
                        missing_count = df.isnull().sum().sum()
                        if missing_count < len(df) * 0.1:  # If less than 10% are missing
                            print(f"Dropping {missing_count} rows with missing values")
                            df = df.dropna()
                        # Otherwise, we'll use imputation in the preprocess_data function

                X = df.iloc[:, :-1]
                # Change the name of columns to avoid "-" to parsing error
                X.columns = [col.replace('-', '_') for col in X.columns]

                if name == "letter_recog":
                    # or y = df.iloc[:, [0]]
                    y = df[['lettr']]
                else:
                    y = df.iloc[:, -1:]
                # Change the name of y dataframe to avoid duplicate "class" keyword
                y.columns = ["target"]
                return X, y
            else:
                # Read arff file
                df, meta = read_arff_file(dataset_info)

                # Check for NaN values
                if df.isnull().any().any():
                    print(f"ARFF file {name} has missing values. Handling...")
                    # Drop rows with NaN if there are not too many
                    missing_count = df.isnull().sum().sum()
                    if missing_count < len(df) * 0.1:  # If less than 10% are missing
                        print(f"Dropping {missing_count} rows with missing values")
                        df = df.dropna()
                    # Otherwise, we'll use imputation in the preprocess_data function

                # Detect target column - typically 'class' or the last column
                if 'class' in df.columns:
                    X = df.drop('class', axis=1)
                    y = df[['class']]
                elif 'xAttack' in df.columns:
                    X = df.drop('xAttack', axis=1)
                    y = df[['xAttack']]
                else:
                    # If no specific target column is identified, use the last column
                    print(f"No target column 'class' or 'xAttack' found. Using last column as target.")
                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1:]
                
                # Change the name of columns to avoid "-" to parsing error
                X.columns = [col.replace('-', '_') for col in X.columns]
                
                # Change the name of y dataframe to avoid duplicate "class" keyword
                y.columns = ["target"]
                return X, y
        except Exception as e:
            print(f"Error loading dataset from file {dataset_info}: {e}")
            return None, None
    else:
        print(f"Invalid dataset specification for {name}")
        return None, None


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

    # Save final CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned dataset to '{output_csv}'")
    return output_csv


def run_command(cmd):
    """Run a shell command and print output"""
    # Normalize path separators for Windows compatibility
    if os.name == 'nt':  # Windows
        cmd = cmd.replace('/', '\\')
    
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    return result.returncode


def process_dataset(dataset_name, dataset_path, size_category, models, base_dir, gpu_id=0, seeds=None, discretize=True):
    """Process a single dataset through the entire pipeline with uniform preprocessing"""
    print(f"\n{'=' * 80}\nProcessing dataset: {dataset_name} ({size_category})\n{'=' * 80}")
    
    # Default seeds if not provided
    if seeds is None:
        seeds = [42, 456, 1710]

    # Create directory structure with absolute paths
    raw_dir = os.path.join(base_dir, "Raw")
    discrete_dir = os.path.join(base_dir, "Discrete")
    data_dir = os.path.join(base_dir, "Data", dataset_name)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(discrete_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Check if this is a UCI dataset (dataset_path will be an integer)
    is_uci_dataset = isinstance(dataset_path, int)
    
    if is_uci_dataset:
        print(f"Loading UCI ML Repository dataset with ID {dataset_path}")
        X, y = load_dataset(dataset_name, dataset_path)
        
        # Save to CSV for future reference
        csv_path = os.path.join(raw_dir, f"{dataset_name}.csv")
        if X is not None and y is not None:
            combined_df = pd.concat([X, y], axis=1)
            combined_df.to_csv(csv_path, index=False)
            print(f"Saved UCI dataset to {csv_path}")
    else:
        # Standard file-based dataset
        # Step 1: Convert ARFF to CSV if needed
        csv_path = os.path.join(raw_dir, f"{dataset_name}.csv")
        
        if dataset_path.endswith('.arff'):
            arff_to_csv(dataset_path, csv_path)
            X, y = load_dataset(dataset_name, dataset_path)
        else:
            X, y = load_dataset(dataset_name, csv_path)

    if X is None or y is None:
        print(f"Error: Failed to load dataset {dataset_name}")
        return

    # Split the dataset for each seed to ensure fair comparison
    print(f"Creating {len(seeds)} different train/test splits for seeds: {seeds}")
    for seed in seeds:
        # Set seed for reproducibility
        np.random.seed(seed)
        
        seed_dir = os.path.join(data_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        # Identify original column types for reference
        continuous_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Apply uniform preprocessing
        print(f"Applying uniform preprocessing for seed {seed}...")
        X_train, X_test, y_train, y_test = preprocess_data(
            X, y, name=dataset_name, discretize=discretize, 
            model_name=None, cv_fold=None, n_folds=None
        )
        
        # Store the count of continuous columns for later use
        original_numeric_cols_count = len(continuous_cols)
        
        # Save train/test split to CSV files
        X_train.to_csv(os.path.join(seed_dir, "x_train.csv"), index=False)
        X_test.to_csv(os.path.join(seed_dir, "x_test.csv"), index=False)
        y_train.to_csv(os.path.join(seed_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(seed_dir, "y_test.csv"), index=False)
        
        # Save combined files for easier loading
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(seed_dir, "train_full.csv"), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(seed_dir, "test_full.csv"), index=False)
        
        # Determine numeric and categorical columns based on the original data types
        # Since we've transformed to numeric indices, we need to track which columns were originally numerical
        original_numeric_cols_count = len(continuous_cols)
        
        # Generate info.json with metadata about the dataset
        # Important: Use integers for indices (not strings) to ensure compatibility
        # Most models expect num_col_idx and cat_col_idx to be integers
        info = {
            "task_type": "multiclass" if len(np.unique(y_train)) > 2 else "binclass",
            "n_classes": int(len(np.unique(y_train))),
            "num_col_idx": [int(i) for i in range(original_numeric_cols_count)],
            "cat_col_idx": [int(i) for i in range(original_numeric_cols_count, X_train.shape[1])]
        }
        
        # Save the info.json file
        with open(os.path.join(seed_dir, "info.json"), 'w') as f:
            json.dump(info, f, indent=2)
        
        # Also save at dataset level for models that don't look in seed directory
        with open(os.path.join(data_dir, "info.json"), 'w') as f:
            json.dump(info, f, indent=2)
            
        print(f"Created train/test split with seed {seed} in {seed_dir}")
        
        # Convert to .npy files for models that require them
        # Since we're using numeric indices in info.json, we need to select columns by index, not by name
        # This avoids issues with column name mismatches
        num_indices = info["num_col_idx"]
        cat_indices = info["cat_col_idx"]
        
        if num_indices:
            X_num_train = X_train.iloc[:, num_indices].values 
            X_num_test = X_test.iloc[:, num_indices].values
        else:
            X_num_train = np.empty((len(X_train), 0))
            X_num_test = np.empty((len(X_test), 0))
            
        if cat_indices:
            X_cat_train = X_train.iloc[:, cat_indices].values
            X_cat_test = X_test.iloc[:, cat_indices].values
        else:
            X_cat_train = np.empty((len(X_train), 0))
            X_cat_test = np.empty((len(X_test), 0))
        
        np.save(os.path.join(seed_dir, "X_num_train.npy"), X_num_train)
        np.save(os.path.join(seed_dir, "X_num_test.npy"), X_num_test)
        np.save(os.path.join(seed_dir, "X_cat_train.npy"), X_cat_train)
        np.save(os.path.join(seed_dir, "X_cat_test.npy"), X_cat_test)
        np.save(os.path.join(seed_dir, "y_train.npy"), y_train.values)
        np.save(os.path.join(seed_dir, "y_test.npy"), y_test.values)
        
        print(f"Created .npy files for seed {seed}")

    # Step 4: Train models and evaluate with multiple seeds
    for model in models:
        print(f"\n{'-' * 40}\nTraining {model} on {dataset_name} with {len(seeds)} different seeds\n{'-' * 40}")

        # Create a directory for this model
        model_dir = os.path.join(base_dir, "Synthetic", dataset_name, model)
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
                script_path = os.path.join(base_dir, "Scripts", "ganblr_train.py")
                run_command(
                    f"python {script_path} --dataset {dataset_name} --size_category {size_category} --seed {seed} --data_dir {seed_data_dir}")

                eval_script = os.path.join(base_dir, "Scripts", "tstr_evaluation.py")
                results_dir = os.path.join(base_dir, "Results", dataset_name)
                os.makedirs(results_dir, exist_ok=True)
                result_file = os.path.join(results_dir, f"{model}_tstr_seed{seed}.csv")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir} --output {result_file}")

            elif model == "ganblrplus":
                # GANBLR++ model
                script_path = os.path.join(base_dir, "Scripts", "ganblrplus_train.py")
                run_command(
                    f"python {script_path} --dataset {dataset_name} --size_category {size_category} --seed {seed} --data_dir {seed_data_dir}")

                eval_script = os.path.join(base_dir, "Scripts", "tstr_evaluation.py")
                results_dir = os.path.join(base_dir, "Results", dataset_name)
                os.makedirs(results_dir, exist_ok=True)
                result_file = os.path.join(results_dir, f"{model}_tstr_seed{seed}.csv")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir} --output {result_file}")

            elif model == "ctgan":
                # CTGAN implementation (original CTGAN)
                script_path = os.path.join(base_dir, "Scripts", "ctgan_train.py")
                
                # Make sure the exact seed directory exists
                if not os.path.exists(seed_data_dir):
                    print(f"ERROR: Seed data directory not found: {seed_data_dir}")
                
                # Build parent directory for dataset
                dataset_parent_dir = os.path.join(base_dir, "Data")
                
                run_command(
                    f"python {script_path} --dataset_name {dataset_name} --real_data_dir {dataset_parent_dir} --data_dir {seed_data_dir} --size_category {size_category} --gpu_id {gpu_id} --seed {seed}")

                eval_script = os.path.join(base_dir, "Scripts", "tstr_evaluation.py")
                results_dir = os.path.join(base_dir, "Results", dataset_name)
                os.makedirs(results_dir, exist_ok=True)
                result_file = os.path.join(results_dir, f"{model}_tstr_seed{seed}.csv")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir} --output {result_file}")

            elif model == "ctabgan":
                # Originally used PyTorch environment
                script_path = os.path.join(base_dir, "Scripts", "ctabgan_train.py")
                
                # Make sure the exact seed directory exists
                if not os.path.exists(seed_data_dir):
                    print(f"ERROR: Seed data directory not found: {seed_data_dir}")
                
                # Build parent directory for dataset
                dataset_parent_dir = os.path.join(base_dir, "Data")
                
                run_command(
                    f"python {script_path} --dataset_name {dataset_name} --real_data_dir {dataset_parent_dir} --data_dir {seed_data_dir} --size_category {size_category} --gpu_id {gpu_id} --seed {seed}")

                eval_script = os.path.join(base_dir, "Scripts", "tstr_evaluation.py")
                results_dir = os.path.join(base_dir, "Results", dataset_name)
                os.makedirs(results_dir, exist_ok=True)
                result_file = os.path.join(results_dir, f"{model}_tstr_seed{seed}.csv")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir} --output {result_file}")

            elif model == "ctabgan_plus":
                # Originally used PyTorch environment
                script_path = os.path.join(base_dir, "Scripts", "ctabganplus_train.py")
                
                # Make sure the exact seed directory exists
                if not os.path.exists(seed_data_dir):
                    print(f"ERROR: Seed data directory not found: {seed_data_dir}")
                
                # Build parent directory for dataset
                dataset_parent_dir = os.path.join(base_dir, "Data")
                
                run_command(
                    f"python {script_path} --dataset_name {dataset_name} --real_data_dir {dataset_parent_dir} --data_dir {seed_data_dir} --size_category {size_category} --device cuda --seed {seed}")

                eval_script = os.path.join(base_dir, "Scripts", "tstr_evaluation.py")
                results_dir = os.path.join(base_dir, "Results", dataset_name)
                os.makedirs(results_dir, exist_ok=True)
                result_file = os.path.join(results_dir, f"{model}_tstr_seed{seed}.csv")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir} --output {result_file}")

            elif model == "tabddpm":
                # Originally used PyTorch environment
                script_path = os.path.join(base_dir, "Scripts", "tabddpm_train.py")
                
                # Make sure the exact seed directory exists
                if not os.path.exists(seed_data_dir):
                    print(f"ERROR: Seed data directory not found: {seed_data_dir}")
                
                # Build parent directory for dataset
                dataset_parent_dir = os.path.join(base_dir, "Data")
                
                run_command(f"python {script_path} --dataset {dataset_name} --real_data_dir {dataset_parent_dir} --data_dir {seed_data_dir} --seed {seed}")

                eval_script = os.path.join(base_dir, "Scripts", "tstr_evaluation.py")
                results_dir = os.path.join(base_dir, "Results", dataset_name)
                os.makedirs(results_dir, exist_ok=True)
                result_file = os.path.join(results_dir, f"{model}_tstr_seed{seed}.csv")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir} --output {result_file}")

            elif model == "tabsyn":
                # Originally used PyTorch environment
                # No need to run create_npy.py as we already created .npy files during preprocessing
                
                vae_script = os.path.join(base_dir, "tabsyn", "vae", "main.py")
                run_command(f"python {vae_script} --dataname {dataset_name} --gpu {gpu_id} --seed {seed} --data_dir {seed_data_dir}")

                train_script = os.path.join(base_dir, "Scripts", "tabsyn_train.py")
                run_command(f"python {train_script} --dataset {dataset_name} --seed {seed} --data_dir {seed_data_dir}")

                eval_script = os.path.join(base_dir, "Scripts", "tstr_evaluation.py")
                results_dir = os.path.join(base_dir, "Results", dataset_name)
                os.makedirs(results_dir, exist_ok=True)
                result_file = os.path.join(results_dir, f"{model}_tstr_seed{seed}.csv")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir} --output {result_file}")

            elif model == "great":
                # Originally used PyTorch environment
                script_path = os.path.join(base_dir, "Scripts", "great_train.py")
                
                # Make sure the exact seed directory exists
                if not os.path.exists(seed_data_dir):
                    print(f"ERROR: Seed data directory not found: {seed_data_dir}")
                
                # Build parent directory for dataset
                dataset_parent_dir = os.path.join(base_dir, "Data")
                
                run_command(f"python {script_path} --dataset {dataset_name} --real_data_dir {dataset_parent_dir} --data_dir {seed_data_dir} --seed {seed}")

                eval_script = os.path.join(base_dir, "Scripts", "tstr_evaluation.py")
                results_dir = os.path.join(base_dir, "Results", dataset_name)
                os.makedirs(results_dir, exist_ok=True)
                result_file = os.path.join(results_dir, f"{model}_tstr_seed{seed}.csv")
                run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir} --output {result_file}")

            elif model == "rlig":
                # Use our internal train_rlig and sample_rlig functions instead of the script
                if RLIG_AVAILABLE:
                    # Load the training data
                    X_train = pd.read_csv(os.path.join(seed_data_dir, "x_train.csv"))
                    y_train = pd.read_csv(os.path.join(seed_data_dir, "y_train.csv"))
                    X_test = pd.read_csv(os.path.join(seed_data_dir, "x_test.csv"))
                    
                    # Calculate the number of samples to generate (same as training set size)
                    n_samples = len(X_train)
                    
                    print(f"Training RLiG model on {dataset_name} with seed {seed}")
                    # Train the RLiG model
                    rlig_model = train_rlig(X_train, y_train, episodes=2, epochs=30)
                    
                    if rlig_model is not None:
                        # Generate synthetic samples
                        print(f"Generating {n_samples} synthetic samples using RLiG model")
                        synthetic_data = sample_rlig(rlig_model, X_train, y_train, n_samples)
                        
                        # Save synthetic data
                        if synthetic_data is not None:
                            # Ensure the synthetic directory exists
                            os.makedirs(synthetic_dir, exist_ok=True)
                            synthetic_file = os.path.join(synthetic_dir, "synthetic_data.csv")
                            synthetic_data.to_csv(synthetic_file, index=False)
                            print(f"Saved synthetic data to {synthetic_file}")
                            
                            # Run evaluation
                            eval_script = os.path.join(base_dir, "Scripts", "tstr_evaluation.py")
                            results_dir = os.path.join(base_dir, "Results", dataset_name)
                            os.makedirs(results_dir, exist_ok=True)
                            result_file = os.path.join(results_dir, f"{model}_tstr_seed{seed}.csv")
                            run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir} --output {result_file}")
                        else:
                            print(f"Failed to generate synthetic data for {dataset_name} with RLiG model")
                    else:
                        print(f"Failed to train RLiG model for {dataset_name}")
                else:
                    # Fall back to script if RLiG is not available
                    print("RLiG model not available in environment, falling back to script")
                    script_path = os.path.join(base_dir, "Scripts", "rlig_train.py")
                    run_command(
                        f"python {script_path} --dataset {dataset_name} --size_category {size_category} --seed {seed} --data_dir {seed_data_dir} "
                        f"--k 1 --epochs 20 --n 1")
    
                    eval_script = os.path.join(base_dir, "Scripts", "tstr_evaluation.py")
                    results_dir = os.path.join(base_dir, "Results", dataset_name)
                    os.makedirs(results_dir, exist_ok=True)
                    result_file = os.path.join(results_dir, f"{model}_tstr_seed{seed}.csv")
                    run_command(f"python {eval_script} --synthetic_dir {synthetic_dir} --real_test_dir {seed_data_dir} --output {result_file}")

            print(f"Completed {model} for {dataset_name} with seed {seed}")

        # After running with all seeds, compute the average of the TSTR evaluation results
        if len(seeds) > 1:
            print(f"\nComputing average TSTR evaluation results for {model} across {len(seeds)} seeds")
            results_dir = os.path.join(base_dir, "Results", dataset_name)
            os.makedirs(results_dir, exist_ok=True)

            # Use our pre-created script to average the TSTR evaluation results across seeds
            avg_script = os.path.join(base_dir, "Scripts", "average_tstr_results.py")

            # Run the averaging script for TSTR evaluation results
            result_file = os.path.join(results_dir, f"{model}_tstr_avg.csv")
            run_command(f"python {avg_script} --results_dir {results_dir} --model {model} --output_file {result_file}")

            print(f"Completed averaging TSTR evaluation results for {model} on {dataset_name}")

        print(f"Completed all runs for {model} on {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Tabular Data Generation Benchmark Runner with Uniform Preprocessing")
    parser.add_argument('--datasets', nargs='+', help='List of specific datasets to run (default: all)')
    parser.add_argument('--models', nargs='+',
                        choices=['ganblr', 'ganblrplus', 'ctabgan', 'ctgan', 'ctabgan_plus', 'tabddpm', 'tabsyn',
                                 'great', 'rlig', 'all'],
                        default=['all'], help='Models to run')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--single_run', action='store_true',
                        help='Run just one model on one dataset (useful for testing)')
    parser.add_argument('--dataset', type=str, help='Dataset name for single run')
    parser.add_argument('--uci_id', type=int, help='UCI ML Repository dataset ID for single run')
    parser.add_argument('--model', type=str,
                        choices=['ganblr', 'ganblrplus', 'ctabgan', 'ctgan', 'ctabgan_plus', 'tabddpm', 'tabsyn',
                                 'great', 'rlig'],
                        help='Model name for single run')
    parser.add_argument('--size', type=str, choices=['small', 'medium', 'large'], default='medium',
                        help='Size category for single run')
    parser.add_argument('--create_envs', action='store_true', help='Create conda environments (only needed once)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44],
                        help='List of random seeds to use (default: 42, 43, 44)')
    parser.add_argument('--discretize', type=lambda x: str(x).lower() == 'true', default=True,
                       help='Whether to discretize numerical features (default: True)')
    parser.add_argument('--numeric_cols', type=lambda x: str(x).lower() == 'true', default=True,
                       help='Use numeric column names (0,1,2...) instead of original column names (default: True)')
    args = parser.parse_args()

    # Handle single run case
    if args.single_run:
        if (not args.dataset and not args.uci_id) or not args.model:
            print("Error: Either --dataset or --uci_id must be specified along with --model for --single_run")
            return

        # For single run, we'll create a streamlined version
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        dataset_name = args.dataset
        dataset_path = None
        
        # Handle UCI dataset ID if provided
        if args.uci_id:
            if UCI_AVAILABLE:
                print(f"Using UCI ML Repository dataset with ID {args.uci_id}")
                dataset_name = f"uci_{args.uci_id}"  # Create a unique name based on UCI ID
                dataset_path = args.uci_id  # Use the ID as the "path"
            else:
                print("Error: UCI ML Repository package not available. Please install with 'pip install ucimlrepo'")
                return
        else:
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
                
            dataset_path = dataset_paths[args.dataset]

        # Process just this single dataset with the selected model
        process_dataset(dataset_name, dataset_path, args.size, [args.model], base_dir, args.gpu,
                        args.seeds, args.discretize)
        return

    # Determine which models to run
    models_to_run = []
    if 'all' in args.models:
        models_to_run = ['ganblr', 'ganblrplus', 'ctabgan', 'ctgan', 'ctabgan_plus', 'tabddpm', 'tabsyn', 'great',
                         'rlig']
    else:
        models_to_run = args.models

    print(f"Will run the following models: {', '.join(models_to_run)}")
    print(f"Using discretization: {args.discretize}")

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
    
    # Add UCI datasets if specified
    if args.uci_id and UCI_AVAILABLE:
        print(f"Adding UCI ML Repository dataset with ID {args.uci_id}")
        # Determine size category based on dataset size (can be customized)
        size = 'medium'  # Default size
        dataset_name = f"uci_{args.uci_id}" 
        
        # Add to appropriate size category
        if size in dataset_groups:
            dataset_groups[size].append((dataset_name, args.uci_id))
        else:
            dataset_groups[size] = [(dataset_name, args.uci_id)]

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
            process_dataset(dataset_name, dataset_path, size_category, models_to_run, base_dir, args.gpu, args.seeds, args.discretize)

    print("\nAll datasets and models have been processed with uniform preprocessing.")


if __name__ == "__main__":
    main()