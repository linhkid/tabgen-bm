#!/usr/bin/env python
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
