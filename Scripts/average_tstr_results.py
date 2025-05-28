#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
import argparse
import glob
import re

def main():
    parser = argparse.ArgumentParser(description="Average evaluation results across multiple seeds")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing evaluation results')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--output_file', type=str, required=True, help='Output file for averaged results')
    args = parser.parse_args()

    # Find all seed result files
    result_pattern = os.path.join(args.results_dir, f"{args.model}_tstr_seed*.csv")
    result_files = glob.glob(result_pattern)

    if not result_files:
        print(f"No result files found matching pattern: {result_pattern}")
        return

    print(f"Found {len(result_files)} result files: {result_files}")

    # Process evaluation results
    all_dfs = []
    for result_file in result_files:
        try:
            df = pd.read_csv(result_file)
            # Add seed number as a column
            seed_match = re.search(r'seed(\d+)', result_file)
            if seed_match:
                seed = seed_match.group(1)
                df['Seed'] = seed
            all_dfs.append(df)
            print(f"Successfully read {result_file}")
        except Exception as e:
            print(f"Error reading {result_file}: {e}")

    # If we have dataframes, average them
    if all_dfs:
        # Stack all dataframes
        combined_df = pd.concat(all_dfs)
        
        # Group by model and metric and calculate mean and std
        avg_df = combined_df.groupby(['Model', 'Metric']).agg({
            'Value': ['mean', 'std']
        }).reset_index()
        
        # Flatten multi-index columns
        avg_df.columns = ['Model', 'Metric', 'Mean', 'Std']
        
        # Format the output
        avg_results = []
        for _, row in avg_df.iterrows():
            result = {
                'Model': row['Model'],
                'Metric': row['Metric'],
                'Value': round(row['Mean'], 4)
            }
            avg_results.append(result)
        
        # Create the final dataframe
        final_df = pd.DataFrame(avg_results)
        
        # Save the result
        final_df.to_csv(args.output_file, index=False)
        print(f"Saved average results to {args.output_file}")
    else:
        print(f"No valid dataframes found for averaging")

if __name__ == "__main__":
    main()