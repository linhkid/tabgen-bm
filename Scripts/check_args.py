#!/usr/bin/env python
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Check command line arguments")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--size_category', type=str, required=True, help='Size category')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    args = parser.parse_args()
    
    print(f"Dataset: {args.dataset}")
    print(f"Size category: {args.size_category}")
    print(f"Seed: {args.seed}")
    print(f"Data directory: {args.data_dir}")
    
    # Print script information
    print(f"Script path: {__file__}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")

if __name__ == "__main__":
    main()