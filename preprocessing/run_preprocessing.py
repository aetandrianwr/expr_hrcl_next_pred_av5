#!/usr/bin/env python3
"""
Master preprocessing script that runs the complete pipeline.

Usage:
    python preprocessing/run_preprocessing.py --dataset geolife
    python preprocessing/run_preprocessing.py --dataset diy
    python preprocessing/run_preprocessing.py --dataset all
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 80)
    print(f"{description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {str(e)}")
        return False


def preprocess_dataset(dataset_name):
    """Run preprocessing pipeline for a specific dataset."""
    print(f"\n{'='*80}")
    print(f"PREPROCESSING {dataset_name.upper()} DATASET")
    print(f"{'='*80}\n")
    
    config_path = f"configs/preprocessing/{dataset_name}.yaml"
    
    # Check if config exists
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        return False
    
    # Step 1: Run dataset-specific preprocessing
    preprocess_script = f"preprocessing/preprocess_{dataset_name}.py"
    if not run_command(
        ["python", preprocess_script, "--config", config_path],
        f"Step 1: Preprocessing {dataset_name} data"
    ):
        return False
    
    # Step 2: Generate transformer data
    if not run_command(
        ["python", "preprocessing/generate_transformer_data.py", "--config", config_path],
        f"Step 2: Generating transformer data for {dataset_name}"
    ):
        return False
    
    print(f"\n{'='*80}")
    print(f"✓ {dataset_name.upper()} PREPROCESSING COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    
    # Print output files
    output_dir = f"data/{dataset_name}"
    print(f"\nOutput files generated in {output_dir}:")
    print(f"  - {dataset_name}_transformer_7_train.pk")
    print(f"  - {dataset_name}_transformer_7_validation.pk")
    print(f"  - {dataset_name}_transformer_7_test.pk")
    print(f"  - dataSet_{dataset_name}.csv")
    print(f"  - locations_{dataset_name}.csv")
    print(f"  - valid_ids_{dataset_name}.pk")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline for mobility datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["geolife", "diy", "all"],
        required=True,
        help="Dataset to preprocess (geolife, diy, or all)"
    )
    args = parser.parse_args()
    
    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    success = True
    
    if args.dataset == "all":
        # Process both datasets
        datasets = ["geolife", "diy"]
        for dataset in datasets:
            if not preprocess_dataset(dataset):
                success = False
                print(f"\n✗ Failed to preprocess {dataset}")
            else:
                print(f"\n✓ Successfully preprocessed {dataset}")
    else:
        # Process single dataset
        success = preprocess_dataset(args.dataset)
    
    if success:
        print("\n" + "=" * 80)
        print("ALL PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("PREPROCESSING FAILED")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
