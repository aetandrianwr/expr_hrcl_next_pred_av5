"""
Generate transformer-format pickle files from preprocessed dataset.

This script creates the final .pk files needed for model training.

Usage:
    python preprocessing/generate_transformer_data.py --config configs/preprocessing/geolife.yaml
"""

import os
import sys
import pickle
import argparse
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from joblib import Parallel, delayed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path):
    """Load preprocessing configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_pk_file(save_path):
    """Function to load data from pickle format given path."""
    return pickle.load(open(save_path, "rb"))


def save_pk_file(save_path, data):
    """Function to save data to pickle format given data and path."""
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def split_dataset(totalData):
    """Split dataset into train, vali and test."""
    totalData = totalData.groupby("user_id", group_keys=False).apply(get_split_days_user)
    
    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()
    
    # final cleaning
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)
    
    return train_data, vali_data, test_data


def get_split_days_user(df):
    """Split the dataset according to the tracked day of each user."""
    maxDay = df["start_day"].max()
    train_split = maxDay * 0.6
    vali_split = maxDay * 0.8
    
    df["Dataset"] = "test"
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    df.loc[(df["start_day"] >= train_split) & (df["start_day"] < vali_split), "Dataset"] = "vali"
    
    return df


def get_valid_sequence_user(df, previous_day, valid_ids):
    """Get the valid sequences per user."""
    df.reset_index(drop=True, inplace=True)
    
    data_single_user = []
    
    # get the day of tracking
    min_days = df["start_day"].min()
    df["diff_day"] = df["start_day"] - min_days
    
    for index, row in df.iterrows():
        # exclude the first records that do not include enough previous_day
        if row["diff_day"] < previous_day:
            continue
        
        # get the history records [curr-previous, curr]
        hist = df.iloc[:index]
        hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_day))]
        
        # should be in the valid user ids
        if not (row["id"] in valid_ids):
            continue
        
        # require at least 3 records (2 history + 1 current)
        if len(hist) < 2:
            continue
        
        data_dict = {}
        # get the features: location, user, weekday, start time, duration, diff to curr day
        data_dict["X"] = hist["location_id"].values
        data_dict["user_X"] = hist["user_id"].values
        data_dict["weekday_X"] = hist["weekday"].values
        data_dict["start_min_X"] = hist["start_min"].values
        data_dict["dur_X"] = hist["duration"].values
        data_dict["diff"] = (row["diff_day"] - hist["diff_day"]).astype(int).values
        
        # the next location is the target
        data_dict["Y"] = int(row["location_id"])
        
        # append the single sample to list
        data_single_user.append(data_dict)
    
    return data_single_user


def apply_parallel(dfGrouped, func, n_jobs, **kwargs):
    """Function wrapper to parallelize functions after .groupby()."""
    if n_jobs == 1:
        return dfGrouped.apply(func, **kwargs)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for name, group in dfGrouped
    )
    return results


def generate_transformer_data(config):
    """Generate transformer format pickle files."""
    print("=" * 80)
    print("GENERATING TRANSFORMER DATA FILES")
    print("=" * 80)
    
    dataset_name = config['dataset']['name']
    output_dir = config['dataset']['output_dir']
    previous_day = config['sequence_generation']['previous_days'][0]
    
    # Load the preprocessed dataset
    dataset_path = os.path.join(output_dir, f"dataSet_{dataset_name}.csv")
    valid_ids_path = os.path.join(output_dir, f"valid_ids_{dataset_name}.pk")
    
    print(f"\nLoading dataset from: {dataset_path}")
    ori_data = pd.read_csv(dataset_path)
    
    print(f"Loading valid IDs from: {valid_ids_path}")
    valid_ids = load_pk_file(valid_ids_path)
    
    # Sort data
    ori_data.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    
    # Truncate too long duration: > 2days to 2 days
    ori_data.loc[ori_data["duration"] > 60 * 24 * 2 - 1, "duration"] = 60 * 24 * 2 - 1
    
    # Split dataset
    print("\nSplitting dataset...")
    train_data, vali_data, test_data = split_dataset(ori_data)
    print(f"Train: {len(train_data)}, Val: {len(vali_data)}, Test: {len(test_data)}")
    
    # Encode locations
    print("\nEncoding locations...")
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        train_data["location_id"].values.reshape(-1, 1)
    )
    # add 2 to account for unseen locations (1) and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2
    
    print(f"Max location ID: {train_data['location_id'].max()}")
    print(f"Unique locations: {train_data['location_id'].nunique()}")
    
    # Generate sequences for each split
    print(f"\nGenerating sequences (previous_day={previous_day})...")
    
    print("Processing train data...")
    train_records = apply_parallel(
        train_data.groupby("user_id"),
        get_valid_sequence_user,
        n_jobs=-1,
        previous_day=previous_day,
        valid_ids=valid_ids
    )
    train_records = [item for sublist in train_records for item in sublist]
    print(f"Train sequences: {len(train_records)}")
    
    print("Processing validation data...")
    vali_records = apply_parallel(
        vali_data.groupby("user_id"),
        get_valid_sequence_user,
        n_jobs=-1,
        previous_day=previous_day,
        valid_ids=valid_ids
    )
    vali_records = [item for sublist in vali_records for item in sublist]
    print(f"Validation sequences: {len(vali_records)}")
    
    print("Processing test data...")
    test_records = apply_parallel(
        test_data.groupby("user_id"),
        get_valid_sequence_user,
        n_jobs=-1,
        previous_day=previous_day,
        valid_ids=valid_ids
    )
    test_records = [item for sublist in test_records for item in sublist]
    print(f"Test sequences: {len(test_records)}")
    
    # Save the pickle files
    print("\nSaving transformer data files...")
    train_file = os.path.join(output_dir, f"{dataset_name}_transformer_{previous_day}_train.pk")
    vali_file = os.path.join(output_dir, f"{dataset_name}_transformer_{previous_day}_validation.pk")
    test_file = os.path.join(output_dir, f"{dataset_name}_transformer_{previous_day}_test.pk")
    
    save_pk_file(train_file, train_records)
    print(f"Saved: {train_file}")
    
    save_pk_file(vali_file, vali_records)
    print(f"Saved: {vali_file}")
    
    save_pk_file(test_file, test_records)
    print(f"Saved: {test_file}")
    
    print("\n" + "=" * 80)
    print("TRANSFORMER DATA GENERATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate transformer format data files")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to preprocessing configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    seed = config.get('seed', 42)
    np.random.seed(seed)
    
    # Generate data
    generate_transformer_data(config)
