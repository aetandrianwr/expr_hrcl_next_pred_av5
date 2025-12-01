import json
import os
import pickle as pickle
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
import yaml

import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import argparse
from shapely.geometry import Point

# trackintel
from trackintel.preprocessing.triplegs import generate_trips
import trackintel as ti

from utils import calculate_user_quality, enrich_time_info, split_dataset, get_valid_sequence


def get_dataset(paths_config, preprocess_config):
    """Construct the raw staypoint with location id dataset from DIY data - skipping first part."""
    
    # Extract all parameters from config
    output_dir = preprocess_config['dataset']['output_dir']
    dataset_name = preprocess_config['dataset']['name']
    timezone = preprocess_config['dataset']['timezone']
    
    # Staypoint parameters
    loc_params = preprocess_config['locations']
    merge_params = preprocess_config['staypoint_merging']
    seq_params = preprocess_config['sequence_generation']
    
    print(f"Using epsilon={loc_params['epsilon']} for location clustering")
    print(f"Using timezone={timezone}")
    
    # Read staypoints from preprocessed file
    print("Reading preprocessed staypoints...")
    sp = ti.read_staypoints_csv('/content/drive/MyDrive/next_location_prediction/data/03_processed/stayloc_full_trackintel_30min_100m/3_staypoints_fun_generate_trips.csv', columns={'geometry':'geom'}, index_col='id')
    print(f"Loaded {len(sp)} staypoints")
    
    # Read valid users
    print("Reading valid users...")
    valid_user_df = pd.read_csv('/content/drive/MyDrive/next_location_prediction/data/03_processed/stayloc_full_trackintel_30min_100m/10_filter_after_user_quality_DIY_slide_filteres.csv')
    valid_user = valid_user_df["user_id"].values
    print(f"Loaded {len(valid_user)} valid users")

    sp = sp.loc[sp["user_id"].isin(valid_user)]
    print(f"Valid users after quality filter: {len(valid_user)}")

    # filter activity staypoints
    sp = sp.loc[sp["is_activity"] == True]
    print(f"Activity staypoints: {len(sp)}")
    
    # Check if we have any data to process
    if len(sp) == 0:
        print("Error: No valid staypoints found after quality filtering. Cannot proceed.")
        print("This might be due to:")
        print("  1. Sample size too small (try increasing --sample parameter)")
        print("  2. Quality thresholds too strict")
        print("  3. Data quality issues")
        return

    # generate locations
    print("Generating locations...")
    sp, locs = sp.as_staypoints.generate_locations(
        epsilon=loc_params['epsilon'],
        num_samples=loc_params['num_samples'],
        distance_metric=loc_params['distance_metric'],
        agg_level=loc_params['agg_level'],
        n_jobs=loc_params['n_jobs']
    )
    # filter noise staypoints
    sp = sp.loc[~sp["location_id"].isna()].copy()
    print("After filter non-location staypoints: ", sp.shape[0])

    # save locations
    locs = locs[~locs.index.duplicated(keep="first")]
    filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]
    filtered_locs.as_locations.to_csv(os.path.join(".", output_dir, f"locations_{dataset_name}.csv"))
    print("Location size: ", sp["location_id"].unique().shape[0], filtered_locs.shape[0])

    sp = sp[["user_id", "started_at", "finished_at", "geom", "location_id"]]
    # Reset index to ensure it's named 'id' for merge_staypoints
    if sp.index.name != 'id':
        sp.index.name = 'id'
    # merge staypoints
    print("Merging staypoints...")
    sp_merged = sp.as_staypoints.merge_staypoints(
        triplegs=pd.DataFrame([]),
        max_time_gap=merge_params['max_time_gap'],
        agg={"location_id": "first"}
    )
    print("After staypoints merging: ", sp_merged.shape[0])
    # recalculate staypoint duration
    sp_merged["duration"] = (sp_merged["finished_at"] - sp_merged["started_at"]).dt.total_seconds() // 60

    # get the time info
    sp_time = enrich_time_info(sp_merged)

    print("User size: ", sp_time["user_id"].unique().shape[0])

    # save intermediate results for analysis
    sp_time.to_csv(f"./{output_dir}/sp_time_temp_{dataset_name}.csv", index=False)

    # Get split parameters
    split_params = preprocess_config.get('dataset_split', {'train_ratio': 0.6, 'val_ratio': 0.2, 'test_ratio': 0.2})
    
    #
    _filter_sp_history(sp_time, output_dir, dataset_name, seq_params, split_params)


def _filter_sp_history(sp, output_dir, dataset_name, seq_params, split_params):
    """To unify the comparision between different previous days"""
    # classify the datasets, user dependent (configurable ratios)
    train_data, vali_data, test_data = split_dataset(sp, split_params)
    
    # Check if we have data to process - early exit for test mode
    if len(train_data) == 0 or len(vali_data) == 0 or len(test_data) == 0:
        print(f"Warning: Insufficient data after initial split. Train: {len(train_data)}, Val: {len(vali_data)}, Test: {len(test_data)}")
        print("Skipping sequence filtering and saving all available data...")
        sp.to_csv(f"./{output_dir}/dataSet_{dataset_name}.csv", index=False)
        print("Final user size: ", sp["user_id"].unique().shape[0])
        print("Dataset saved (test mode - no train/val/test split)")
        return

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        train_data["location_id"].values.reshape(-1, 1)
    )
    # add 2 to account for unseen locations and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

    # the days to consider when generating final_valid_id
    previous_day_ls = seq_params['previous_days']
    all_ids = sp[["id"]].copy()

    # for each previous_day, get the valid staypoint id
    for previous_day in tqdm(previous_day_ls):
        valid_ids = get_valid_sequence(train_data, previous_day=previous_day)
        valid_ids.extend(get_valid_sequence(vali_data, previous_day=previous_day))
        valid_ids.extend(get_valid_sequence(test_data, previous_day=previous_day))

        all_ids[f"{previous_day}"] = 0
        all_ids.loc[all_ids["id"].isin(valid_ids), f"{previous_day}"] = 1

    # get the final valid staypoint id
    all_ids.set_index("id", inplace=True)
    final_valid_id = all_ids.loc[all_ids.sum(axis=1) == all_ids.shape[1]].reset_index()["id"].values

    # filter the user again based on final_valid_id:
    # if an user has no record in final_valid_id, we discard the user
    valid_users_train = train_data.loc[train_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_vali = vali_data.loc[vali_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_test = test_data.loc[test_data["id"].isin(final_valid_id), "user_id"].unique()

    valid_users = set.intersection(set(valid_users_train), set(valid_users_vali), set(valid_users_test))

    filtered_sp = sp.loc[sp["user_id"].isin(valid_users)].copy()

    train_data, vali_data, test_data = split_dataset(filtered_sp, split_params)
    
    # Check if we have data to process
    if len(train_data) == 0 or len(vali_data) == 0 or len(test_data) == 0:
        print(f"Warning: Insufficient data after split. Train: {len(train_data)}, Val: {len(vali_data)}, Test: {len(test_data)}")
        print("Saving minimal dataset with available data...")
        # Just save what we have
        filtered_sp.to_csv(f"./{output_dir}/dataSet_{dataset_name}.csv", index=False)
        print("Final user size: ", filtered_sp["user_id"].unique().shape[0])
        print("Dataset saved (without proper train/val/test split)")
        return

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        train_data["location_id"].values.reshape(-1, 1)
    )
    # add 2 to account for unseen locations and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    print(
        f"Max location id:{train_data.location_id.max()}, unique location id:{train_data.location_id.unique().shape[0]}"
    )

    # after user filter, we reencode the users, to ensure the user_id is continues
    # we do not need to encode the user_id again in dataloader.py
    enc = OrdinalEncoder(dtype=np.int64)
    filtered_sp["user_id"] = enc.fit_transform(filtered_sp["user_id"].values.reshape(-1, 1)) + 1

    # save the valid_ids and dataset
    data_path = f"./{output_dir}/valid_ids_{dataset_name}.pk"
    with open(data_path, "wb") as handle:
        pickle.dump(final_valid_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filtered_sp.to_csv(f"./{output_dir}/dataSet_{dataset_name}.csv", index=False)

    print("Final user size: ", filtered_sp["user_id"].unique().shape[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing/diy_skip_first_part.yaml",
        help="Path to preprocessing configuration file"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use only first N rows for testing (optional)"
    )
    args = parser.parse_args()
    
    # Load paths configuration
    DBLOGIN_FILE = os.path.join(".", "paths.json")
    with open(DBLOGIN_FILE) as json_file:
        PATHS_CONFIG = json.load(json_file)
    
    # Load preprocessing configuration
    with open(args.config, 'r') as f:
        PREPROCESS_CONFIG = yaml.safe_load(f)
    
    # Add sample parameter to config if provided
    if args.sample:
        PREPROCESS_CONFIG['sample_rows'] = args.sample
        print(f"Using sample of {args.sample} rows for testing")
    
    # Set random seed
    if 'seed' in PREPROCESS_CONFIG:
        np.random.seed(PREPROCESS_CONFIG['seed'])
    
    get_dataset(paths_config=PATHS_CONFIG, preprocess_config=PREPROCESS_CONFIG)
