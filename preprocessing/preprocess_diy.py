"""
DIY dataset preprocessing script with YAML configuration support.

This script processes the raw DIY mobility dataset into the format required for 
the next-location prediction model.

Usage:
    python preprocessing/preprocess_diy.py --config configs/preprocessing/diy.yaml
"""

import os
import pickle
import sys
from pathlib import Path
import argparse
import yaml

import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder
from shapely.geometry import Point

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# trackintel
from trackintel.preprocessing.triplegs import generate_trips
import trackintel as ti

from preprocessing.utils import calculate_user_quality, enrich_time_info, split_dataset, get_valid_sequence


def load_config(config_path):
    """Load preprocessing configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def read_diy_dataset(raw_data_path, timezone='Asia/Jakarta'):
    """Read DIY dataset and convert to trackintel positionfixes format."""
    print(f"Reading DIY data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Convert to GeoDataFrame
    print("Converting to GeoDataFrame...")
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Rename columns to trackintel format
    gdf = gdf.rename(columns={'tracked_at': 'tracked_at_str'})
    
    # Parse timestamps and set timezone
    print("Parsing timestamps...")
    gdf['tracked_at'] = pd.to_datetime(gdf['tracked_at_str'])
    gdf['tracked_at'] = gdf['tracked_at'].dt.tz_convert(timezone)
    
    # Drop intermediate column
    gdf = gdf.drop(columns=['tracked_at_str', 'latitude', 'longitude'])
    
    # Rename geometry column
    gdf = gdf.rename(columns={'geometry': 'geom'})
    
    # Add required id column
    gdf.index.name = 'id'
    gdf = gdf.reset_index()
    
    print(f"Loaded {len(gdf)} position fixes from {gdf['user_id'].nunique()} users")
    
    return gdf


def get_dataset(config):
    """Construct the raw staypoint with location id dataset from DIY data."""
    print("=" * 80)
    print("DIY DATASET PREPROCESSING")
    print("=" * 80)
    
    # Get configuration parameters
    raw_data_path = config['dataset']['raw_data_path']
    output_dir = config['dataset']['output_dir']
    timezone = config['dataset']['timezone']
    
    # Staypoint parameters
    sp_params = config['staypoints']
    activity_params = config['activity_flag']
    quality_params = config['user_quality']
    loc_params = config['locations']
    merge_params = config['staypoint_merging']
    seq_params = config['sequence_generation']
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    quality_path = os.path.join(output_dir, "quality")
    os.makedirs(quality_path, exist_ok=True)
    
    # Read DIY dataset
    pfs = read_diy_dataset(raw_data_path, timezone)
    
    # Convert to trackintel positionfixes
    print("\nConvert to trackintel format...")
    pfs = pfs.set_index('id')
    pfs = pfs.as_positionfixes
    
    # Generate staypoints
    print("\nGenerating staypoints...")
    pfs, sp = pfs.generate_staypoints(
        gap_threshold=sp_params['gap_threshold'],
        include_last=sp_params.get('include_last', True),
        print_progress=sp_params['print_progress'],
        dist_threshold=sp_params['dist_threshold'],
        time_threshold=sp_params['time_threshold'],
        n_jobs=sp_params['n_jobs']
    )
    print(f"Generated {len(sp)} staypoints")
    
    # Create activity flag
    print("\nCreating activity flags...")
    sp = sp.as_staypoints.create_activity_flag(
        method=activity_params['method'],
        time_threshold=activity_params['time_threshold']
    )
    
    # Select valid users based on quality
    quality_file = os.path.join(quality_path, "diy_slide_filtered.csv")
    if Path(quality_file).is_file():
        print(f"\nLoading pre-computed user quality from: {quality_file}")
        valid_user = pd.read_csv(quality_file)["user_id"].values
    else:
        print("\nCalculating user quality...")
        # Generate triplegs
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
        # Generate trips
        sp_temp, tpls_temp, trips = generate_trips(sp, tpls, add_geometry=False)
        
        quality_filter = {
            "day_filter": quality_params['day_filter'],
            "window_size": quality_params['window_size']
        }
        if quality_params.get('min_thres') is not None:
            quality_filter['min_thres'] = quality_params['min_thres']
        if quality_params.get('mean_thres') is not None:
            quality_filter['mean_thres'] = quality_params['mean_thres']
            
        valid_user = calculate_user_quality(sp_temp.copy(), trips.copy(), quality_file, quality_filter)
    
    print(f"Valid users after quality filter: {len(valid_user)}")
    sp = sp.loc[sp["user_id"].isin(valid_user)]
    
    # Filter activity staypoints
    print("\nFiltering activity staypoints...")
    sp = sp.loc[sp["is_activity"] == True]
    print(f"Activity staypoints: {len(sp)}")
    
    # Generate locations
    print("\nGenerating locations...")
    sp, locs = sp.as_staypoints.generate_locations(
        epsilon=loc_params['epsilon'],
        num_samples=loc_params['num_samples'],
        distance_metric=loc_params['distance_metric'],
        agg_level=loc_params['agg_level'],
        n_jobs=loc_params['n_jobs']
    )
    
    # Filter noise staypoints
    sp = sp.loc[~sp["location_id"].isna()].copy()
    print(f"After filter non-location staypoints: {len(sp)}")
    
    # Save locations
    locs = locs[~locs.index.duplicated(keep="first")]
    filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]
    locations_file = os.path.join(output_dir, f"locations_diy.csv")
    filtered_locs.as_locations.to_csv(locations_file)
    print(f"Location size: {sp['location_id'].unique().shape[0]}")
    
    sp = sp[["user_id", "started_at", "finished_at", "geom", "location_id"]]
    
    # Merge staypoints
    print("\nMerging staypoints...")
    sp_merged = sp.as_staypoints.merge_staypoints(
        triplegs=pd.DataFrame([]),
        max_time_gap=merge_params['max_time_gap'],
        agg={"location_id": "first"}
    )
    print(f"After staypoints merging: {len(sp_merged)}")
    
    # Recalculate staypoint duration
    sp_merged["duration"] = (sp_merged["finished_at"] - sp_merged["started_at"]).dt.total_seconds() // 60
    
    # Get the time info
    print("\nEnriching time information...")
    sp_time = enrich_time_info(sp_merged)
    print(f"User size: {sp_time['user_id'].unique().shape[0]}")
    
    # Save intermediate results
    intermediate_file = os.path.join(output_dir, "sp_time_temp_diy.csv")
    sp_time.to_csv(intermediate_file, index=False)
    print(f"Saved intermediate results to: {intermediate_file}")
    
    # Filter and prepare final dataset
    _filter_sp_history(sp_time, config)


def _filter_sp_history(sp, config):
    """To unify the comparison between different previous days."""
    output_dir = config['dataset']['output_dir']
    seq_params = config['sequence_generation']
    
    print("\n" + "=" * 80)
    print("DATASET SPLITTING AND FILTERING")
    print("=" * 80)
    
    # Classify the datasets, user dependent 0.6, 0.2, 0.2
    print("\nSplitting dataset into train/val/test...")
    train_data, vali_data, test_data = split_dataset(sp)
    print(f"Train: {len(train_data)}, Val: {len(vali_data)}, Test: {len(test_data)}")
    
    # Encode unseen locations in validation and test into -1, then add 2
    print("\nEncoding location IDs...")
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        train_data["location_id"].values.reshape(-1, 1)
    )
    # add 2 to account for unseen locations and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2
    
    # Get valid sequences for each previous_day
    previous_day_ls = seq_params['previous_days']
    all_ids = sp[["id"]].copy()
    
    print(f"\nGenerating valid sequences for previous_days: {previous_day_ls}")
    for previous_day in tqdm(previous_day_ls):
        valid_ids = get_valid_sequence(train_data, previous_day=previous_day)
        valid_ids.extend(get_valid_sequence(vali_data, previous_day=previous_day))
        valid_ids.extend(get_valid_sequence(test_data, previous_day=previous_day))
        
        all_ids[f"{previous_day}"] = 0
        all_ids.loc[all_ids["id"].isin(valid_ids), f"{previous_day}"] = 1
    
    # Get the final valid staypoint id
    all_ids.set_index("id", inplace=True)
    final_valid_id = all_ids.loc[all_ids.sum(axis=1) == all_ids.shape[1]].reset_index()["id"].values
    print(f"Final valid IDs: {len(final_valid_id)}")
    
    # Filter the user again based on final_valid_id
    valid_users_train = train_data.loc[train_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_vali = vali_data.loc[vali_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_test = test_data.loc[test_data["id"].isin(final_valid_id), "user_id"].unique()
    
    valid_users = set.intersection(set(valid_users_train), set(valid_users_vali), set(valid_users_test))
    print(f"Valid users across all splits: {len(valid_users)}")
    
    filtered_sp = sp.loc[sp["user_id"].isin(valid_users)].copy()
    
    # Re-split after user filtering
    train_data, vali_data, test_data = split_dataset(filtered_sp)
    
    # Encode unseen locations in validation and test into -1, then add 2
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        train_data["location_id"].values.reshape(-1, 1)
    )
    # add 2 to account for unseen locations and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    print(f"Max location id: {train_data.location_id.max()}, unique location id: {train_data.location_id.unique().shape[0]}")
    
    # Re-encode the users to ensure the user_id is continuous
    enc = OrdinalEncoder(dtype=np.int64)
    filtered_sp["user_id"] = enc.fit_transform(filtered_sp["user_id"].values.reshape(-1, 1)) + 1
    
    # Save the valid_ids and dataset
    valid_ids_path = os.path.join(output_dir, "valid_ids_diy.pk")
    with open(valid_ids_path, "wb") as handle:
        pickle.dump(final_valid_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved valid IDs to: {valid_ids_path}")
    
    dataset_path = os.path.join(output_dir, "dataSet_diy.csv")
    filtered_sp.to_csv(dataset_path, index=False)
    print(f"Saved dataset to: {dataset_path}")
    
    print(f"\nFinal user size: {filtered_sp['user_id'].unique().shape[0]}")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DIY dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing/diy.yaml",
        help="Path to preprocessing configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    seed = config.get('seed', 42)
    np.random.seed(seed)
    
    # Run preprocessing
    get_dataset(config)
