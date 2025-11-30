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
    """Construct the raw staypoint with location id dataset from DIY data."""
    
    # Extract all parameters from config
    output_dir = preprocess_config['dataset']['output_dir']
    timezone = preprocess_config['dataset']['timezone']
    
    # Staypoint parameters
    sp_params = preprocess_config['staypoints']
    activity_params = preprocess_config['activity_flag']
    quality_params = preprocess_config['user_quality']
    loc_params = preprocess_config['locations']
    merge_params = preprocess_config['staypoint_merging']
    seq_params = preprocess_config['sequence_generation']
    
    print(f"Using epsilon={loc_params['epsilon']} for location clustering")
    print(f"Using timezone={timezone}")
    
    # Read raw CSV file
    print("Reading DIY dataset...")
    nrows = preprocess_config.get('sample_rows', None)
    if nrows:
        print(f"Sampling first {nrows} rows for testing...")
        raw_df = pd.read_csv(os.path.join(paths_config["raw_diy"], "raw_diy_mobility_dataset.csv"), nrows=nrows)
    else:
        raw_df = pd.read_csv(os.path.join(paths_config["raw_diy"], "raw_diy_mobility_dataset.csv"))
    
    print(f"Loaded {len(raw_df)} rows")
    
    # Convert to GeoDataFrame
    print("Converting to GeoDataFrame...")
    geometry = [Point(xy) for xy in zip(raw_df['longitude'], raw_df['latitude'])]
    gdf = gpd.GeoDataFrame(raw_df, geometry=geometry, crs='EPSG:4326')
    
    # Drop lat/lon columns and set index
    gdf = gdf.drop(columns=['latitude', 'longitude'])
    gdf.index.name = 'id'
    gdf = gdf.reset_index()
    
    # Parse timestamps and set timezone
    print("Parsing timestamps...")
    gdf['tracked_at'] = pd.to_datetime(gdf['tracked_at'])
    gdf['tracked_at'] = gdf['tracked_at'].dt.tz_convert(timezone)
    
    # Rename geometry column to geom AFTER setting timezone
    gdf = gdf.rename(columns={'geometry': 'geom'})
    gdf = gdf.set_geometry('geom')
    
    # Set index
    gdf = gdf.set_index('id')
    pfs = gdf.as_positionfixes
    
    print(f"Loaded {len(pfs)} position fixes from {pfs['user_id'].nunique()} users")
    
    # Generate staypoints
    print("Generating staypoints...")
    pfs, sp = pfs.as_positionfixes.generate_staypoints(
        gap_threshold=sp_params['gap_threshold'],
        include_last=sp_params['include_last'],
        print_progress=sp_params['print_progress'],
        dist_threshold=sp_params['dist_threshold'],
        time_threshold=sp_params['time_threshold'],
        n_jobs=sp_params['n_jobs']
    )
    
    # Create activity flag
    sp = sp.as_staypoints.create_activity_flag(
        method=activity_params['method'],
        time_threshold=activity_params['time_threshold']
    )

    ## select valid user, generate the file if user quality file is not generated
    quality_path = os.path.join(".", output_dir, "quality")
    quality_file = os.path.join(quality_path, "diy_slide_filtered.csv")
    if Path(quality_file).is_file():
        print("Loading pre-computed user quality...")
        valid_user = pd.read_csv(quality_file)["user_id"].values
    else:
        if not os.path.exists(quality_path):
            os.makedirs(quality_path)
        # generate triplegs
        print("Generating triplegs for user quality assessment...")
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
        # the trackintel trip generation
        sp, tpls, trips = generate_trips(sp, tpls, add_geometry=False)

        # Build quality filter from config (similar to GC preprocessing)
        quality_filter = {
            "day_filter": quality_params['day_filter'],
            "window_size": quality_params['window_size']
        }
        if quality_params.get('min_thres') is not None:
            quality_filter['min_thres'] = quality_params['min_thres']
        if quality_params.get('mean_thres') is not None:
            quality_filter['mean_thres'] = quality_params['mean_thres']
        
        print(f"Quality filter: {quality_filter}")
        valid_user = calculate_user_quality(sp.copy(), trips.copy(), quality_file, quality_filter)

    sp = sp.loc[sp["user_id"].isin(valid_user)]
    print(f"Valid users after quality filter: {len(valid_user)}")

    # filter activity staypoints
    sp = sp.loc[sp["is_activity"] == True]
    print(f"Activity staypoints: {len(sp)}")

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
    filtered_locs.as_locations.to_csv(os.path.join(".", output_dir, f"locations_diy.csv"))
    print("Location size: ", sp["location_id"].unique().shape[0], filtered_locs.shape[0])

    sp = sp[["user_id", "started_at", "finished_at", "geom", "location_id"]]
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
    sp_time.to_csv(f"./{output_dir}/sp_time_temp_diy.csv", index=False)

    #
    _filter_sp_history(sp_time, output_dir, seq_params)


def _filter_sp_history(sp, output_dir, seq_params):
    """To unify the comparision between different previous days"""
    # classify the datasets, user dependent 0.6, 0.2, 0.2
    train_data, vali_data, test_data = split_dataset(sp)

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

    train_data, vali_data, test_data = split_dataset(filtered_sp)

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
    data_path = f"./{output_dir}/valid_ids_diy.pk"
    with open(data_path, "wb") as handle:
        pickle.dump(final_valid_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filtered_sp.to_csv(f"./{output_dir}/dataSet_diy.csv", index=False)

    print("Final user size: ", filtered_sp["user_id"].unique().shape[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing/diy.yaml",
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
