import os
import gc
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import math
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
)
import xgboost as xgb
import torch

df = pd.read_csv('/kaggle/input/nfl-player-contact-detection/test_baseline_helmets.csv')
df = df.fillna(99)

det_dict = {}
for i, row in tqdm(df.iterrows()):
    vid = row['game_play'] + '_' + row['view']
    frame = row['frame']
    p_id = row['nfl_player_id']
    if vid not in det_dict:
        det_dict[vid] = {}

    if frame not in det_dict[vid]:
        det_dict[vid][frame] = {}

    if p_id not in det_dict[vid][frame]:
        if 'H' in row['player_label']:
            t = 'home'
        else:
            t = 'v'

        det_dict[vid][frame][p_id] = {'box': [row['left'], row['top'], row['width'], row['height']], 'contact': [], 't': t}

np.save('det_dict.npy', det_dict)

trk_df = pd.read_csv('/kaggle/input/nfl-player-contact-detection/test_player_tracking.csv')
trk_df = trk_df[trk_df.step>-60]
print(trk_df.shape)
trk_dict = {}
for i, row in tqdm(trk_df.iterrows()):
    vid = row['game_play']
    p_id = row['nfl_player_id']
    step = row['step']
    idx = f'{vid}_{step}'
    if idx not in trk_dict:
        trk_dict[idx] = {}

    trk_dict[idx][p_id] = {'x': row['x_position'], 'y': row['y_position'], 't': row['team']}

np.save('trk_pos.npy', trk_dict)

trk_df = pd.read_csv('/kaggle/input/nfl-player-contact-detection/test_player_tracking.csv')
trk_df = trk_df[trk_df.step>-60]
print(trk_df.shape)
trk_dict = {}
for i, row in tqdm(trk_df.iterrows()):
    vid = row['game_play']
    idx = row['nfl_player_id']
    idx = f'{vid}_{idx}'
    step = row['step']
    if idx not in trk_dict:
        trk_dict[idx] = {}

    trk_dict[idx][step] = {'s': row['speed'], 'dis': row['distance'], 'dir': row['direction'], 'o': row['orientation'], 'a': row['acceleration'], 'sa': row['sa'], 'x': row['x_position'], 'y': row['y_position'], 't': row['team']}

np.save('trk_dict.npy',trk_dict)

def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df

def compute_distance(df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
    """
    Merges tracking data on player1 and 2 and computes the distance.
    """
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id",] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c+"_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] +  use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={c: c+"_2" for c in use_cols})
        .copy()
    )

    df_combo["distance"] = np.sqrt(
        np.square(df_combo["x_position_1"] - df_combo["x_position_2"])
        + np.square(df_combo["y_position_1"] - df_combo["y_position_2"])
    )
    return df_combo

use_cols = [
    'x_position', 'y_position', 'datetime'
]

df_l = expand_contact_id(pd.read_csv("/kaggle/input/nfl-player-contact-detection/sample_submission.csv"))
tr_tracking = pd.read_csv("/kaggle/input/nfl-player-contact-detection/test_player_tracking.csv")
df_l = compute_distance(df_l, tr_tracking, use_cols=use_cols)

df_l['datetime'] = df_l['datetime_1']

df_m = pd.read_csv('/kaggle/input/nfl-player-contact-detection/test_video_metadata.csv')
df_m = df_m[df_m.view=='Endzone'][['game_play', 'start_time']]
df = df_l.merge(df_m, on=['game_play'])

df['datetime'] = pd.to_datetime(df["datetime"], utc=True)
df['start_time'] = pd.to_datetime(df["start_time"], utc=True)

df['frame'] = (df['datetime'] - df['start_time'] - pd.to_timedelta(50, "ms")).astype('timedelta64[ms]')*59.94/1000

df = df[['contact_id', 'nfl_player_id_1',
       'nfl_player_id_2', 'x_position_1', 'y_position_1',
      'x_position_2', 'y_position_2', 'contact', 'frame', 'distance']]
df.to_csv('test_folds.csv', index=False, float_format='%.3f')

def feature_engineering():
    train_df = pd.read_csv('test_folds.csv')
    train_df = train_df[train_df.nfl_player_id_2 == 'G']
    train_df['step'] = train_df['contact_id'].apply(lambda x: int(x.split('_')[2]))
    train_df['vid'] = train_df['contact_id'].apply(lambda x: '_'.join(x.split('_')[:2]))

    trk_dict = np.load('trk_dict.npy', allow_pickle=True).item()
    results = []
    nan_val = 0
    window_size = 25
    k=0
    for i, row in tqdm(train_df.iterrows()):
        vid = row['vid']
        idx = row['nfl_player_id_1']
        idx = f'{vid}_{idx}'
        step = row['step']

        agg_dict = {'s': [], 'dis': [], 'dir': [], 'o': [], 'a': [], 'sa': [], 'x': [], 'y': []}

        if idx not in trk_dict:
            item = {'s': nan_val, 'dis': nan_val, 'dir': nan_val, 'o': nan_val, 'a': nan_val, 'sa': nan_val, 'x': nan_val, 'y': nan_val}
            for j in range(-window_size,window_size):
                item[f's_{j}'] = nan_val
                item[f'dis_{j}'] = nan_val
                item[f'dir_{j}'] = nan_val
                item[f'o_{j}'] = nan_val
                item[f'a_{j}'] = nan_val
                item[f'sa_{j}'] = nan_val
                item[f'x_{j}'] = nan_val
                item[f'y_{j}'] = nan_val
        else:
            if step in trk_dict[idx]:
                item = {'s': trk_dict[idx][step]['s'], 'dis': trk_dict[idx][step]['dis'], 'dir': trk_dict[idx][step]['dir'], 'o': trk_dict[idx][step]['o']} 
                item['a'] = trk_dict[idx][step]['a']
                item['sa'] = trk_dict[idx][step]['sa']
                item['x'] = trk_dict[idx][step]['x']
                item['y'] = trk_dict[idx][step]['y'] 
            else:
                item = {'s': nan_val, 'dis': nan_val, 'dir': nan_val, 'o': nan_val, 'a': nan_val, 'sa': nan_val, 'x': nan_val, 'y': nan_val}
            for j in range(-window_size,window_size):
                step1 = step + j 

                if j == 0:
                    continue
                
                if step1 in trk_dict[idx]:
                    item[f's_{j}'] = item[f's'] - trk_dict[idx][step1]['s']
                    item[f'dis_{j}'] = item[f'dis'] - trk_dict[idx][step1]['dis']
                    item[f'dir_{j}'] = item[f'dir'] - trk_dict[idx][step1]['dir']
                    item[f'o_{j}'] = item[f'o'] - trk_dict[idx][step1]['o']
                    item[f'a_{j}'] = item[f'a'] - trk_dict[idx][step1]['a']
                    item[f'sa_{j}'] = item[f'sa'] - trk_dict[idx][step1]['sa']
                    item[f'x_{j}'] = item[f'x'] - trk_dict[idx][step1]['x']
                    item[f'y_{j}'] = item[f'y'] - trk_dict[idx][step1]['y']
                else:
                    item[f's_{j}'] = nan_val
                    item[f'dis_{j}'] = nan_val
                    item[f'dir_{j}'] = nan_val
                    item[f'o_{j}'] = nan_val
                    item[f'a_{j}'] = nan_val
                    item[f'sa_{j}'] = nan_val
                    item[f'x_{j}'] = nan_val
                    item[f'y_{j}'] = nan_val

        # item['step'] = row['step']

        if k==0: feature_cols = list(item.keys())
        k+=1
        
        item['step'] = row['step']        
        item['fold'] = -1
        item['contact'] = row['contact']
        item['contact_id'] = row['contact_id']
        item['frame'] = row['frame']        
        

        results.append(item)

    train_df = pd.DataFrame(results)

    return train_df, feature_cols

train_df, feature_cols = feature_engineering()
_ = gc.collect()


x_valid = train_df[feature_cols]
dvalid = xgb.DMatrix(x_valid)


for fold in [0,1,2,3,4]:
    model_path = f'/kaggle/input/nfl-data/xgb_models/xgb_models/xgb_fold{fold}_xgb_1st.model'
    model = xgb.Booster()
    model.load_model(model_path)
    
    if fold==0:
        pred_i = model.predict(dvalid) 
    else:
        pred_i += model.predict(dvalid)
    # print(pred_i.shape)
    
pred_i = pred_i/5

train_df['pred'] = pred_i
train_df = train_df[['contact_id', 'contact', 'pred', 'frame']]

# print(train_df[train_df.pred>0.1].head(10))

train_df.to_csv('test_G.csv', index=False)