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

def extract_feat(idx, trk_dict, step, nan_val=0, window_size=10):
    if idx not in trk_dict:
        item = {'s': nan_val, 'dis': nan_val, 'dir': nan_val, 'o': nan_val, 'a': nan_val, 'sa': nan_val, 'x': nan_val, 'y': nan_val, 't': nan_val}
    else:
        if step in trk_dict[idx]:
            item = {'s': trk_dict[idx][step]['s'], 'dis': trk_dict[idx][step]['dis'], 'dir': trk_dict[idx][step]['dir'], 'o': trk_dict[idx][step]['o']} 
            item['a'] = trk_dict[idx][step]['a']
            item['sa'] = trk_dict[idx][step]['sa']
            item['x'] = trk_dict[idx][step]['x']
            item['y'] = trk_dict[idx][step]['y'] 
            item['t'] = trk_dict[idx][step]['t'] 
        else:
            item = {'s': nan_val, 'dis': nan_val, 'dir': nan_val, 'o': nan_val, 'a': nan_val, 'sa': nan_val, 'x': nan_val, 'y': nan_val, 't': nan_val}

    return item

def calc_dist(idx1, idx2, trk_dict, step, nan_val=0):
    if idx1 not in trk_dict or idx2 not in trk_dict:
        return nan_val, nan_val, nan_val
    else:
        if step not in trk_dict[idx1] or step not in trk_dict[idx2]:
            return nan_val, nan_val, nan_val

        x1 = trk_dict[idx1][step]['x']
        y1 = trk_dict[idx1][step]['y'] 

        x2 = trk_dict[idx2][step]['x']
        y2 = trk_dict[idx2][step]['y'] 

        dist = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

        sa_dif = trk_dict[idx1][step]['sa'] - trk_dict[idx2][step]['sa'] 
        a_dif = trk_dict[idx1][step]['a'] - trk_dict[idx2][step]['a'] 

        return dist, a_dif, sa_dif

def feature_engineering():
    train_df = pd.read_csv('test_folds.csv')
    train_df = train_df[train_df.nfl_player_id_2 != 'G']
    train_df = train_df[train_df.distance<4.0]# 3
    # feature_cols = ['speed_1','distance_1', 'direction_1', 'orientation_1', 'acceleration_1', 'sa_1', 'x_position', 'y_position']
    train_df['step'] = train_df['contact_id'].apply(lambda x: int(x.split('_')[2]))
    train_df['vid'] = train_df['contact_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
    print(train_df.shape)

    train_df['game_play'] = train_df['vid']

    trk_dict = np.load('trk_dict.npy', allow_pickle=True).item()
    det_dict = np.load('det_dict.npy', allow_pickle=True).item()

    results = []
    nan_val = np.nan
    window_size = 2
    m=0
    for i, row in tqdm(train_df.iterrows()):
        vid = row['vid']
        idx = row['nfl_player_id_1']
        idx = f'{vid}_{idx}'

        idx2 = row['nfl_player_id_2']
        idx2 = f'{vid}_{idx2}'

        step = row['step']

        item1 = extract_feat(idx, trk_dict, step, nan_val=nan_val, window_size=window_size)
        item2 = extract_feat(idx2, trk_dict, step, nan_val=nan_val, window_size=window_size)

        item = {}
        for k, val in item1.items():
            if k in ['t']:
                if val == item2[k]:
                    item[k] = 0
                else:
                    item[k] = 1
                continue

            item[k] = val 
            item[f'{k}_2'] = item2[k]

            if k not in ['pos']:
                item[f'{k}_dif'] = val - item2[k]

            if k in ['o', 'dir']:
                item[f'{k}_s'] = math.sin(val)
                item[f'{k}_c'] = math.cos(val)
                item[f'{k}_s2'] = math.sin(item2[k])
                item[f'{k}_c2'] = math.cos(item2[k])
                item[f'{k}_sd'] = math.sin(val - item2[k])
                item[f'{k}_cd'] = math.cos(val - item2[k])

            if k in ['o', 'dir']:
                item[f'{k}_s'] = math.sin(math.pi*val/180)
                item[f'{k}_c'] = math.cos(math.pi*val/180)
                item[f'{k}_s2'] = math.sin(math.pi*item2[k]/180)
                item[f'{k}_c2'] = math.cos(math.pi*item2[k]/180)
                item[f'{k}_sd'] = math.sin(math.pi*(val - item2[k])/180)
                item[f'{k}_cd'] = math.cos(math.pi*(val - item2[k])/180)


        item['distance'] = row['distance']
        item['step'] = row['step']

        for j in range(20):
            dist, a_dif, sa_dif = calc_dist(idx, idx2, trk_dict, step+1+j, nan_val=np.nan)
            item[f'dist_{j}'] = dist
            if j<20:
                item[f'a_{j}'] = a_dif
                item[f'sa_{j}'] = sa_dif

        for j in range(20):
            dist, a_dif, sa_dif = calc_dist(idx, idx2, trk_dict, step-1-j, nan_val=np.nan)
            item[f'dist_p{j}'] = dist
            if j<20:
                item[f'a_p{j}'] = a_dif
                item[f'sa_p{j}'] = sa_dif

        idx1 = int(row['nfl_player_id_1'])
        idx2 = int(row['nfl_player_id_2'])
        step = row['step']
        frame = int(row['frame'])+6

        for view in ['Sideline', 'Endzone']:
            v_vid = vid + '_' + view
            area1_list = []
            area2_list = []
            for ff in range(-30,30,2):
                fr = frame + ff
                if fr in det_dict[v_vid] and idx1 in det_dict[v_vid][fr] and idx2 in det_dict[v_vid][fr]: 
                    x1, y1, w1, h1 = det_dict[v_vid][fr][idx1]['box']
                    x2, y2, w2, h2 = det_dict[v_vid][fr][idx2]['box']

                    x1 = x1 + w1/2
                    y1 = y1 + h1/2
                    x2 = x2 + w2/2
                    y2 = y2 + h2/2
                    dist = math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

                    item[f'{view}_{ff}_dist'] = dist
                    area1_list.append(w1*h1)
                    area2_list.append(w2*h2)
                else:
                    item[f'{view}_{ff}_dist'] = np.nan

            if len(area2_list)>0:
                item[f'{view}_area1'] = np.mean(area1_list)
                item[f'{view}_area2'] = np.mean(area2_list)
            else:
                item[f'{view}_area1'] = np.nan
                item[f'{view}_area2'] = np.nan


        if m==0: feature_cols = list(item.keys())
        m+=1

        item['fold'] = -1
        item['contact'] = row['contact']
        item['contact_id'] = row['contact_id']
        item['frame'] = row['frame']
        item['nfl_player_id_1'] = row['nfl_player_id_1']
        item['nfl_player_id_2'] = row['nfl_player_id_2']

        results.append(item)

    train_df = pd.DataFrame(results)

    return train_df, feature_cols

train_df, feature_cols = feature_engineering()
_ = gc.collect()

x_valid = train_df[feature_cols]
dvalid = xgb.DMatrix(x_valid)
for fold in [0,1,2,3,4]:
    model_path = f'/kaggle/input/nfl-data/xgb_models/xgb_models/xgb_not_fold{fold}_xgb_1st.model'
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

train_df.to_csv('test_pair_xgb_v1.csv', index=False)