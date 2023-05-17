#!/usr/bin/env python
# coding: utf-8

# In[4]:


import albumentations as A
import cv2

# "base_config.py" file is present inside folder "configs"
from configs.base_config import basic_cfg
# from base_config import basic_cfg

import os


# In[2]:


cfg = basic_cfg


# In[ ]:


cfg.debug = 0
cfg.batch_size = 6
# cfg.out_dir = f"outputs/{os.path.basename(__file__).split('.')[0]}"
cfg.out_dir = f"outputs/base"
cfg.train_csv_path = "slicing_not_g.csv"
cfg.seed = 42
cfg.epochs = 1
cfg.mode = 'train'
cfg.model_name = 'r50ir'
cfg.lr = 4e-5
cfg.scheduler = "linear" #linear  cosine, step cosinewarmup warmupv2
cfg.optimizer = "AdamW"  # Adam, SGD, AdamW
cfg.weight_decay = 1 # weigth_decay for Adam, SGD, AdamW
cfg.num_workers = 8
cfg.folds = [0,1,2,3,4,6,7,8]
cfg.model = 'model_csn'
cfg.dataset = 'dataset_3d_3ch'
cfg.drop_rate = 0.15
cfg.img_size = 256
cfg.use_swa = 0
cfg.frac = 1.8
cfg.pos_frac = 1
cfg.val_frac = 1.
cfg.is_G = 0


# In[ ]:


# https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/csn/metafile.yml
# https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_ig65m_pretrained_r50_32x2x1_58e_kinetics400_rgb_20210617-86d33018.pth
cfg.load_weight = 'pretrained/vmz_ircsn_ig65m_pretrained_r50_32x2x1_58e_kinetics400_rgb_20210617-86d33018.pth'


# In[ ]:


base_aug = [
        A.OneOf([            
            A.HorizontalFlip(p=1.), 
            A.VerticalFlip(p=1.),
            A.Transpose(p=1.),                
        ], p=0.8),
    
        A.OneOf([    
            A.RandomGamma(gamma_limit=(100, 150), p=1.),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.3, p=1.),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=1.),
            A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=60, val_shift_limit=40, p=1.),
            A.CLAHE(clip_limit=5.0, tile_grid_size=(5, 5), p=1.),
            
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=4, p=1.),
            A.CoarseDropout(max_height=50, max_width=50, max_holes=2, p=1., fill_value=1.),
            A.RandomResizedCrop(always_apply=False, p=1.0, height=cfg.img_size, width=cfg.img_size, scale=(0.7, 1.2), ratio=(0.7, 1.3), interpolation=cv2.INTER_AREA)
        ], p=0.8),            
    ]


# In[ ]:


#https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html

# e_resize = [A.RandomResizedCrop(always_apply=False, p=1.0, height=cfg.img_size, width=cfg.img_size, scale=(0.7, 1.2), ratio=(0.75, 1.3), interpolation=1)]
# e_resize = [A.RandomResizedCrop(always_apply=False, p=1.0, height=cfg.img_size, width=cfg.img_size, scale=(0.7, 1.2), ratio=(0.7, 1.3), interpolation=cv2.INTER_AREA)]

# s_resize = [A.RandomResizedCrop(always_apply=False, p=1.0, height=cfg.img_size, width=cfg.img_size, scale=(0.7, 1.2), ratio=(0.75, 1.3), interpolation=1)]
# s_resize = [A.RandomResizedCrop(always_apply=False, p=1.0, height=cfg.img_size, width=cfg.img_size, scale=(0.7, 1.2), ratio=(0.7, 1.3), interpolation=cv2.INTER_AREA)]


# In[ ]:


# cfg.train_e_transform = A.ReplayCompose(e_resize+base_aug)
# cfg.train_s_transform = A.ReplayCompose(s_resize+base_aug)

# ReplayCompose tracks augmentation parameters.You can inspect those parameters or reapply them to another image.
# To apply the same set of augmentations to a new target, you can use the ReplayCompose.replay function.
cfg.train_e_transform = cfg.train_s_transform = A.ReplayCompose(base_aug)


# In[ ]:


cfg.val_e_transform = A.ReplayCompose([
        A.Resize(cfg.img_size, cfg.img_size, interpolation=cv2.INTER_AREA, p=1.),
    ])
cfg.val_s_transform = A.ReplayCompose([
        A.Resize(cfg.img_size, cfg.img_size, interpolation=cv2.INTER_AREA, p=1.),
    ])


# In[ ]:




