#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from types import SimpleNamespace
import albumentations


# In[ ]:


cfg = SimpleNamespace(**{})


# In[ ]:


cfg.debug = 0
cfg.train_csv_path = "train.csv"
cfg.batch_size = 64
# cfg.out_dir = f"output/{os.path.basename(__file__).split('.')[0]}"
cfg.out_dir = f"output/base"
cfg.seed = 42
cfg.img_size = 256
cfg.epochs = 20
cfg.mode = 'train'
cfg.model_name = 'tf_efficientnet_b0_ns'
cfg.lr = 1e-3
cfg.scheduler = "linear" #linear  cosine, step cosinewarmup warmupv2
cfg.optimizer = "Adam"  # Adam, SGD, AdamW
cfg.weight_decay = 0  # weigth_decay for Adam, SGD, AdamW
cfg.num_workers = 0
cfg.folds = [0,1,2,3,4]
cfg.model = 'model1'
cfg.dataset = 'dataset1'
cfg.loss_fn  = 'bce'
cfg.load_weight = ''
cfg.apex=1 # enable or disable gradient scaling.
cfg.use_meta = 0
cfg.frac = 0.25
cfg.val_frac = 1
cfg.pos_frac = 1
cfg.pool_type = 'avg'
cfg.trk_type = 1
cfg.num_freeze = 0
cfg.hoge = 0
cfg.use_swa = 0
cfg.warmup = 500 
cfg.is_G = 0
cfg.sampler = 0
cfg.skip_frame = 0
cfg.use_oof = 0 # dependent on cfg.pl_path


# In[ ]:


cfg.train_transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Rotate( p=0.5, limit=(-90, 90), interpolation=1, mask_value=None),

        # albumentations.Normalize(mean=(0.2, 0.2, 0.2), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=0.5),
    ])
cfg.val_transform = albumentations.Compose([
        # albumentations.Resize(cfg.img_size, cfg.img_size, interpolation=1, p=1),
        # albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])


# In[ ]:


basic_cfg = cfg

