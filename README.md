## 1st-and-future-player-contact-detection
## score at 20th position is achieved.
<kbd>![nfl_submission](https://github.com/bishnarender/1st-and-future-player-contact-detection/assets/49610834/c9891c29-e4ec-4f96-8f81-00c1a487fc7a)</kbd>

Project is completed over just 1 TB hard disk. 

Data is partitioned over only 5 folds, instead of 8-10 folds even the data is huge in size (240 videos covers approx. 2.5 TB space).

Player-to-ground contact detection is trained over approx. 80 videos (out of 240 videos). 

Player-to-player contact detection is trained over approx. 180 videos (out of 240 videos).

-----

### Start 
For better understanding of project, read the files in the following order:
1. metadata.ipynb 
2. folds.ipynb
3. xgb_pre.ipynb
4. xgb_pre_not.ipynb
5. slicing_g.ipynb
6. slicing_g_next.ipynb
7. slicing_not_g.ipynb
8. train.ipynb
9. xgb_post.ipynb
10. xgb_post_not.ipynb
11. nfl-1st-inference.ipynb

-----

### metadata.ipynb 
Creates three major dictionaries.

![metadata_1](https://github.com/bishnarender/1st-and-future-player-contact-detection/assets/49610834/4cf824f9-10f5-4292-8fd7-01c32cca8bf8)

-----

### xgb_pre.ipynb
The xgb_pre.ipynb trains the xgb model for player-to-ground contact detection. At a particular step (say 25) features are [s, dis, dir, o, a, sa, x, y], total eight features. Then at this particular step 392 more features are created by wondering over these eight features over two windows [-25,-1] and [1,24]. That means generating these eight features over 25 steps prior and 24 steps later from the given step (say 25). Thus, forming a total of 400 features. The first feature of these 400 is defined as (for particular step 25):

<p align="center">s_-25 = ['s' value at step 25] - ['s' value at step 0 i.e. 25-25]</p>

-----

### xgb_pre_not.ipynb
The xgb_pre_not.ipynb trains the xgb model for player-to-player contact detection. At a particular step (say 25) features are [s, dis, dir, o, a, sa, x, y, team], total nine features. Nine features are for both players and thus total 18 features are present. Then at this particular step (say 25) some more features are created using above 18 and some from helmets bbox data. Thus, forming a total of 200 features.

-----

### slicing_g.ipynb
The endzone and sideline videos are processed for player-to-ground detection. Images are extracted from the frame and its neighbor frames [-54, -48, -42, -36, -30, -24, -18, -13, -8, -4, -2, 0, 2, 4, 8, 13, 18, 24, 30, 36, 42, 48, 54]. This sampling technique enables the model to observe more frames close to the estimated frame. The decision on these sampling frames was based on experiments. Further, each frame is cropped into “256 x 256 x 3” image.

In frames, player’s head is masked with a white circle to guide the model's attention to the relevant player (player in contact_id). Head position is obtained from the helmet bounding box data.

![slicing_g](https://github.com/bishnarender/1st-and-future-player-contact-detection/assets/49610834/2c5b2f93-c663-4885-b388-7eb3aec02356)

-----

### slicing_g_next.ipynb
The file is a copy of slicing_g.ipynb but with a minute difference. The difference is that instead of a 3-channel RGB image, a gray scale image is used.  

The gray scale image is placed at center and leftover 2 channels are filled with 2 frame prior image and 2 frame later image. Thus, finally forming a 3-channel image.

The head position is masked over 3-channel by considering the center-channel image (target frame).

![slicing_g_next](https://github.com/bishnarender/1st-and-future-player-contact-detection/assets/49610834/2e07e216-d384-4f43-bc00-861caa41fcf6)

-----

### slicing_not_g.ipynb
The endzone and sideline videos are processed for player-to-player detection. Images are extracted from the frame and its neighbor frames [-38, -31, -24, -18, -12, -7, -2, 2, 4, 6, 8, 10, 14, 19, 24, 30, 36, 43]. This sampling technique enables the model to observe more frames close to the estimated frame. The decision on these sampling frames was based on experiments. Further, each frame is cropped into “256 x 256 x 3” image.
In frames, 1st player’s head is masked with a “white circle” and 2nd with “black circle” to guide the model's attention to the relevant players (players in contact_id). Head position is obtained from the helmet bounding box data.

![slicing_not_g](https://github.com/bishnarender/1st-and-future-player-contact-detection/assets/49610834/62709945-38d8-4adb-898e-9a6369d904e1)

-----

### train.ipynb
This problem appears to resemble an action classification task rather than a standard 3D classification.  The “”mmaction2”” repository is best for this type of action classification task. ResNet50 using CSN technique performed well in action detection task.

![resnet3d](https://github.com/bishnarender/1st-and-future-player-contact-detection/assets/49610834/623dc272-9116-4f06-8630-c0a0481ab4d2)

CSN (Channel-Separated Convolutional Networks). In a standard CNN, convolutional filters operate on all input channels simultaneously. However, in CSNs, the channels are split into groups, and each group is convolved with a different set of filters. This separation of channels allows the network to learn specialized filters for different groups of channels, enabling more diverse and effective feature extraction. <b>Channel separation in CSN regularizes the model and prevents overfitting</b>.

![csn](https://github.com/bishnarender/1st-and-future-player-contact-detection/assets/49610834/e01fb0ad-05b1-46fb-913a-be4ae1ca1578)

Channel-separated networks use group convolution as their main building block. Depthwise convolution is the extreme version of group convolution where the number of groups is equal to the number of input and output channels. 

![model](https://github.com/bishnarender/1st-and-future-player-contact-detection/assets/49610834/63b7d70b-cc4e-42d1-91ed-ce973921e6cf)

The input [BS, 3, 23, 256, 512] in case of player-to-ground, is defined as:

3 -> denotes the number of channels, 23 -> denotes the number of frames/images, 256 -> image height, 512 -> “256+256” combined width of sideline and endzone image.

The input [BS, 3, 23, 256, 640] in case of player-to-player, is defined as:

3 -> denotes the number of channels, 23 -> denotes the number of frames/images, 256 -> image height, 640 -> “256+128+256” combined width of sideline, simulated and endzone image. The simulated image contains position of all players i.e., inclusive of players which are not seen in frame. The positions are shown via the tracking data. Two different colors are used to represent the two teams. “”Players in contact”” are depicted with bigger and brighter circles (radius is 5, and pixel value is 255), while background players are depicted with smaller and darker circles (radius is 3, and pixel value is 125).

![3](https://github.com/bishnarender/1st-and-future-player-contact-detection/assets/49610834/ab5bf626-3d2f-4324-a73e-57692455f2ea)

-----

### xgb_post.ipynb
As training is over limited videos (for player-to-ground) so used only 3 checkpoints (for inference) corresponding to 3 folds.  Simple xgb model is trained over the results of xgb_pre and cnn. Probabilities from the 20 neighboring steps are used as features i.e., {prob(-10), prob(-9), …, prob(0), prob(1), …, prob(9)}, where prob(-10) represents the probability of the same player in the 10 steps prior.

-----

### xgb_post_not.ipynb
Simple xgb model is trained over the results of xgb_pre_not and cnn (for the case of player-to-player contact detection). Ensemble probabilities from the 20 neighboring steps are used as features i.e., {prob(-10), prob(-9), …, prob(0), prob(1), …, prob(9)}, where prob(-10) represents the probability of the same pair of players in the 10 steps prior.
