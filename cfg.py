from path import Path
import numpy as np



# dataset
dataset_root=Path('./datasets')


dataset='kitti'
dataset_path=dataset_root/dataset
split_ratio=0.6
batch_size=16
sequence_len=2

# image
image_means=[0.5,0.5,0.5]
std = [0.5,0.5,0.5]
image_size=[]


# intrinsics


# model
pretrain=False
from_checkpoint=False
pretrained_weights=''


# optimizer
max_epoch=10



# losses



# post training

save_pth=Path('./checkpoints')