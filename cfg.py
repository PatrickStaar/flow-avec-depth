from path import Path
import numpy as np



# dataset
dataset_root=Path('./dataset')
dataset='tum'
dataset_path=dataset_root/dataset
split_ratio=0.6
batch_size=4
sequence_len=20

# image
# unsure
mean=[0.5,0.5,0.5]
std = [0.5,0.5,0.5]
image_size=[]


# intrinsics
fixed_intrinsics=dataset_path/'cam.txt'

# model
pretrain=False
from_checkpoint=False
pretrained_weights=''


# optimizer
max_epoch=10
lr = 0.01
steps=1

# losses
loss_weight={
    'depth_consistency':1.,
    'flow_consistency':1.,
}
multi_scale_weight=[1.,0.5,0.1,0.01]
reconstruction_weights=[1,0.5,0.1]
# post training

save_pth=Path('./checkpoints')
log=Path('./log')