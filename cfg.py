from path import Path
import numpy as np



# dataset
dataset_root=Path('./dataset')
dataset='tum'
dataset_path=dataset_root/dataset
split_ratio=0.6
batch_size=4
sequence_len=20
max_interval=3
rigid=True

# image
# unsure
mean=[0.5,0.5,0.5]
std = [0.5,0.5,0.5]
image_size=[]

# intrinsics
fixed_intrinsics=dataset_path/'cam.txt'

# parameters
save_pth=Path('./checkpoints')
log=Path('./log')

# model
pretrain=True
pretrained_weights=save_pth/'12.02.22.14.24_epoch_99.pt'

# optimizer
max_epoch=100
lr = 0.0001
steps=100

# losses
loss_weight={
    'depth_consistency':0.,
    'flow_consistency':1.,
    'depth_supervise':0,
    'flow_supervise':0,
    'pose_supervise':0,
    'smoothness':0.5,
}

multi_scale_weight=[1.,1.,1.,1.]
reconstruction_weights=[1.,1.,0.5]
# post training

# test

test_tmp = Path('./tmp')
weight_for_test = save_pth/'12.02.22.14.24_epoch_99.pt'
