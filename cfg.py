from path import Path
import numpy as np



# dataset
dataset_root=Path('./dataset')
dataset='kitti'
dataset_path=dataset_root/dataset
batch_size=2
sequence_len=0
max_interval=1
rigid=True

# image
# unsure
mean=[0.5,0.5,0.5]
std = [0.5,0.5,0.5]
image_size=[]
eps=1e-5

# intrinsics
fixed_intrinsics=dataset_path/'cam.txt'

# parameters
save_pth=Path('./checkpoints')
log=Path('./log')

# model
pretrain=True
pretrained_weights=save_pth/'12.16.09.25.48_ep22_val.pt'

# optimizer
max_epoch=50
lr = 0.001
steps=100

# losses
loss_weight={
    'depth_consistency':1.,
    'flow_consistency':0.,
    'depth_supervise':0,
    'flow_supervise':0,
    'pose_supervise':0,
    'depth_smoothness':0.,
    'flow_smoothness':0.,
}

multi_scale_weight=[1.,1.,1.,1.]
reconstruction_weights=[1.,1.,0.1]
# post training

# test

test_tmp = Path('./tmp')
weight_for_test = save_pth/'12.16.09.25.48_ep22_val.pt'

# log

log = Path('./log')
