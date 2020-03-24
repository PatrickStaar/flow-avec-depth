from path import Path
import numpy as np
from transforms import *


# dataset
# dataset_root=Path('./dataset')
# dataset='kitti'
# dataset_path=dataset_root/dataset
# batch_size=2
# sequence_len=0
# max_interval=1
# rigid=True

data=dict(
    train=dict(
        root='./dataset/kitti',
        sample_list='eigen.txt',
        transform=Compose([
            Scale(384,1280),
            RandomHorizontalFlip(),
            ArrayToTensor(),
            Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5]),]),
        train=True,
        batch_size=4,
        sequence=[-1,0],
        # rigid=True,
        input_size=(640,192),
        intrinsics=None,
        shuffle=True,
        pin_memory=True,
    ),
    val=dict(
        root='./dataset/kitti',
        sample_list='eigen_val.txt',
        val_cfg=Compose([
            Scale(384,1280),
            ArrayToTensor(),
            Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5]),]),
        train=False,
        batch_size=1,
        sequence=[-1,0],
        input_size=(640,192),
        intrinsics=None,
        shuffle=False,
        pin_memory=False,
    )
)


device='gpu'
eps=1e-5

# transform=dict(
#     train_cfg=Compose([
#         Scale(384,1280),
#         RandomHorizontalFlip(),
#         ArrayToTensor(),
#         Normalize(mean=cfg.mean, std=cfg.std),]),
#     val_cfg=Compose([
#         Scale(384,1280),
#         ArrayToTensor(),
#         Normalize(mean=cfg.mean, std=cfg.std),])
# )

# intrinsics
fixed_intrinsics=dataset_path/'cam.txt'

# parameters


# model
model_cfg=dict(
    use_depth=True,
    use_flow=True,
    use_pose=True,
)
save_pth=Path('./checkpoints')

pretrain=True
pretrained_weights=save_pth/'12.16.09.25.48_ep22_val.pt'

# optimizer
max_epoch=50
lr = 0.0001
steps=100

# losses
loss_weight={
    'depth_consistency':0.,
    'flow_consistency':1.,
    'depth_supervise':0,
    'flow_supervise':0,
    'pose_supervise':0,
    'depth_smoothness':0.,
    'flow_smoothness':0.1,
}

multi_scale_weight=[1.,1.,1.,1.]
reconstruction_weights=[1.,1.,0.1]
# post training

# test

test_tmp = Path('./tmp')
weight_for_test = save_pth/'12.16.09.25.48_ep22_val.pt'

# log

log = Path('./log')

log=Path('./log')