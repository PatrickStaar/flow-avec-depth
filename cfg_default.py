from path import Path
import numpy as np
from transforms import *

config=dict(
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
    ),

    device='gpu',
    eps=1e-5,

    # intrinsics
    # fixed_intrinsics=dataset_path/'cam.txt'

    # parameters
    # model
    model=dict(
        use_depth=True,
        use_flow=True,
        use_pose=True,
    ),
    # optimizer
    max_epoch=50,
    lr = 0.0001,
    steps=100,

    # losses
    losses=dict(
        use_depth=True,
        use_flow=True,
        use_pose=True,
        weights=dict(
            reprojection_loss=0.,
            flow_consistency=1.,
            depth_smo=0.,
            flow_smo=0.1,
            depth_loss=0,
            flow_loss=0,
            pose_loss=0,
            disc=0., # Discriminator loss, not implemented for now.
            multi_scale=[1.,1.,1.,1.],
            ssim=1.,
            l1=0.5,
        ),
    ),

    save_pth='./checkpoints',
    pretrain=True,
    pretrained_weights='./checkpoints/12.16.09.25.48_ep22_val.pt',
    log='./checkpoints/log',

    # test
    # test_tmp = Path('./tmp')
    # weight_for_test = save_pth/'12.16.09.25.48_ep22_val.pt'

)