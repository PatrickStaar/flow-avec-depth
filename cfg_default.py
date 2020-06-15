from path import Path
import numpy as np
from transforms import *

config=dict(
    data=dict(
        train=dict(
            root='/dataset/kitti',
            sample_list='split/eigen_full/train_mod.txt',
            transform=Compose([
                Scale(192,640),
                RandomHorizontalFlip(),
                ArrayToTensor(),
                Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5]),]),
            train=True,
            batch_size=4,
            sequence=(-1,0),
            # rigid=True,
            input_size=(192,640),
            with_default_intrinsics=None,
            shuffle=True,
            pin_memory=True,
        ),
        val=dict(
            root='/dataset/kitti',
            sample_list='split/eigen_full/val_mod.txt',
            transform=Compose([
                Scale(192,640),
                ArrayToTensor(),
                Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5]),]),
            train=False,
            batch_size=1,
            sequence=(-1,0),
            input_size=(192,640),
            with_default_intrinsics=None,
            with_depth=True, 
            with_flow=False, 
            with_pose=False,
            shuffle=False,
            pin_memory=True,
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
        pretrain_encoder='pretrain/resnet50_mod.pth',
        
    ),
    # optimizer
    max_epoch=50,
    lr = 1e-4,
    steps=100,

    # losses
    losses=dict(
        use_depth=True,
        use_flow=True,
        use_pose=False,
        use_disc=False,
        use_mask=True,
        depth_scale=10,
        depth_eps=0.1,
        weights=dict(
            reprojection_loss=1,
            flow_consistency=1,
            depth_smo=0.1,
            # mask_loss=1,
            flow_smo=0.1,
            depth_loss=1,
            flow_loss=0,
            pose_loss=0,
            disc=0, # Discriminator loss, not implemented for now.
            multi_scale=[1, 1/4, 1/8, 1/16, 1/32],
            ssim=0.75,
            l1=0.25
        ),
    ),

    save_pth='./checkpoints',
    pretrain=False,
    pretrained_weights='./checkpoints/04.22.20.50.04_ep25.pt',
    log = './checkpoints/log',

    # test
    # test_tmp = Path('./tmp')
    # weight_for_test = save_pth/'12.16.09.25.48_ep22_val.pt'

)
