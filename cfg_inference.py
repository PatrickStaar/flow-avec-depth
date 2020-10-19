from path import Path
import numpy as np
from transforms import *

config=dict(
    data=dict(
        val=dict(
            root='/dataset/kitti',
            sample_list='split/eigen_full/val_no_static_lite.txt',
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
            pin_memory=False,
            interp=False
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
    max_epoch=20,
    lr = 1e-3,
    steps=100,

    # losses
    losses=dict(
        use_depth=False,
        use_flow=True,
        use_pose=False,
        use_disc=False,
        use_mask=False,
        depth_scale=10,
        depth_eps=0.1,
        weights=dict(
            reprojection_loss=1.,
            flow_consistency=1.,
            depth_smo=0.1,
            flow_smo=0.1,
            depth_loss=1,
            flow_loss=0,
            pose_loss=0,
            disc=0., # Discriminator loss, not implemented for now.
            multi_scale=[1./16,1./8,1./4,1./2,1.],
            ssim=1.,
            l1=1,
        ),
    ),

    output_dir='outputs.10.06.1',

    save_pth='./checkpoints',
    pretrain=True,
    pretrained_weights='./checkpoints/10.06.17.22.39_ep38_val.pt',
    log='./checkpoints/log',

    # test
    # test_tmp = Path('./tmp')
    # weight_for_test = save_pth/'12.16.09.25.48_ep22_val.pt'

)
