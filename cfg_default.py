from path import Path
import numpy as np
from transforms import *

#ps -a | awk '/.*python$/ {print $1}'|while read line;do sudo renice -16 -p $line; done
#ps -a | awk '/.*python$/ {print $1}'|while read line;do sudo ionice -c 0 -p $line; done

config=dict(
    data=dict(
        train=dict(
            root='/dataset/KITTI',
            sample_list='split/eigen_full/train_no_static.txt',
            transform=Compose([
                RandomHorizontalFlip(),
                Scale(192,640),
                Color(0.2,0.2,0.2,0.08),
                ArrayToTensor(),
                Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5]),]),
            train=True,
            batch_size=36,
            sequence=(-1,0),
            # rigid=True,
            input_size=(192,640),
            with_default_intrinsics=None,
            shuffle=True,
            pin_memory=False,
            workers=12
        ),
        val=dict(
            root='/dataset/KITTI',
            sample_list='split/eigen_full/val_no_static.txt',
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
            workers=12
        )
    ),

    device='gpu',
    eps=1e-5,

    # intrinsics
    # fixed_intrinsics=dataset_path/'cam.txt'

    # parameters
    # model
    pretrained_weights=dict(
        depth='pretrain/resnet50.pth',
        pose='pretrain/resnet18.pth',
    ),
    # optimizer
    max_epoch=150,
    lr = 1e-4,

    # losses
    losses=dict(
        use_mask=False,
        depth_scale=10,
        multi_scale=5,
        depth_eps=0.1,
        weights=dict(
            reprojection_loss=1,
            depth_smo=0.01,
            depth_loss=1,
            pose_loss=0,
            multi_scale=[1,1/4,1/16,1/64],
            ssim=0.5,
            l1=0.5
        ),
    ),

    save_pth='./checkpoints',
    eval_weights='./checkpoints/best_smooth_1.pth',
    log = './checkpoints/log',

    # test
    # test_tmp = Path('./tmp')
    # weight_for_test = save_pth/'12.16.09.25.48_ep22_val.pt'

)
