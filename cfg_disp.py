from path import Path
import numpy as np
from transforms import *

#ps -a | awk '/.*python$/ {print $1}'|while read line;do sudo renice -16 -p $line; done
#ps -a | awk '/.*python$/ {print $1}'|while read line;do sudo ionice -c 0 -p $line; done

config=dict(
    data=dict(
        train=dict(
            root='/dataset/KITTI',
            sample_list='split/eigen_full/train_files_l.txt',
            transform=Compose([
                RandomHorizontalFlip(),
                Scale(192,640),
                Color(0.2,0.2,0.2,0.08),
                ArrayToTensor(),
                Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5]),]),
            train=True,
            batch_size=32,
            seq_len=1,
            # rigid=True,
            input_size=(192,640),
            with_default_intrinsics=True,
            with_stereo=True,
            with_depth=False,
            shuffle=True,
            pin_memory=True,
            workers=16
        ),
        val=dict(
            root='/dataset/KITTI',
            sample_list='split/eigen_full/val_files_l.txt',
            transform=Compose([
                Scale(192,640),
                ArrayToTensor(),
                Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5]),]),
            train=False,
            batch_size=1,
            seq_len=1,
            input_size=(192,640),
            with_default_intrinsics=True,
            with_depth=True, 
            with_stereo=True, 
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
    max_epoch=100,
    lr = 2e-4,
    

    # losses
    losses=dict(
        use_mask=False,
        depth_scale=10,
        multi_scale=5,
        depth_eps=0.1,
        use_right=True,
        weights=dict(
            reprojection_loss=1,
            depth_smo=0.1,
            depth_smo2=0.05,
            pose_loss=0,
            multi_scale=[1,1/4,1/16,1/64],
            ssim=0.85,
            l1=0.15
        ),
    ),

    save_pth='./checkpoints',
    eval_weights='./checkpoints/best_mask.pth',
    log = './checkpoints/log',

    # test
    # test_tmp = Path('./tmp')
    # weight_for_test = save_pth/'12.16.09.25.48_ep22_val.pt'

)
