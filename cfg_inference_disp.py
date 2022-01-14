from path import Path
import numpy as np
from transforms import *

config=dict(
    data=dict(
        val=dict(
            root='/dataset/KITTI',
            # sample_list='split/eigen_full/val_files_l.txt',
            sample_list='split/eigen_benchmark/test_files.txt',
            transform=Compose([
                Scale(192,640),
                ArrayToTensor(),
                Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5]),]),
            train=False,
            batch_size=1,
            seq_len=1,
            input_size=(192,640),
            with_default_intrinsics=False,
            with_depth=True, 
            with_stereo=True, 
            with_pose=False,
            shuffle=False,
            pin_memory=False,
            interp=False,
            workers=1
        )
    ),

    device='gpu',
    eps=1e-5,
    use_right=True,
    # model
    output_dir='outputs.disp.val',

    save_pth='./checkpoints',
    eval_weights='./checkpoints/01.13.14.35.10_ep11.pt',
    log='./checkpoints/log',

    # test
    # test_tmp = Path('./tmp')
    # weight_for_test = save_pth/'12.16.09.25.48_ep22_val.pt'

)
