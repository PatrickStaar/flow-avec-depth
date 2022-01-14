from path import Path
import numpy as np
from transforms import *

config=dict(
    data=dict(
        val=dict(
            root='/dataset/custom',
            sample_list='test_custom.txt',
            transform=Compose([
                Scale(192,640),
                ArrayToTensor(),
                Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5]),]),
            train=False,
            batch_size=1,
            seq_len=1,
            mode='jpg',
            input_size=(192,640),
            with_default_intrinsics=True,
            with_depth=False, 
            with_stereo=False, 
            with_pose=False,
            shuffle=False,
            pin_memory=False,
            interp=False,
            workers=0
        )
    ),

    device='gpu',
    output_dir='outputs.disp.custom',

    save_pth='./checkpoints',
    eval_weights='./checkpoints/01.13.02.04.31_ep122.pt',
    log='./checkpoints/log',
)