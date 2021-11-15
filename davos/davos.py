import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from path import Path
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from .utils import *
from .model import EncoderHrnet, DecoderHrnet
from dataset import Kitti
from .cfg_inference import config as cfg_davos

class Mask:
    cmap={'red':2,'green':1,'blue':0}
    PWC=get_pwc()

    def __init__(self) -> None:
        if cfg_davos.train.device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.manual_seed(8848)
        else:
            self.device = torch.device('cpu')
            torch.manual_seed(8848)

        self.encoder = EncoderHrnet(
            cfg_davos.hrnet,
            fusion=cfg_davos.train.etc['fusion'],
        ).to(self.device)

        self.decoder=DecoderHrnet(
            cfg_davos.hrnet,
            input_channels=720,
            output_shape=None # (384,384)
        ).to(self.device)

        self.encoder.load_state_dict(torch.load(cfg_davos.val.pretrain[0]))
        self.decoder.load_state_dict(torch.load(cfg_davos.val.pretrain[1]))
        self.encoder.eval()
        self.decoder.eval()
    
        self.threshold=cfg_davos.val.threshold
        self.flow_size=cfg_davos.val.flow_size

    def get(self, inputs):
        
        img0, img1 = inputs
        _,_,H, W=img0.shape
        _size=int(round(H/32.0))*32,int(round(W/32.0))*32

        # img0,img1=interp(img0,size=(H,W)), interp(img1,size=(H,W))
        _img0,_img1=interp(img0,size=self.flow_size), interp(img1,size=self.flow_size)

        try:
            flow=interp(
                Mask.PWC(torch.cat([_img0,_img1],1)),
                size=_size,
            )
        except:
            print(img0.shape,img1.shape)
            exit(0)

        mask = self.decoder(
            self.encoder([normalize(interp(img0.clone(),size=_size)), normalize(flow.clone())])[0])
        mask=interp(torch.sigmoid(mask).detach_(),size=(H,W))

        # limit mask range
        mask[mask<0.5]=0
        mask[...,0:W//3]*=0
        mask[...,W//3*2:]*=0

        return 1-mask
    
   
            
    def inference(self, cfg_global):

        val_loader=DataLoader(
            Kitti(**cfg_global.get('data').get('val')),
            batch_size=cfg_davos.val.loader.batch_size,
            shuffle=cfg_davos.val.loader.shuffle,
            num_workers=cfg_davos.val.loader.workers
        )

        print('Val Samples:', len(val_loader))   
        
        val_process = tqdm(enumerate(val_loader))

        for i, input_dict in val_process:
            _,_,H, W=input_dict['images'][0].shape
            img0 = input_dict['images'][0].to(self.device)
            img1 = input_dict['images'][1].to(self.device)
            # mask_gt=input_dict['gt'].squeeze_(0).squeeze_(0).numpy()
            # H,W=mask_gt.shape
            
            # real_id=input_dict.get('real_id', 0) 
            # if not isinstance(real_id,int):
                # real_id=int(real_id.squeeze_(0).item())

            # scene = num2scene[str(int(input_dict['scene'].data))]
            # seq=input_dict['seq'].squeeze_(0).item()
            # seq_str=f'{scene}.{int(seq)}'
            
            # if scene not in score_by_scene.keys():
                # score_by_scene[scene]=defaultdict(float)

            # for forward prop
            _size=int(round(H/32.0))*32,int(round(W/32.0))*32
            img0,img1=interp(img0,size=(H,W)), interp(img1,size=(H,W))
            # for flow produce
            _img0,_img1=interp(img0,size=self.flow_size), interp(img1,size=self.flow_size)

            # produce flow then resize to forward size
            try:
                flow=interp(
                    Mask.PWC(torch.cat([_img0,_img1],1)),
                    size=_size,
                )
            except:
                print(img0.shape,img1.shape)
                exit(0)

            mask = self.decoder(
                self.encoder([normalize(interp(img0.clone(),size=_size)), normalize(flow.clone())])[0])
            mask=interp(torch.sigmoid(mask).detach_(),size=(H,W))
            
            mask=mask.squeeze_(0).squeeze_(0).cpu().numpy()
            img0,img1=to_numpy(img0), to_numpy(img1)
                    
            if cfg_davos.val.output is not None:

                visual_dir=Path(cfg_davos.val.output+'.visual')
                if not os.path.exists(visual_dir):
                    os.makedirs(visual_dir)
                
                mask[mask>self.threshold]=1.0
                mask[mask<=self.threshold]=0.0
                color=Mask.cmap[cfg_davos.val.color]

                colored=img0.copy().astype(np.float32)
                colored-=(colored*np.expand_dims(mask,-1)*0.5)
                colored=colored.astype(np.int)

                colored[...,color]+=((mask/2*255).astype(np.int))
                colored[colored>255]=255
                cv2.imwrite(visual_dir/'{}_src_color.jpg'.format(i), colored.astype(np.uint8))
                #cv2.imwrite(save_dir/'{}_{}_msk_gt.png'.format(scene,i),(mask_gt*255).astype(np.uint8))
                # cv2.imwrite(save_dir/scene/'{:06d}.png'.format(real_id),(mask*255).astype(np.uint8))
                # f = flow.detach_().cpu().numpy().squeeze(0).transpose(1, 2, 0)
                #     f[..., 0] = f[..., 0]*(f.shape[0]-1)/2
                #     f[..., 1] = f[..., 1]*(f.shape[1]-1)/2
                # f = np.concatenate(
                #    [f, np.ones((f.shape[0], f.shape[1], 1))], axis=-1)
                # color_map = flow_visualize(f)
                # plt.imsave(visual_dir/'{}_flow.jpg'.format(i), color_map)

