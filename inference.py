from dataset import Kitti
from transforms import *
from torch.utils.data import DataLoader

import time
import os
from net.model import PDF
from geometrics import inverse_warp, flow_warp, pose2flow, mask_gen, pose_vec2mat
from losses import *
# from tensorboardX import SummaryWriter
from tqdm import tqdm
from cfg_inference import config
from collections import defaultdict
from train import get_loader 
import cv2
from path import Path
from matplotlib import pyplot as plt


def get_time():
    T = time.strftime('%y.%m.%d-%H.%M.%S', time.localtime())
    return T.split('-')

def flow_visualize(flow):

    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
    flow_norm = (flow[:,:,0]**2+flow[:,:,1]**2)**0.5
    flow_dire = np.arctan2(flow[:,:,0], flow[:,:,1])

    max_value = np.abs(flow[:,:,:2]).max()

    channel0= ang*180/np.pi
    channel1 = cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX)
    channel2 = flow[:,:,2]
    # channel1 = channel1.clip(0,1)
    colormap=np.stack([channel0,channel1,channel2], axis=-1)
    colormap=cv2.cvtColor(np.float32(colormap),cv2.COLOR_HSV2RGB)

    return colormap


def flow_write(filename, flow):

    if flow.shape[2]==2:
        flow = np.stack([flow,np.ones_like(flow[:,:,0])], axis=-1)

    flow = np.float64(flow)
    flow[:,:,:2] = (flow[:,:,:2]*64+2**15)#.clip(0,2**16-1)
    flow = np.uint16(flow)
    flow = cv2.cvtColor(flow,cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, flow)


def post_process(img):
    img = img.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)*0.5+0.5
    img = cv2.cvtColor(img*255,cv2.COLOR_BGR2RGB).astype(np.uint8)
    return img


def inference(net, dataloader, device, cfg, save_dir):

    eps=1e-4
    net.eval()
    val_process = tqdm(enumerate(dataloader))
    
    for i, input_dict in val_process:
        img0 = input_dict['images'][0].to(device)
        img1 = input_dict['images'][1].to(device)
        intrinsics = input_dict['intrinsics'].to(device)
        intrinsics_inv = input_dict['intrinsics_inv'].to(device)
        depth_gt=input_dict['depth_gt'].numpy().squeeze()
        print('\nmax {}, min {}, mean {}'.format(depth_gt.max(), depth_gt.min(), depth_gt.mean()))
        
        depth, pose, flow = net([img0, img1])

        if depth is not None:
            depth_map = 1/(depth*cfg['depth_scale']+cfg['depth_eps'])

            img_warped = inverse_warp(img0, depth_map.squeeze_(dim=1), pose, intrinsics, intrinsics_inv)

            valid_area = 1 - (img_warped == 0).prod(1, keepdim=True).type_as(img_warped)

            flow_rigid = pose2flow(depth_map, pose, intrinsics, intrinsics_inv)
            # color_map_flow_rigid=flow_visualize(flow_rigid)

            img_warped = post_process(img_warped)
    
            f = flow_rigid.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
            f[...,0]=f[...,0]*(f.shape[0]-1)/2
            f[...,1]=f[...,1]*(f.shape[1]-1)/2
            f = np.concatenate([f,np.ones((f.shape[0],f.shape[1],1))],axis=-1)
            
            color_map=flow_visualize(f)

            depth_map = depth_map.cpu().detach().numpy().transpose(1, 2, 0)
            depth_map = depth_map*(depth_gt.mean()/depth_map.mean())
            depth_map = np.clip(depth_map,0,80)
            depth_map = depth_map.squeeze(axis=-1)
            print('max {}, min {}, mean {}'.format(depth_map.max(), depth_map.min(), depth_map.mean()))


            plt.imshow(depth_map,cmap=plt.cm.jet)
            plt.imsave(save_dir/'{}_depth_color.jpg'.format(i),depth_map)
            plt.imshow(depth_gt,cmap=plt.cm.jet)
            plt.imsave(save_dir/'{}_depth_gt_color.jpg'.format(i),depth_gt)


            forward_tgt = post_process(img1)
            forward_src = post_process(img0)

            
            cv2.imwrite(save_dir/'{}_forward_tgt.jpg'.format(i),forward_tgt)
            cv2.imwrite(save_dir/'{}_forward_src.jpg'.format(i),forward_src)            
            cv2.imwrite(save_dir/'{}_depth_gt.png'.format(i),depth_gt)
            cv2.imwrite(save_dir/'{}_depth.png'.format(i),depth_map)
            cv2.imwrite(save_dir/'{}_forward_depth_warped.jpg'.format(i), img_warped)
            cv2.imwrite(save_dir/'{}_rigid_flow_color.jpg'.format(i), color_map)
            flow_write(save_dir/('{}_rigid_flow.png'.format(i)),f)

            

        if flow is not None:
        
            forward_warped = post_process(flow_warp(img0, flow))
            cv2.imwrite(save_dir/'{}_forward_flow_warped.jpg'.format(i),forward_warped)    
        
            ## save flow to png file
            f = flow.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
            f[...,0]=f[...,0]*(f.shape[0]-1)/2
            f[...,1]=f[...,1]*(f.shape[1]-1)/2

            f = np.concatenate([f,np.ones((f.shape[0],f.shape[1],1))],axis=-1)
            color_map=flow_visualize(f)
            flow_write(save_dir/('{}_flow.png'.format(i)),f)
            cv2.imwrite(save_dir/'{}_flow_color.jpg'.format(i), color_map)


save_dir=Path(config['output_dir'])
val_loader = get_loader(**config['data']['val'])
print('Data Loaded.')
# TODO: 定义summary
# 设置随机数种子
SEED = time.time()
# 设置GPU or CPU
if config['device'] == 'gpu' and torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.manual_seed(SEED)
else:
    device = torch.device('cpu')
    torch.manual_seed(SEED)
print('Torch Device:', device)
# 定义saver

save_pth = config['save_pth']
print('load model')
# 定义模型
net = PDF(**config['model'])
net.to(device)
# 是否导入预训练
if config['pretrain']:
    net.load_state_dict(torch.load(config['pretrained_weights']),strict=False)
else:
    net.init_weights()
# 设置优化器

# 启动summary
global_steps = 0

print('Val Samples:', len(val_loader))
# 开始迭代

inference(net, val_loader, device, config['losses'], save_dir)

#     print('EP {} training loss:{:.6f} min:{:.6f}'.format(
#         epoch, train_avg_loss, min_loss), file=log)
#     print('EP {} validation loss:{:.6f} min:{:.6f}'.format(
#         epoch, eval_avg_loss, min_val_loss), file=log)
# log.close()
# # except:
# #     print('****Exception****', file=log)
# #     log.close()
# print('Training Done.')
