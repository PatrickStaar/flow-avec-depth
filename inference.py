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


def inference(net, dataloader, device, cfg):

    eps=1e-4
    net.eval()
    val_process = tqdm(enumerate(dataloader))
    save_dir=Path('./outputs')
    for i, input_dict in val_process:
        img0 = input_dict['images'][0].to(device)
        img1 = input_dict['images'][1].to(device)
        intrinsics = input_dict['intrinsics'].to(device)
        intrinsics_inv = input_dict['intrinsics_inv'].to(device)
        
        depth, pose, flow = net([img0, img1])
        # depth = 1./(depth+eps)

        # depth1 = [1./(d[:, 1]+eps) for d in depthmap]
        # flow_backward = [-f for f in flow]

        # generate multi scale mask, including forward and backward masks
        # if not cfg.rigid:
        #     masks = multi_scale_mask(
        #         multi_scale=1, depth=(depth0, depth1),
        #         pose=pose, flow=(flow, flow_backward),
        #         intrinsics=intrinsics, intrinsics_inv=intrinsics_inv)
        # else:
        #     # mask is not needed in full rigid scenes
        #     masks = None
        # cv2.imwrite(os.path.join(save_dir,'{}_depth_pred.png'.format(i)),depth)
        # cv2.imwrite(os.path.join(save_dir,'{}_img.jpg'.format(i)),img0)
        forward_warped = flow_warp(
        img0, flow).cpu().detach().numpy().squeeze(0)
        print(torch.max(flow))

        forward_warped = forward_warped.transpose(1, 2, 0)*0.5+0.5
    
        forward_img = img1.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)*0.5+0.5
        forward_img=cv2.cvtColor(forward_img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_dir/'{}_forward_src.jpg'.format(i),
                        np.uint8(forward_img*255))
        forward_warped=cv2.cvtColor(forward_warped,cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_dir/'{}_forward.jpg'.format(i),
                        np.uint8(forward_warped*255)[:,:,])
        ## save flow to png file
        f = flow.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
        f[...,0]=f[...,0]*(f.shape[0]-1)/2
        f[...,1]=f[...,1]*(f.shape[1]-1)/2

        f = np.concatenate([f,np.ones((f.shape[0],f.shape[1],1))],axis=-1)

        # color_map=flow_visualize(f)
        flow_write(save_dir/('{}_flow.png'.format(i)),f)



        # 具体的validation loss计算的指标和输出的形式还需确定
    #     loss_per_iter = evaluate(target,pred,cfg['weights'])
    #     val_process.set_description("evaluating..., ")
    #     loss_per_validation=update(loss_per_validation,loss_per_iter)
    
    # msg=''
    # for k,v in loss_per_validation.items():
    #     msg+= '{}:{:.6f},'.format(k,v/len(dataloader))

    # # TODO: 验证集各项损失显示
    # print('>> Average Validation Loss:'+msg)



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
    net.load_state_dict(torch.load(config['pretrained_weights']))
else:
    net.init_weights()
# 设置优化器

# 启动summary
global_steps = 0

print('Val Samples:', len(val_loader))
# 开始迭代

inference(net, val_loader, device,config['losses'])

#     print('EP {} training loss:{:.6f} min:{:.6f}'.format(
#         epoch, train_avg_loss, min_loss), file=log)
#     print('EP {} validation loss:{:.6f} min:{:.6f}'.format(
#         epoch, eval_avg_loss, min_val_loss), file=log)
# log.close()
# # except:
# #     print('****Exception****', file=log)
# #     log.close()
# print('Training Done.')