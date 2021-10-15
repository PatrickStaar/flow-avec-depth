import time
import os
import cv2

from net.model import Model
from geometrics import inverse_warp
from losses import *
from tqdm import tqdm
from cfg_inference import config
from collections import defaultdict
from train import get_loader
from path import Path
from matplotlib import pyplot as plt
from transforms import *

MIN_DEPTH = 1e-3
MAX_DEPTH = 80


def get_time():
    T = time.strftime('%y.%m.%d-%H.%M.%S', time.localtime())
    return T.split('-')


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    error_dict={}
    thresh = np.maximum((gt / pred), (pred / gt))
    error_dict['a1'] = (thresh < 1.25     ).mean()
    error_dict['a2'] = (thresh < 1.25 ** 2).mean()
    error_dict['a3'] = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    error_dict['rmse'] = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    error_dict['rmse_log'] = np.sqrt(rmse_log.mean())

    error_dict['abs_rel'] = np.mean(np.abs(gt - pred) / gt)

    error_dict['sq_rel'] = np.mean(((gt - pred) ** 2) / gt)
    
    return error_dict


def flow_visualize(flow):

    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    flow_norm = (flow[:, :, 0]**2+flow[:, :, 1]**2)**0.5
    flow_dire = np.arctan2(flow[:, :, 0], flow[:, :, 1])

    max_value = np.abs(flow[:, :, :2]).max()

    channel0 = ang*180/np.pi
    channel1 = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    channel2 = flow[:, :, 2]
    # channel1 = channel1.clip(0,1)
    colormap = np.stack([channel0, channel1, channel2], axis=-1)
    colormap = cv2.cvtColor(np.float32(colormap), cv2.COLOR_HSV2RGB)

    return colormap


def flow_write(filename, flow):

    if flow.shape[2] == 2:
        flow = np.stack([flow, np.ones_like(flow[:, :, 0])], axis=-1)

    flow = np.float64(flow)
    flow[:, :, :2] = (flow[:, :, :2]*64+2**15)  # .clip(0,2**16-1)
    flow = np.uint16(flow)
    flow = cv2.cvtColor(flow, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, flow)


def post_process(img):
    img = img.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)*0.5+0.5
    img = cv2.cvtColor(img*255, cv2.COLOR_BGR2RGB).astype(np.uint8)
    return img


def inference(net, dataloader, device, cfg, save_dir):

    eps = 1e-4
    net.eval()
    val_process = tqdm(enumerate(dataloader))
    errors=defaultdict(float)

    for i, input_dict in val_process:
        img0 = input_dict['images'][0].to(device)
        img1 = input_dict['images'][1].to(device)
        intrinsics = input_dict['intrinsics'].to(device)
        intrinsics_inv = input_dict['intrinsics_inv'].to(device)
        depth_gt = input_dict['depth_gt']  # .numpy().squeeze()

        depth, pose = net([img0, img1])
        flow =None
        if depth is not None:
            depth_map = 1/(depth*cfg['depth_scale']+cfg['depth_eps'])

            img_warped = inverse_warp(img0, depth_map.squeeze_(
                dim=1), pose, intrinsics, intrinsics_inv)
            img_warped = post_process(img_warped)

            # valid_area = 1 - (img_warped == 0).prod(1, keepdim=True).type_as(img_warped)
            # flow_rigid = pose2flow(depth_map, pose, intrinsics, intrinsics_inv)
            # color_map_flow_rigid=flow_visualize(flow_rigid.detach().cpu().numpy())
            # f = flow_rigid.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
            # f[..., 0] = f[..., 0]*(f.shape[0]-1)/2
            # f[..., 1] = f[..., 1]*(f.shape[1]-1)/2
            # f = np.concatenate(
                # [f, np.ones((f.shape[0], f.shape[1], 1))], axis=-1)

            # color_map = flow_visualize(f)

            depth_map = depth_map.cpu().detach().numpy().transpose(1, 2, 0)
            depth_map = np.clip(depth_map, MIN_DEPTH,
                                MAX_DEPTH).squeeze(axis=-1)
            depth_gt = depth_gt.numpy().squeeze()
            

            valid_area = np.logical_and(
                depth_gt > MIN_DEPTH, depth_gt < MAX_DEPTH)
            gt_height, gt_width = depth_gt.shape

            depth_map2 = cv2.resize(depth_map,(gt_width,gt_height))

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            # visual_crop=depth_map[int(0.4*(depth_map.shape[0])):,int(0.03*(depth_map.shape[1])):int(0.96*(depth_map.shape[1]))]
            # depth_map=cv2.resize(visual_crop,depth_map.shape[::-1])

            crop_mask = np.zeros(valid_area.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            valid_area = np.logical_and(valid_area, crop_mask)

            pred_depth_vector = depth_map2[valid_area]
            gt_depth_vector = depth_gt[valid_area]
            # print(f'gt均值：{gt_depth_vector.mean()}')


            median_ratio = np.median(gt_depth_vector) / \
                np.median(pred_depth_vector)
            pred_depth_vector *= median_ratio

            # print('\npredict median: {}'.format(np.median(pred_depth_vector),np.median(gt_depth_vector)))
            # print('gt median: {}'.format(np.median(gt_depth_vector)))

            error_dict=compute_errors(gt_depth_vector,pred_depth_vector)
            for k,v in error_dict.items():
                errors[k]+=v

            plt.imshow(depth_map, cmap=plt.cm.jet)
            plt.imsave(save_dir/'{}_depth_color.jpg'.format(i), 10/(depth_map*median_ratio))
            plt.imshow(depth_gt, cmap=plt.cm.jet)
            # plt.imsave(save_dir/'{}_depth_gt_color.jpg'.format(i), depth_gt)

            forward_tgt = post_process(img1)
            forward_src = post_process(img0)

            cv2.imwrite(save_dir/'{}_forward_tgt.jpg'.format(i), forward_tgt)
            cv2.imwrite(save_dir/'{}_forward_src.jpg'.format(i), forward_src)
            # cv2.imwrite(save_dir/'{}_depth_gt.png'.format(i), depth_gt)
            # cv2.imwrite(save_dir/'{}_depth.png'.format(i), depth_map*median_ratio)
            cv2.imwrite(
                save_dir/'{}_forward_warped_by_depth.jpg'.format(i), img_warped)
            # cv2.imwrite(save_dir/'{}_rigid_flow_color.jpg'.format(i), color_map)
            # plt.imsave(save_dir/'{}_flow_rigid_color.jpg'.format(i), color_map)

            # flow_write(save_dir/('{}_flow_rigid.png'.format(i)), f)

        if flow is not None:

            forward_warped = post_process(flow_warp(img0, flow))
            cv2.imwrite(
                save_dir/'{}_forward_warped_by_flow.jpg'.format(i), forward_warped)

            ## save flow to png file
            f = flow.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
            f[..., 0] = f[..., 0]*(f.shape[0]-1)/2
            f[..., 1] = f[..., 1]*(f.shape[1]-1)/2

            f = np.concatenate(
                [f, np.ones((f.shape[0], f.shape[1], 1))], axis=-1)
            color_map = flow_visualize(f)
            # flow_write(save_dir/('{}_flow.png'.format(i)), f)

            plt.imsave(save_dir/'{}_flow_color.jpg'.format(i), color_map)
            
    for k in errors.keys():
        print(f'{k}: {errors[k]/len(dataloader)}')


save_dir = Path(config['output_dir'])
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

val_loader = get_loader(**config['data']['val'])
print('Data Loaded.')
# 设置随机数种子
SEED = 100000
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

net = Model()
net.to(device)
net.load_state_dict(torch.load(config['eval_weights']), strict=False)
# 启动summary
global_steps = 0

print('Val Samples:', len(val_loader))
# 开始迭代

inference(net, val_loader, device, config['losses'], save_dir)