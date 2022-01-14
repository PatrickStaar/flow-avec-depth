import time
import os
import cv2

from net import Disp
from geometrics import flow_warp, inverse_warp
from losses import *
from tqdm import tqdm
from cfg_custom import config
from collections import defaultdict
from train_disp import get_loader
from path import Path
from matplotlib import pyplot as plt
from transforms import *

MIN_DEPTH = 1e-3
MAX_DEPTH = 80
BASELINE = 0.54
FOCAL_LEN = 721.54
eps = 1e-7

def get_time():
    T = time.strftime('%y.%m.%d-%H.%M.%S', time.localtime())
    return T.split('-')


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    error_dict = {}
    thresh = np.maximum((gt / pred), (pred / gt))
    error_dict['a1'] = (thresh < 1.25).mean()
    error_dict['a2'] = (thresh < 1.25 ** 2).mean()
    error_dict['a3'] = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    error_dict['rmse'] = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    error_dict['rmse_log'] = np.sqrt(rmse_log.mean())

    error_dict['abs_rel'] = np.mean(np.abs(gt - pred) / gt)

    error_dict['sq_rel'] = np.mean(((gt - pred) ** 2) / gt)

    return error_dict



def post_process(img):
    img = img.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)*0.5+0.5
    img = cv2.cvtColor(img*255, cv2.COLOR_BGR2RGB).astype(np.uint8)
    return img


@torch.no_grad()
def run_test(net, dataloader, device, save_dir):

    
    net.eval()
    val_process = tqdm(enumerate(dataloader))
    errors = defaultdict(float)
    # ratios = []
    for i, input_dict in val_process:
        img0 = input_dict['images'][0].to(device)
        # img1 = input_dict['stereo'][0].to(device)
        # intrinsics = input_dict['intrinsics'].to(device)
        # intrinsics_inv = input_dict['intrinsics_inv'].to(device)
        # depth_gt = input_dict['depth_gt'].numpy().squeeze()
        # gt_height, gt_width = depth_gt.shape

        disp = net(img0)[0]/2
        # img_warped = flow_warp(
        #     img1, torch.cat([-disp, torch.zeros_like(disp)], dim=1)
        # )
        # img_warped = post_process(img_warped)

        disp = disp.squeeze_(0).squeeze_(0).cpu().numpy()
        # disp = cv2.resize(disp, (gt_width, gt_height))

        # depth_map = BASELINE*FOCAL_LEN/(disp*gt_width+eps)
        # depth_map = np.clip(depth_map, MIN_DEPTH, MAX_DEPTH)

        # crop = np.array([
        #     0.40810811 * gt_height, 0.99189189 * gt_height,
        #     0.03594771 * gt_width,  0.96405229 * gt_width
        # ]).astype(np.int32)
        # # visual_crop=depth_map[int(0.4*(depth_map.shape[0])):,int(0.03*(depth_map.shape[1])):int(0.96*(depth_map.shape[1]))]
        # depth_map=cv2.resize(visual_crop,depth_map.shape[::-1])
        # valid_area = np.logical_and(
        #     depth_gt > MIN_DEPTH, depth_gt < MAX_DEPTH)
        # crop_mask = np.zeros(valid_area.shape)
        # crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        # valid_area = np.logical_and(valid_area, crop_mask)

        # pred_depth_vector = depth_map[valid_area]
        # gt_depth_vector = depth_gt[valid_area]

        # median_ratio = np.median(gt_depth_vector) / \
        #     np.median(pred_depth_vector)
        # ratios.append(median_ratio)

        # pred_depth_vector *= median_ratio

        # print('\npredict median: {}'.format(np.median(pred_depth_vector),np.median(gt_depth_vector)))
        # print('gt median: {}'.format(np.median(gt_depth_vector)))

        # error_dict = compute_errors(gt_depth_vector, pred_depth_vector)
        # for k, v in error_dict.items():
            # errors[k] += v

        plt.imsave(save_dir/'{}_depth_color.jpg'.format(i),
                   disp, cmap=plt.cm.magma)
        # depth_gt[depth_gt == 0] = 2000
        # plt.imsave(save_dir/'{}_depth_gt_color.jpg'.format(i), 10/(depth_gt),cmap=plt.cm.magma)

        left = post_process(img0)
        # right = post_process(img1)

        cv2.imwrite(save_dir/'{}_left.jpg'.format(i), left)
        # cv2.imwrite(save_dir/'{}_right.jpg'.format(i), right)
        # cv2.imwrite(save_dir/'{}_depth_gt.png'.format(i), depth_gt)
        # cv2.imwrite(save_dir/'{}_depth.png'.format(i), depth_map*median_ratio)
        # cv2.imwrite(
            # save_dir/'{}_left_warped.jpg'.format(i), img_warped)

    # np.save('median_ratio.npy',np.array(ratios))
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

net = Disp()
net.to(device)
net.load_state_dict(torch.load(config['eval_weights']), strict=False)
# 启动summary
global_steps = 0

print('Val Samples:', len(val_loader))
# 开始迭代

run_test(net, val_loader, device, save_dir)
