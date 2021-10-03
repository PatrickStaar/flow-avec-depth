import torch
import torch.nn.functional as F
from torch import nn
from ssim import SSIM
from geometrics import flow_warp, inverse_warp, pose2flow, mask_gen
from collections import defaultdict
import numpy as np
import cv2


get_ssim=SSIM().cuda()
eps=1e-3
MIN_DEPTH=1e-3
MAX_DEPTH=80

def gradient(pred):
    dy = torch.abs(pred[..., :-1, :] - pred[..., 1:, :])
    dx = torch.abs(pred[..., :, -1:] - pred[..., :, 1:])
    return dx, dy


def upsample(src, shape):
    return F.interpolate(src, shape, mode="bilinear",align_corners=False)
    

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


def summerize(pred, target, cfg):

    loss_dict = defaultdict(float)
    weights = cfg['weights']
    B, _, H, W = target['img_src'].size()

    for scale in range(len(weights['multi_scale'])):

        scale_weight = weights['multi_scale'][scale]

        loss_dict['reprojection_loss'] += loss_reconstruction(
            pred['depth_warped'][scale],
            target['img_tgt'],
            mask=pred['mask'],
            weights=weights)*scale_weight

        loss_dict['depth_smo'] += loss_smo_edge_aware(
            pred['depth_map'][scale], target['img_src'])*scale_weight
            
    for k in weights.keys():
        if k in loss_dict.keys():
            loss_dict['loss_G'] += loss_dict[k]*weights[k]

    return loss_dict


def loss_reconstruction(img_tgt, img_warped, weights, mask=None):
    
    valid_area = 1 - (img_warped == 0).prod(1, keepdim=True).type_as(img_warped)
    penalty=valid_area.nelement()/(valid_area.sum()+1.)

    l1=torch.abs(img_warped-img_tgt)
    ssim_map = get_ssim(img_warped, img_tgt)
    loss_map=(ssim_map * weights['ssim'] + l1 * weights['l1'])*valid_area+0.01*penalty
    
    return loss_map.mean()


def loss_reprojection(rigid_flow, img_src, img_tgt, weights, mask=None,):
    # B,_, H, W = depth.size()
    # depth = torch.squeeze(depth, dim=1)
    img_tgt_warped = flow_warp(img_src, rigid_flow)
    return loss_reconstruction(img_tgt, img_tgt_warped, weights, mask)


def loss_smo_edge_aware(tgt, img):
    depth_normal=tgt.sub(tgt.mean()).div(tgt.std()+1e-5)
    grad_target_x, grad_target_y = gradient(depth_normal)
    grad_image_x, grad_image_y = gradient(img*128)
    grad_image_x = torch.mean(grad_image_x, 1, keepdim=True)
    grad_image_y = torch.mean(grad_image_y, 1, keepdim=True)
    loss_smo_x = torch.exp(-grad_image_x)*grad_target_x
    loss_smo_y = torch.exp(-grad_image_y)*grad_target_y
    return loss_smo_x.mean()+loss_smo_y.mean()


def loss_smo(tgt,img):
    # B, _, H, W = tgt.size()
    dx, dy = gradient(tgt)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    return dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()


def loss_disc(pred, gt):
    bce_loss = nn.BCELoss()
    return bce_loss(pred,gt)

# supervised

def evaluate_depth(gt, pred):
    gt = gt['depth_gt'].numpy().squeeze()
    pred = pred['depthmap'].cpu().detach().numpy().transpose(1, 2, 0)
    pred= np.clip(pred, MIN_DEPTH, MAX_DEPTH).squeeze(axis=-1)

    valid_area = np.logical_and(
        gt > MIN_DEPTH, gt < MAX_DEPTH)
    gt_height, gt_width = gt.shape
    pred = cv2.resize(pred,(gt_width,gt_height))

    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                    0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(valid_area.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    valid_area = np.logical_and(valid_area, crop_mask)

    pred_depth_vector =pred[valid_area]
    gt_depth_vector = gt[valid_area]
    # print(f'gt均值：{gt_depth_vector.mean()}')


    median_ratio = np.median(gt_depth_vector) / \
        np.median(pred_depth_vector)
    pred_depth_vector *= median_ratio
    return compute_errors(gt_depth_vector,pred_depth_vector)


def loss_pose(gt, pred):         # [B,6]
    return F.l1_loss(pred,gt)

