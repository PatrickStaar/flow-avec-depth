import torch
import torch.nn.functional as F
from torch import nn
from ssim import SSIM
from geometrics import flow_warp, inverse_warp, pose2flow, mask_gen
from collections import defaultdict

get_ssim=SSIM().cuda()
eps=1e-3

def gradient(pred):
    dy = torch.abs(pred[..., :-1, :] - pred[..., 1:, :])
    dx = torch.abs(pred[..., :, -1:] - pred[..., :, 1:])
    return dx, dy

def upsample(src, shape):
    return F.interpolate(src, shape, mode="bilinear",align_corners=False)


def summerize(pred, target, cfg):

    loss_dict = defaultdict(float)
    weights = cfg['weights']
    B, _, H, W = target['img_src'].size()

    for scale in range(len(weights['multi_scale'])):
        scale_weight = weights['multi_scale'][scale]
        if cfg['use_flow']:
            # flowmap = upsample(pred['flowmap'][scale], (H, W))
            loss_dict['flow_consistency'] += loss_reconstruction(
                pred['flow_warped'][scale],
                target['img_tgt'],
                weights=weights)*scale_weight

            loss_dict['flow_smo'] += loss_smo_edge_aware(
                pred['flow_map'][scale], 
                target['img_src']
                )*scale_weight


        if cfg['use_depth']:
            # depthmap = upsample(pred['depthmap'][scale], (H, W))
            # depthmap.squeeze_(dim=1)
            # pose = pred['pose']
            # rigid_flow = pose2flow(
            #     depthmap, pose, target['intrinsics'], target['intrinsics_inv'])
            # mask = mask_gen(rigid_flow, flowmap) if cfg['use_mask'] else None
    
            loss_dict['reprojection_loss'] += loss_reconstruction(
                pred['depth_warped'][scale],
                target['img_tgt'],
                mask=pred['mask'],
                weights=weights)*scale_weight

            loss_dict['depth_smo'] += loss_smo_edge_aware(
                pred['depth_map'][scale], target['img_src'])*scale_weight
            
    # loss_dict['depth_disc']+=loss_disc(pred['depth_disc_score'],torch.ones_like(pred['depth_disc_score']))
    loss_dict['flow_disc']+=loss_disc(pred['flow_disc_score'],torch.ones_like(pred['flow_disc_score']))
    
    # if cfg['use_mask']:
    #     loss_dict['mask_loss']=(1-pred['mask']).mean()

    for k in weights.keys():
        if k in loss_dict.keys():
            loss_dict['loss_G'] += loss_dict[k]*weights[k]

    return loss_dict


def loss_reconstruction(img_tgt, img_warped, weights, mask=None):
    
    valid_area = 1 - (img_warped == 0).prod(1, keepdim=True).type_as(img_warped)
    if mask is not None:
        valid_area*=mask
    # penalty not applied this time
    penalty=valid_area.nelement()/(valid_area.sum()+1.)

    l1=torch.abs(img_warped-img_tgt)
    ssim_map = get_ssim(img_warped, img_tgt)
    loss_map=(ssim_map * weights['ssim'] + l1 * weights['l1'])*valid_area
    
    return loss_map.mean()


def loss_reprojection(rigid_flow, img_src, img_tgt, weights, mask=None,):
    # B,_, H, W = depth.size()
    # depth = torch.squeeze(depth, dim=1)
    img_tgt_warped = flow_warp(img_src, rigid_flow)
    return loss_reconstruction(img_tgt, img_tgt_warped, weights, mask)


def loss_flow_consistency(flow, img_src, img_tgt, weights):
    # B, _, H, W = flow.size()
    img_warped = flow_warp(img_src, flow)
    return loss_reconstruction(img_tgt, img_warped, weights)


def loss_smo_edge_aware(tgt, img):
    # B, _, H, W = tgt.size()
    grad_target_x, grad_target_y = gradient(tgt)
    grad_image_x, grad_image_y = gradient(img)
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
def loss_depth(gt, pred):
    # B, _, H, W = p.size()
    # gt = torch.nn.functional.adaptive_avg_pool2d(gt, (H, W))
    pred.squeeze_(dim=1)
    valid_area = 1 - (gt == 0).prod(1, keepdim=True).type_as(gt)
    pred*=valid_area
    gt_mean = gt.sum()/(valid_area.sum())
    pred_mean = pred.sum()/(valid_area.sum())
    pred = pred/pred_mean*gt_mean
    loss = torch.abs(pred-gt).mean()
   # loss = F.l1_loss(pred,gt)
    return loss


def loss_flow(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt = gt[:, 0, :, :], gt[:, 1, :, :]
    # resize to the gt
    pred = F.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    # mind the scale
    u_pred = pred[:, 0, :, :] * (w_gt/w_pred)
    v_pred = pred[:, 1, :, :] * (h_gt/h_pred)
    dist = F.l1_loss(u_pred,u_gt) + F.l1_loss(v_pred,v_gt)
    return dist


def loss_pose(gt, pred):         # [B,6]
    return F.l1_loss(pred,gt)


def evaluate(gt, pred, cfg):
    loss_dict = {}
    loss=0.
    if gt.get('depth_gt') is not None:
        val_loss_depth=loss_depth(gt['depth_gt'], pred['depthmap'].to('cpu'))
        loss_dict['depth_loss'] = val_loss_depth
        loss+=val_loss_depth*cfg['depth_loss']
    if gt.get('flow_gt') is not None:
        loss_flow=loss_flow(gt['flow_gt'], pred['flowmap'])
        loss_dict['flow_loss'] = loss_flow
        loss+=loss_flow*cfg['flow_loss']
    if gt.get('pose_gt') is not None:
        loss_flow=loss_flow(gt['pose_gt'], pred['pose'])
        loss_dict['pose_loss'] = loss_flow
        loss+=loss_flow*cfg['pose_loss']

    loss_dict['loss']=loss
    
    return loss_dict
