import torch
import torch.nn.functional as F
from ssim import ssim
from geometrics import flow_warp, inverse_warp, pose2flow, mask_gen


def gradient(pred):
    dy = torch.abs(pred[..., :-1, :] - pred[..., 1:, :])
    dx = torch.abs(pred[..., :, -1:] - pred[..., :, 1:])
    return dx, dy

def upsample(src, shape):
    return F.interpolate(src, shape, mode="bilinear")


# def mask(multi_scale, depth, pose, flow, intrinsics, intrinsics_inv):
#     masks = []
#     d_t0, d_t1 = depth
#     f_forward, f_backward = flow

#     for s in range(multi_scale):
#         flow_rigid_foward = pose2flow(
#             d_t0[s], pose, intrinsics, intrinsics_inv)
#         flow_rigid_backward = pose2flow(
#             d_t1[s], -pose, intrinsics, intrinsics_inv)

#         mask = mask_gen(f_forward[s], flow_rigid_foward)
#         # mask1 = mask_gen(f_backward[s], flow_rigid_backward)
#         masks.append((mask0, mask1))
#     return masks


def summerize(pred, target, cfg):
    loss_dict = dict(
        loss=0.,
        reprojection_loss=0.,
        flow_consistency=0.,
        flow_smo=0.,
        depth_smo=0.,
        # disc=0.,
    )
    weights = cfg['weights']
    B, _, H, W = target['img_src'].size()

    for scale in range(len(weights['multi_scale'])):
        scale_weight = weights['multi_scale'][scale]
        if cfg['use_depth']:
            depthmap = upsample(pred['depthmap'][scale], (H, W))
            depthmap.squeeze_(dim=1)
            pose = pred['pose']
            loss_dict['reprojection_loss'] += loss_reprojection(
                depthmap, pose,
                target['img_src'],
                target['img_tgt'],
                target['intrinsics'],
                target['intrinsics_inv'],
                mask=pred['mask'],
                weights=weights)*scale_weight

            loss_dict['depth_smo'] += loss_smo(
                depthmap, target['img_src'])*scale_weight

        if cfg['use_flow']:
            flowmap = upsample(pred['flowmap'][scale], (H, W))
            loss_dict['flow_consistency'] += loss_flow_consistency(
                flowmap,
                target['img_src'],
                target['img_tgt'],
                weights=weights)*scale_weight

            loss_dict['flow_smo'] += loss_smo(
                flowmap, target['img_src'],
            )*scale_weight

        if cfg['use_disc']:
            pass

    for k in weights.keys():
        if k in loss_dict.keys():
            loss_dict['loss'] += loss_dict[k]*weights[k]

    return loss_dict


def loss_reconstruction(img_tgt, img_warped, weights, mask=None):
    if mask is not None:
        img_tgt = img_tgt*mask
        img_warped = img_warped*mask
    ssim_value = 1-ssim(img_warped, img_tgt)
    l1 = F.l1_loss(img_warped, img_tgt)
    return ssim_value * weights['ssim'] + l1 * weights['l1']


def loss_reprojection(depth, pose, img_src, img_tgt,
                      intrinsics, intrinsics_inv, weights, mask=None,):
    # B,_, H, W = depth.size()
    # depth = torch.squeeze(depth, dim=1)
    img_tgt_warped = inverse_warp(
        img_src, depth, pose, intrinsics, intrinsics_inv)
    return loss_reconstruction(img_tgt, img_tgt_warped, weights, mask)


def loss_flow_consistency(flow, img_src, img_tgt, weights):
    # B, _, H, W = flow.size()
    img_warped = flow_warp(img_src, flow)
    return loss_reconstruction(img_tgt, img_warped, weights)


def loss_smo(tgt, img):
    # B, _, H, W = tgt.size()
    grad_target_x, grad_target_y = gradient(tgt)
    grad_image_x, grad_image_y = gradient(img)
    grad_image_x = torch.mean(grad_image_x, 1, keepdim=True)
    grad_image_y = torch.mean(grad_image_y, 1, keepdim=True)
    loss_smo_x = torch.exp(-grad_image_x)*grad_target_x
    loss_smo_y = torch.exp(-grad_image_y)*grad_target_y
    return loss_smo_x.mean()+loss_smo_y.mean()


# supervised
def loss_depth(gt, pred):
    # B, _, H, W = p.size()
    # gt = torch.nn.functional.adaptive_avg_pool2d(gt, (H, W))
    loss = F.l1_loss(pred,gt)
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
    loss=0
    if gt.get('depth_gt') is not None:
        loss_depth=loss_depth(gt['depth_gt'], pred['depthmap'])
        loss_dict['depth_loss'] = loss_depth
        loss+=loss_depth*cfg['depth_loss']
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
