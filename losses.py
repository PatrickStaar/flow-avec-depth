import torch
import torch.nn.functional as F
from ssim import ssim
from geometrics import flow_warp, inverse_warp, pose2flow, mask_gen
import cfg


eps = 1e-8
multi_scale_weights = cfg.multi_scale_weight
reconstruction_weights = cfg.reconstruction_weights

# image reconstruction loss


def l2_norm(x):
    x = x.norm(dim=1)
    return x.mean()


def gradient(pred):
    dy = pred[:, :, 1:] - pred[:, :, :-1]
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return dx, dy


def loss_reconstruction(img_tgt, img_warped, mask=None):

    B, _, H, W = img_warped.size()
    # create valid mask of
    valid_mask = 1 - (img_warped == 0).\
                    prod(1, keepdim=True).type_as(img_warped)

    if mask is not None:
        valid_mask = valid_mask*mask
    # to make sure the valid mask won't be all zeros
    img_warped = img_warped*valid_mask
    img_tgt = img_tgt*valid_mask

    valid_loss = 1 - valid_mask.sum()/float(valid_mask.nelement())
    ssim_loss = (1-ssim(img_warped, img_tgt))
    l2_loss = l2_norm(img_warped-img_tgt)

    loss = l2_loss*reconstruction_weights[0] +\
        ssim_loss*reconstruction_weights[1] +\
        valid_loss*reconstruction_weights[2]
    return loss


# edge-aware smoothness loss
def loss_smooth(depth):
    loss = 0.
    for i, d in enumerate(depth):
        dx, dy = gradient(d)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += ((dx2.abs().mean() +
                  dxdy.abs().mean() +
                  dydx.abs().mean() +
                  dy2.abs().mean()) *
                 multi_scale_weights[i])
    return loss


# supervised loss
def loss_depth(gt, pred):
    loss = 0.
    for i, p in enumerate(pred):
        B, _, H, W = p.size()
        gt = torch.nn.functional.adaptive_avg_pool2d(gt, (H, W))
        dist = l2_norm(gt-p)
        loss += dist*multi_scale_weights[i]
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

    dist = l2_norm(u_gt - u_pred) + l2_norm(v_gt - v_pred)
    return dist


def loss_pose(gt, pred):
    # [B,6]
    dist = l2_norm(gt-pred)
    return dist


# multi_scale masks
def multi_scale_mask(multi_scale, depth, pose, flow, intrinsics, intrinsics_inv):
    masks = []
    d_t0, d_t1 = depth
    f_forward, f_backward = flow

    for s in range(multi_scale):
        flow_rigid_foward = pose2flow(
            d_t0[s], pose, intrinsics, intrinsics_inv)
        flow_rigid_backward = pose2flow(
            d_t1[s], -pose, intrinsics, intrinsics_inv)

        mask0 = mask_gen(f_forward[s], flow_rigid_foward)
        mask1 = mask_gen(f_backward[s], flow_rigid_backward)
        masks.append((mask0, mask1))
    return masks


# unsupervised loss
# forward-backward consistency loss
# def loss_flow_consistency(forward, backward, img_src, img_tgt, multi_scale=0):

#     def flow_consistency(forward, backward):
#         return l2_norm(-forward-backward)

#     losses = []
#     if multi_scale > 0:
#         for s in range(multi_scale):
#             B, _, H, W = forward[s].size()
#             img_src_s = torch.nn.functional.adaptive_avg_pool2d(
#                 img_src, (H, W))
#             img_tgt_s = torch.nn.functional.adaptive_avg_pool2d(
#                 img_tgt, (H, W))

#             img_tgt_warped = flow_warp(img_src_s, forward[s])
#             img_src_warped = flow_warp(img_tgt_s, backward[s])
#             losses.append(
#                 loss_reconstruction(img_tgt_s, img_tgt_warped) +
#                 loss_reconstruction(img_src_s, img_src_warped) +
#                 flow_consistency(forward[s], backward[s]))
#     else:
#         losses.append(loss_reconstruction(img_tgt, flow_warp(img_src, forward)) +
#                       loss_reconstruction(img_src, flow_warp(img_tgt, backward)))
#     loss = 0
#     for i in range(len(losses)):
#         loss += losses[i]*multi_scale_weights[i]

#     return loss


def loss_flow_consistency(flow, img_src, img_tgt, multi_scale=0):

    loss = 0.
    if multi_scale > 0:
        for s in range(multi_scale):
            B, _, H, W = flow[s].size()
            img_src_s = torch.nn.functional.adaptive_avg_pool2d(
                img_src, (H, W))
            img_tgt_s = torch.nn.functional.adaptive_avg_pool2d(
                img_tgt, (H, W))

            img_tgt_warped = flow_warp(img_src_s, flow[s])

            loss+=(loss_reconstruction(img_tgt_s, img_tgt_warped)*multi_scale_weights[s])
    else:
        loss += loss_reconstruction(img_tgt, flow_warp(img_src, flow[0]))

    return loss



def loss_depth_consistency(
        depth_t0, depth_t1, pose, img_src, img_tgt,
        intrinsics, intrinsics_inv, mask=None, multi_scale=0):

    loss = 0

    if multi_scale > 0:
        origine = img_src.size()[-1]
        for s in range(multi_scale):
            _, H, W = depth_t0[s].size()
            ratio = origine/W

            img_src_s = torch.nn.functional.adaptive_avg_pool2d(
                img_src, (H, W))
            img_tgt_s = torch.nn.functional.adaptive_avg_pool2d(
                img_tgt, (H, W))

            intrinsics_s = torch.cat(
                (intrinsics[:, 0:2]/ratio, intrinsics[:, 2:]), dim=1)
            intrinsics_inv_s = torch.cat(
                (intrinsics_inv[:, 0:2]/ratio, intrinsics[:, 2:]), dim=1)

            img_tgt_warped = inverse_warp(
                img_src_s, depth_t0[s], pose, intrinsics_s, intrinsics_inv_s)
            img_src_warped = inverse_warp(
                img_tgt_s, depth_t1[s], -pose, intrinsics_s, intrinsics_inv_s)

            l = loss_reconstruction(img_tgt_s, img_tgt_warped, mask[s][0]) +\
                loss_reconstruction(img_src_s, img_src_warped, mask[s][1])

            loss += (l*multi_scale_weights[s])

    else:
        img_tgt_warped = inverse_warp(
            img_src, depth_t0[0], pose, intrinsics, intrinsics_inv)
        img_src_warped = inverse_warp(
            img_tgt, depth_t1[0], -pose, intrinsics, intrinsics_inv)

        l = loss_reconstruction(img_tgt_s, img_tgt_warped, mask[0]) + \
            loss_reconstruction(img_src_s, img_src_warped, mask[1])

        loss += l

    return loss


def loss_sum(loss_dict):
    alpha = cfg.loss_weight
    loss = 0
    for key in loss_dict.keys():
        loss += (loss_dict[key]*alpha[key])
    return loss
