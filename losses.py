import torch
import torch.nn.functional as F
from ssim import ssim
from geometrics import flow_warp, inverse_warp
eps=1e-8

# image reconstruction loss
def l1_norm(x):
    x=x.sqrt(x.pow(2)) # +eps
    return x.mean()

def loss_reconstruction(img_warped, img_tgt, mask=None):
    weights=[1,1,0.1]
    B,_,H,W = img_warped.size()
    # create valid mask of 
    valid_mask = 1 - (img_warped == 0).prod(1, keepdim=True).type_as(img_warped)
    if mask is not None:
        valid_mask = valid_mask*mask
    # to make sure the valid mask won't be all zeros
    valid_loss = 1 - torch.sum(valid_mask,dim=0)/valid_mask.nelement()
    dist = (img_warped-img_tgt)*valid_mask
    ssim_loss=(1-ssim(img_warped, img_tgt))*valid_mask

    loss = l1_norm(dist)*weights[0]+ssim_loss*weights[1]+valid_loss*weights[2]

# edge-aware smoothness loss



# supervised loss

def loss_flow(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    # resize to the gt
    pred = F.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    # mind the scale 
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    dist = l1_norm(u_gt - u_pred) + l1_norm(v_gt - v_pred)
    return dist


# forward-backward consistency loss

def loss_flow_consistency(forward, backward, img_src, img_tgt):
    B,_,H,W = forward.size()
    forward_warped=flow_warp(img_src,forward)
    backward_warped=flow_warp(img_tgt,backward)
    loss = loss_reconstruction(img_tgt,forward_warped)+loss_reconstruction(img_src, backward_warped)
    return loss


def loss_depth_consistency(depth_t0, depth_t1, pose, img_src, img_tgt, intrinsics,intrinsics_inv):
    B,_,H,W = depth_t0.size()
    forward_warped=inverse_warp(img_src, depth_t0, pose, intrinsics, intrinsics_inv)
    backward_warped=inverse_warp(img_tgt, depth_t1, pose.t(), intrinsics, intrinsics_inv) # the pose.t() is not set
    loss = loss_reconstruction(img_tgt,forward_warped)+loss_reconstruction(img_src, backward_warped)
    return loss


