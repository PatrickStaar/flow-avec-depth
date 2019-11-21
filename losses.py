import torch
import torch.nn.functional as F
from ssim import ssim
from geometrics import flow_warp, inverse_warp, pose2flow, mask_gen
eps=1e-8



# image reconstruction loss
def l1_norm(x):
    x=x.sqrt(x.pow(2)).mean() # +eps
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


# multi_scale masks
def multi_scale_mask(multi_scale, depth, pose, flow, intrinsics, intrinsics_inv):
    masks_multi = []
    for s in range(multi_scale):

        flow_rigid_foward = pose2flow(depth[s][:,0], pose, intrinsics, intrinsics_inv)
        flow_rigid_backward = pose2flow(depth[s][:,1], pose.inverse(), intrinsics, intrinsics_inv)

        mask0=mask_gen(flow[s][:,0],flow_rigid_foward)
        mask1=mask_gen(flow[s][:,1],flow_rigid_backward) 
        masks_multi.append((mask0, mask1))


# forward-backward consistency loss
def loss_flow_consistency(forward, backward, img_src, img_tgt, multi_scale=0):

    loss=[]
    if multi_scale > 0:
        for s in range(multi_scale):
            B,_,H,W = forward.size()
            img_src_s=torch.nn.functional.adaptive_avg_pool2d(img_src,(H, W))
            img_tgt_s=torch.nn.functional.adaptive_avg_pool2d(img_tgt,(H, W))

            forward_warped=flow_warp(img_src_s,forward)
            backward_warped=flow_warp(img_tgt_s,backward)
            loss.append(loss_reconstruction(img_tgt_s,forward_warped)+loss_reconstruction(img_src_s, backward_warped))

    else:
        loss.append(loss_reconstruction(img_tgt, flow_warp(img_src,forward)) + \
                    loss_reconstruction(img_src, flow_warp(img_tgt, backward)))
    return loss


def loss_depth_consistency(depth_t0, depth_t1, pose, img_src, img_tgt, intrinsics,intrinsics_inv, mask=None, multi_scale=0):
       
    loss=[]
    if multi_scale > 0:
        origine=img_src.size()[-1]
        for s in range(multi_scale):
            _,_,H,W = depth_t0[s].size()
            ratio=origine/W

            img_src_s=torch.nn.functional.adaptive_avg_pool2d(img_src,(H, W))
            img_tgt_s=torch.nn.functional.adaptive_avg_pool2d(img_tgt,(H, W))

            intrinsics_s=torch.cat((intrinsics[:, 0:2]/ratio, intrinsics[:, 2:]), dim=1)
            intrinsics_inv_s=torch.cat((intrinsics_inv[:, 0:2]/ratio, intrinsics[:, 2:]), dim=1)
            
            forward_warped=inverse_warp(img_src_s, depth_t0[s], pose, intrinsics_s, intrinsics_inv_s)
            backward_warped=inverse_warp(img_tgt_s, depth_t1[s], pose.inverse(), intrinsics_s, intrinsics_inv_s)
            
            l = loss_reconstruction(img_tgt_s,forward_warped,mask[:,0])+loss_reconstruction(img_src_s, backward_warped,mask[:,1])
            
            loss.append(l)
    
    else:
        forward_warped=inverse_warp(img_src, depth_t0[-1], pose, intrinsics, intrinsics_inv)
        backward_warped=inverse_warp(img_tgt, depth_t1[-1], pose.inverse(), intrinsics, intrinsics_inv)
        l = loss_reconstruction(img_tgt_s,forward_warped,mask)+loss_reconstruction(img_src_s, backward_warped,mask)
        loss.append(l)

    return loss


