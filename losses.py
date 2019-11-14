import torch
from ssim import ssim

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




# forward-backward consistency loss




