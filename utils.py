
import cv2
from torch.nn.functional import grid_sample
import torch
import numpy as np

im = cv2.imread('rs.jpg')
im=im.astype(np.float)
im /=255
im = im.transpose([2,0,1])
im = torch.from_numpy(im).unsqueeze(0).float()
_,C,H,W = im.shape
x = torch.randn(1,H,W,2)-0.5
warp=grid_sample(im,x)
warp=warp.numpy().squeeze(0)

warp*=255
cv2.imwrite('ss.jpg',warp.astype(np.uint8).transpose([1,2,0]))

















