from __future__ import division
import torch
import random
import numpy as np
# from cv2 import imresize, imrotate
import cv2


'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics):
        img, ins = images, intrinsics
        for t in self.transforms:
            img, ins = t(img, ins)
        return img, ins


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, intrinsics


class NormalizeLocally(object):

    def __call__(self, images, intrinsics):
        image_tensor = torch.stack(images)
        assert(image_tensor.size(1)==3)   #3 channel image
        mean = image_tensor.transpose(0,1).contiguous().view(3, -1).mean(1)
        std = image_tensor.transpose(0,1).contiguous().view(3, -1).std(1)

        for tensor in images:
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
        return images, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix
     to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)          
        return tensors, torch.from_numpy(intrinsics)


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[0,2] = w - output_intrinsics[0,2]
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, output_intrinsics


class Scale(object):
    """Scales images to a particular size"""
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        in_h, in_w, _ = images[0].shape
        output_intrinsics = np.copy(intrinsics)

        output_intrinsics[0] *= (self.w / in_w)
        output_intrinsics[1] *= (self.h / in_h)
        scaled_images = [cv2.resize(im, (self.w, self.h)) for im in images]

        return scaled_images, output_intrinsics
