# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F
import cv2
import time

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


def img_center_padding(img_np, pad_ratio):

    ori_w, ori_h = img_np.shape[:2]

    w = round((1 + pad_ratio) * ori_w)
    h = round((1 + pad_ratio) * ori_h)

    img_pad_np = np.zeros((w, h, 3), dtype=np.uint8)
    offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
    img_pad_np[
        offset_h : offset_h + img_np.shape[0] :, offset_w : offset_w + img_np.shape[1]
    ] = img_np

    return img_pad_np, offset_w, offset_h

def normalize_rgb_tensor(img, imgenet_normalization=True):
    img = img / 255.
    if imgenet_normalization:
        img = (img - torch.tensor(IMG_NORM_MEAN, device=img.device).view(1, 3, 1, 1)) / torch.tensor(IMG_NORM_STD, device=img.device).view(1, 3, 1, 1)
    return img

def normalize_rgb(img, imagenet_normalization=True):
    """
    Args:
        - img: np.array - (W,H,3) - np.uint8 - 0/255
    Return:
        - img: np.array - (3,W,H) - np.float - -3/3
    """
    img = img.astype(np.float32) / 255.
    img = np.transpose(img, (2,0,1))
    if imagenet_normalization:
        img = (img - np.asarray(IMG_NORM_MEAN).reshape(3,1,1)) / np.asarray(IMG_NORM_STD).reshape(3,1,1)
    img = img.astype(np.float32)
    return img

def denormalize_rgb(img, imagenet_normalization=True):
    """
    Args:
        - img: np.array - (3,W,H) - np.float - -3/3
    Return:
        - img: np.array - (W,H,3) - np.uint8 - 0/255
    """
    if imagenet_normalization:
        img = (img * np.asarray(IMG_NORM_STD).reshape(3,1,1)) + np.asarray(IMG_NORM_MEAN).reshape(3,1,1)
    img = np.transpose(img, (1,2,0)) * 255.
    img = img.astype(np.uint8)
    return img

def unpatch(data, patch_size=14, c=3, img_size=224):
    # c = 3
    if len(data.shape) == 2:
        c=1
        data = data[:,:,None].repeat([1,1,patch_size**2])

    B,N,HWC = data.shape
    HW = patch_size**2
    c = int(HWC / HW)
    h = w = int(N**.5)
    p = q = int(HW**.5)
    data = data.reshape([B,h,w,p,q,c])
    data = torch.einsum('nhwpqc->nchpwq', data)
    return data.reshape([B,c,img_size,img_size])

def image_pad(img, img_size, device=torch.device('cuda')):
    img_pil = ImageOps.contain(img, (img_size, img_size))
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255))
    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side

    # Go to numpy 
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)

    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
    return x, img_pil_bis

def image_pad_cuda(img, img_size, rot=0, device=torch.device('cuda'), vis=False):
    img = torch.Tensor(img).to(device)
    img = torch.flip(img, dims=[2]).unsqueeze(0).permute(0, 3, 1, 2)
    if rot != 0:
        img = torch.rot90(img, rot, [2, 3])

    if vis:
        image = img.clone()[0].permute(1, 2, 0).cpu().numpy()
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        cv2.imshow('k4a', image[..., ::-1])
        cv2.waitKey(1)
    _, _, h, w = img.shape
    scale_factor = min(img_size / w, img_size / h)
    
    img = F.interpolate(img, scale_factor=scale_factor, mode='bilinear')
    
    _, _, h, w = img.shape
    
    pad_w = (img_size - w) // 2
    pad_h = (img_size - h) // 2

    
    img = F.pad(img,(pad_w, pad_w, pad_h, pad_h), mode='constant', value=255)
    
    # Normalize and go to torch.
    resize_img = normalize_rgb_tensor(img)
    return resize_img, img