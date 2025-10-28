# utils.py
import torch
import torch.nn as nn
import numpy as np
import cv2
import math
import random

def weights_init(m):
    """
    Initializes custom weights for Conv and BatchNorm layers.
    From Cell 9.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def denorm(tensor):
    """
    Reverses the Tanh normalization from [-1, 1] to [0, 1].
    This is the reverse of transforms.Normalize(mean=[0.5], std=[0.5]).
    """
    # Moves tensor to CPU and converts to numpy
    img_np = tensor.detach().cpu().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    img_np = np.transpose(img_np, (1, 2, 0))
    # Denormalize (value * std + mean)
    img_np = (img_np * 0.5) + 0.5
    # Clip values to be safe
    img_np = np.clip(img_np, 0, 1)
    return img_np

def degrade_frame(frame_rgb):
    """
    Applies the same artificial degradation from training.
    From Cell 3 (and your inference cell).
    """
    # 1. Apply Gaussian Blur
    degraded_np = cv2.GaussianBlur(frame_rgb, (7, 7), 0)
    h, w, _ = degraded_np.shape
    
    # 2. Add random gray boxes
    for _ in range(random.randint(3, 6)):
        ph, pw = random.randint(20, 60), random.randint(20, 60)
        px, py = random.randint(0, w - pw), random.randint(0, h - ph)
        degraded_np[py:py+ph, px:px+pw, :] = 128 # 128-gray
        
    # 3. Add noise
    noise = np.random.normal(0, 10, degraded_np.shape).astype(np.float32)
    degraded_np = np.clip(degraded_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return degraded_np

def calculate_psnr(img1, img2):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    From your inference cell.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr
