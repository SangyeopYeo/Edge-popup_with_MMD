import torch
import math
from tqdm import tqdm
import numpy as np


def prepare_generated_img(opt, netG, latent, device):
    with torch.no_grad():
        images = netG(latent)
        images = quantize_images(images, opt.netGType)
    return images


def get_noise(opt, test_num, device):
    """
    device: testing for CUDA
    """
    if opt.netGType == "sngan":
        latent = torch.randn(test_num, opt.nz, device=device)
    else:
        latent = torch.randn(test_num, opt.nz, 1, 1, device=device)
    return latent


def quantize_images(x, gen_model):
    # -1 ~ 1 -> 0 ~ 255
    x = (x + 1) / 2
    x = (255.0 * x + 0.5).clamp(0.0, 255.0)
    x = x.detach().cpu().numpy().astype(np.uint8)
    return x
