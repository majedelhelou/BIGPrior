import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
import skimage.color as color
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'PerceptualSimilarity'))
from PerceptualSimilarity import models
from PerceptualSimilarity.util import util


def compute_auc(img0, img1):
    
    MAX_THRESHOLD = 150
    n_pixels = img0.shape[0]*img1.shape[1]

    # Convert the image to the AB color space
    img0_lab = color.rgb2lab(img0)
    img1_lab = color.rgb2lab(img1)
    img0_lab[:,:,0] = 0
    img1_lab[:,:,0] = 0

    dist = color.deltaE_cie76(img0_lab,img1_lab)

    # Compute the cumulative mass function of the distance function over the 0-MAX_THRESHOLD range
    auc = 0.0
    for threshold in range(0,MAX_THRESHOLD):
        pix_under_curve = len(dist[dist<=threshold])
        auc += pix_under_curve/n_pixels
    auc /= MAX_THRESHOLD
    
    return 100*auc


def compute_lpips(gt_path, inp_path, version='0.0', use_gpu=True):
    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=use_gpu,version=version)
    img0_np = util.load_image(gt_path)
    img1_np = util.load_image(inp_path)
    img0 = util.im2tensor(img0_np)
    img1 = util.im2tensor(img1_np)
    if(use_gpu):
        img0 = img0.cuda()
        img1 = img1.cuda()

    dist01 = model.forward(img0,img1)
    if use_gpu:
        return dist01.item()
    return dist01


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname != 'BNReLUConv': #filtered for MemNet: BNReLUConv, ResidualBlock, MemoryBlock
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            # nn.init.uniform(m.weight.data, 1.0, 0.02)
            m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
            nn.init.constant_(m.bias.data, 0.0)
