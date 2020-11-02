from utils import file_utils

import argparse

from PIL import Image
import numpy as np

from skimage import metrics




def main(args):
    
    out_list = file_utils.image_files(args.gen_dir)
    gt_list = file_utils.image_files(args.gt_dir)
    
    for i, (out_path, gt_path) in enumerate(list(zip(out_list, gt_list))):
        
        
        out = np.array(file_utils.pil_loader(out_path))
        gt = np.array(file_utils.pil_loader(gt_path))
        
        psnr = metrics.peak_signal_noise_ratio(gt, out)
        ssim = metrics.structural_similarity(gt, out, multichannel=True)
        print("Image {0}, PSNR: {1}, SSIM: {2}".format(i,psnr,ssim))

if __name__=='__main__':

    parser = argparse.ArgumentParser(prog='stats', usage='python3 %(prog)s.py -gt_dir [groundtruth folder] -gen_dir [generation folder]', description='Compute psnr and ssim on images')
    parser.add_argument('-gt_dir', type=str, required=True, help='groundtruth directory')
    parser.add_argument('-gen_dir', type=str, required=True, help='generation directory')

    args = parser.parse_args()
    main(args)