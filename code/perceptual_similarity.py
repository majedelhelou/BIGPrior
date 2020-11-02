# Inspired from PerceptualSimilarity/compute_dists_dirs.py
import sys
import os
import numpy as np
from skimage import color,metrics
sys.path.append(os.path.join(os.path.dirname(__file__), 'PerceptualSimilarity'))

import argparse
import os
from PerceptualSimilarity import models
from PerceptualSimilarity.util import util

def main(ref_dir, generated_dir, version='0.0', use_gpu=True):
    """
    Compute the mean and standard deviation of the LPIPS, PSNR and SSIM metrics over an image directory
    
    Args:
        ref_dir: reference images directory
        generated_dir: generated images directory
        version: version of LPIPS to use, default 0.0
        use_gpu: whether to use gpu for faster computation
    """
    
    ## Initialize the LPIPS model
    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=use_gpu,version=version)

    files = os.listdir(ref_dir)

    lpips_list = np.empty(len(files))
    psnr_list = np.empty(len(files))
    ssim_list = np.empty(len(files))

    for i, file in enumerate(files):
        if(os.path.exists(os.path.join(generated_dir,file))):
        
            # Load images
            img0_np = util.load_image(os.path.join(ref_dir,file))
            img1_np = util.load_image(os.path.join(generated_dir,file))
        
            img0 = util.im2tensor(img0_np)
            img1 = util.im2tensor(img1_np)



            if(use_gpu):
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute LPIPS distance
            dist01 = model.forward(img0,img1)
            lpips_list[i] = dist01
        
            # Compute PSNR value
            psnr = metrics.peak_signal_noise_ratio(img0_np, img1_np)
            psnr_list[i] = psnr
        
            # Compute SSIM value
            ssim = metrics.structural_similarity(img0_np, img1_np, multichannel=True)
            ssim_list[i] = ssim

            print('%s: %.4f, %.4f, %.4f'%(file,dist01,psnr,ssim))
            
    print("LPIPS mean: {:.4f}".format(lpips_list.mean()))
    print("LPIPS std: {:.4f}".format(lpips_list.std()))

    print("PSNR mean: {:.4f}".format(psnr_list.mean()))
    print("PSNR std: {:.4f}".format(psnr_list.std()))

    print("SSIM mean: {:.4f}".format(ssim_list.mean()))
    print("SSIM std: {:.4f}".format(ssim_list.std()))

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d0','--dir0', type=str, required=True, help='reference images directory')
    parser.add_argument('-d1','--dir1', type=str, required=True, help='generated images directory')
    parser.add_argument('-v','--version', type=str, default='0.0', help='version of LPIPS')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

    opt = parser.parse_args()
    main(opt.dir0, opt.dir1, opt.version, opt.use_gpu)

