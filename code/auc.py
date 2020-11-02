import sys
import numpy as np
import skimage

import argparse
import os


import skimage.io as io
import skimage.color as color


def main(ref_dir, gen_dir):
    """
    Compute the mean and standard deviation of the AuC metric from an image directory
    
    Args:
        ref_dir: reference images directory
        gen_dir: generated images directory
    """
    MAX_THRESHOLD = 150

    files = os.listdir(ref_dir)

    auc_list = np.empty(len(files))

    for i, file in enumerate(files):
        if(os.path.exists(os.path.join(gen_dir,file))):
        
            # Load images
            img0 = io.imread(os.path.join(ref_dir,file))
            img1 = io.imread(os.path.join(gen_dir,file))
        
            n_pixels = img0.shape[0]*img1.shape[1]
        
            # Convert the image to the AB color space
            img0_lab = color.rgb2lab(img0)
            img1_lab = color.rgb2lab(img1)
            img0_lab[:,:,0] = 0
            img1_lab[:,:,0] = 0
        
            dist = color.deltaE_cie76(img0_lab,img1_lab)
        
            auc = 0.0
        
            # Compute the cumulative mass function of the distance function over the 0-150 range
            for threshold in range(0,MAX_THRESHOLD):
                pix_under_curve = len(dist[dist<=threshold])
                auc += pix_under_curve/n_pixels
            auc /= MAX_THRESHOLD
            auc_list[i] = auc
        
            print('%s: %.4f'%(file,auc))
            
    print("Auc mean: {:.4f}".format(auc_list.mean()))
    print("Auc std: {:.4f}".format(auc_list.std()))
    
if __name__=='__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d0','--dir0', type=str, required=True, help='reference images directory')
    parser.add_argument('-d1','--dir1', type=str, required=True, help='generated images directory')
    
    opt = parser.parse_args()
    main(opt.dir0, opt.dir1)