import torch
import random as rand
import numpy as np


def get_var_mask(shape,min_p=2,max_p=4,width_mean=64,width_var=32):
    ''' returns a variable binary mask with torch dimensions shape '''
    
    total_patches = rand.randint(min_p,max_p)
    
    mask = torch.zeros(shape)
    
    for _ in range(total_patches):
        valid_patch = False
        while not valid_patch:
            x, y = rand.randint(0,shape[0]), rand.randint(0,shape[1])
            stretch_x, stretch_y = np.random.normal(width_mean, width_var), np.random.normal(width_mean, width_var)
            stretch_x, stretch_y = abs(round(stretch_x)), abs(round(stretch_y))
            
            if stretch_x>8 and stretch_y>8:
                if (x+stretch_x)<shape[0] and (y+stretch_y)<shape[1]:
                    valid_patch = True
                    mask[x:x+stretch_x, y:y+stretch_y] = 1
                    
    return mask