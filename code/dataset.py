import os
import torch
import cv2
import torch.utils.data as data
import numpy as np

def normalize(x):
    return x/255.


class Dataset(data.Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        
        self.experiment = args.experiment
        self.train_count = args.train_count
        self.args = args

    def __len__(self):
        if self.args.test:
            return 300-self.train_count
        else:
            return self.train_count

    def __getitem__(self, index):
        
        if self.args.test:
            index += 250
        
        root_path = os.path.join('inter_data', self.experiment)
        if 'col' in self.experiment:
            degraded = normalize(  cv2.imread(os.path.join(root_path, 'input', str(index)+'.png'), cv2.IMREAD_GRAYSCALE)  )
        elif 'den' in self.experiment:
            degraded = np.load(os.path.join(root_path, 'input', str(index)+'.npy'))
        else:
            degraded = normalize(cv2.cvtColor(  cv2.imread(os.path.join(root_path, 'input', str(index)+'.png'))  , cv2.COLOR_RGB2BGR))
        
        ground_truth = normalize(cv2.cvtColor(  cv2.imread(os.path.join(root_path, 'GT', str(index)+'.png'))  , cv2.COLOR_RGB2BGR))
        gan_inverted = normalize(cv2.cvtColor(  cv2.imread(os.path.join(root_path, 'mGANoutput', str(index)+'.png'))  , cv2.COLOR_RGB2BGR))
        
        
        if 'col' in self.experiment:
            return torch.Tensor(degraded).unsqueeze_(-1).permute(2,0,1), torch.Tensor(ground_truth).permute(2,0,1), torch.Tensor(gan_inverted).permute(2,0,1)
        else:
            return torch.Tensor(degraded).permute(2,0,1), torch.Tensor(ground_truth).permute(2,0,1), torch.Tensor(gan_inverted).permute(2,0,1)



# print('0-- ', torch.Tensor(degraded).shape)
# print('1-- ', torch.Tensor(degraded).unsqueeze_(-1).shape)
# print('2-- ', torch.Tensor(degraded).unsqueeze_(-1).permute(2,0,1).shape)
