import torch.nn as nn
from models import DnCNN, RIDNET, MemNet
from utils import weights_init_kaiming
import glob
import os
import json

class full_model(nn.Module):
    def __init__(self, args):
        super(full_model, self).__init__()
        
        if 'col' in args.experiment:
            self.model_channels = 1
        else:
            self.model_channels = 3
            
        if args.extend_input:
            self.model_channels += 3
            
        if args.backbone == 'D':
            print('** DnCNN backbone **')
            net = DnCNN(channels=self.model_channels, num_of_layers=args.dncnn_layers)
            net.apply(weights_init_kaiming)
        elif args.backbone == 'M':
            print('** MemNet backbone **')
            net = MemNet(in_channels=self.model_channels, channels=args.memnet_channels, num_memblock=args.memnet_memblocks, num_resblock=args.memnet_resblocks)
            net.apply(weights_init_kaiming)
        elif args.backbone == 'R':
            print('** RIDNet backbone **')
            net = RIDNET(in_channels=self.model_channels)
            net.apply(weights_init_kaiming)
        elif args.backbone == 'N':
            net = RNAN(n_colors=self.model_channels)
                
        self.backbone = net
        self.args = args
        self.sigmoid = nn.Sigmoid()
        
        
#     def forward(self, x, gan):
#         if self.args.extend_input:
#             phi = self.backbone(torch.cat((x, gan),1))
#         else:
#             phi = self.backbone(x)
#         out = (1-phi)*x + phi*gan
# #         print(f'phi: ({phi.mean().item():.4f},{phi.std().item():.4f}) x: {x.mean().item():.4f} gan: {gan.mean().item():.4f} out: {out.mean().item():.4f}')
# #         print(phi[0,0,100,100].item()*x[0,0,100,100].item(), (x*phi)[0,0,100,100].item())
# #         print(phi[0,1,100,100].item()*x[0,0,100,100].item(), (x*phi)[0,1,100,100].item())
# #         print('-----')
#         return out
    
    def forward(self, x, gan):
        if self.args.extend_input:
            phi = self.backbone(torch.cat((x, gan),1))
        else:
            phi = self.backbone(x)
        out = (1-phi)*x + phi*gan
        return out, phi
        
    
    def get_model_path(self):
        # Check available names in net_data dir first
        exp_name_list = glob.glob(os.path.join('net_data',self.args.experiment)+'*')
        ID = 0
        if exp_name_list != []:
            ID = len(exp_name_list)
        return os.path.join('net_data',self.args.experiment)+'_'+str(ID)
    
    
    def save_args(self, model_path):
        with open(os.path.join(model_path,'OURargs')+'.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
        return
    

    
    
    
    
def transfer_train_args(args):
    '''this function enables to run a test with only an experiment name
    the network information is auto-transfered to build the test model'''
    
    with open(os.path.join("net_data",args.experiment + '_' + str(args.test_model),"OURargs.txt"), "r") as read_file:
        args_train = json.load(read_file)
    
    args.backbone = args_train['backbone']
    args.dncnn_layers = args_train['dncnn_layers']
    args.memnet_channels = args_train['memnet_channels']
    args.memnet_memblocks = args_train['memnet_memblocks']
    args.memnet_resblocks = args_train['memnet_resblocks']
    args.batch_size = args_train['batch_size']
    args.epochs = args_train['epochs']
    args.lr = args_train['lr']
    args.phi_weight = args_train['phi_weight']
    args.train_count = args_train['train_count']
    args.experiment = args_train['experiment']
    args.extend_input = args_train['extend_input']

    return args