import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from our_model import full_model
from our_model import transfer_train_args
import json
from dataset import Dataset
from utils import *
import time
from tqdm import tqdm
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lrs

def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    
    if args.test == False:
        # Dataset
        dataset_train = Dataset(args)
        loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=args.batch_size, shuffle=True)
        print(f'{int(len(dataset_train))} training samples')

        # Model
        model = full_model(args)
        model_path = model.get_model_path()
        os.makedirs(model_path, exist_ok=True)
        model.save_args(model_path)
        model = nn.DataParallel(model).cuda()
        print('Trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Loss
        criterion = nn.MSELoss(reduction='mean')
        criterion.cuda()

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        iters = len(loader_train)
        scheduler = lrs.CosineAnnealingWarmRestarts(optimizer,round(4*iters/args.batch_size))
        
        # Training
        train_loss_log = np.zeros(args.epochs)
        for epoch in range(args.epochs):
            for param_group in optimizer.param_groups:
                print(f'\nCurrent lr={param_group["lr"]}')
            
            start_time = time.time()
            for i, train_data in (enumerate(loader_train)):
                model.train()
                model.zero_grad()
                optimizer.zero_grad()

                degraded, ground_truth, gan_inverted = train_data[0].cuda(), train_data[1].cuda(), train_data[2].cuda()
                degraded, ground_truth, gan_inverted = degraded-0.5, ground_truth-0.5, gan_inverted-0.5
                
                out_train, phi_train = model(degraded, gan_inverted)
                loss = criterion(out_train, ground_truth)
                if args.phi_weight > 0:
                    loss += args.phi_weight * abs(phi_train.mean())
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + i / iters)

                train_loss_log[epoch] += loss.item()
                np.save(os.path.join(model_path,'train_loss'), train_loss_log)

            train_loss_log[epoch] = train_loss_log[epoch] / len(loader_train)
            elapsed = time.time() - start_time
            print('Epoch %d: loss=%.4f, elapsed time: %.2fmin' %(epoch, train_loss_log[epoch], elapsed/60.))
            torch.save(model.state_dict(), os.path.join(model_path, 'net_%d.pth' % (epoch)) )

        # Run test
        args_test = args
        args_test.test = True
        p, n = model_path.split('/')
        args_test.test_model = n.split('_')[-1]
        args_test.test_epoch = epoch
        main(args_test)
        print('Completed: ', model_path)
        
        
    else:
        args = transfer_train_args(args)            
        if not (args.test_data == 'Default'):
            temp = args.experiment
            args.experiment = args.test_data
            print(args.experiment)
            dataset_test = Dataset(args)
            args.experiment = temp
        else:
            dataset_test = Dataset(args)
        loader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=1, shuffle=False)
        print(f'{int(len(dataset_test))} test samples, with model {args.experiment}_{args.test_model}, epoch {args.test_epoch}')

        # Load model
        model = full_model(args)
        save_name = 'OURoutput' if args.test_data=='Default' else 'OUROODoutput'
        save_path = os.path.join('inter_data', args.experiment, save_name)
        os.makedirs(save_path, exist_ok=True)
        model.save_args(os.path.join('inter_data', args.experiment, save_name))
        model = nn.DataParallel(model).cuda()
        model_path = os.path.join("net_data",args.experiment + '_' + str(args.test_model))
        model.load_state_dict(torch.load(os.path.join(model_path, f'net_{args.test_epoch}.pth' )))
        model.eval()
        
        
        # Evaluate
        Mphi = np.zeros(len(dataset_test))
        MSE = np.zeros(len(dataset_test))
        MSE_GAN = np.zeros(len(dataset_test))
        AUC = np.zeros(len(dataset_test))
        AUC_GAN = np.zeros(len(dataset_test))
        for i, test_data in tqdm(enumerate(loader_test)):
            with torch.no_grad():

                degraded, ground_truth, gan_inverted = test_data[0].cuda(), test_data[1].cuda(), test_data[2].cuda()
                degraded, ground_truth, gan_inverted = degraded-0.5, ground_truth-0.5, gan_inverted-0.5
                
                out_test, phi_test = model(degraded, gan_inverted)
                torch.clamp(out_test, min=-0.5, max=0.5)

                out_test, degraded, ground_truth, gan_inverted = out_test+0.5, degraded+0.5, ground_truth+0.5, gan_inverted+0.5
                
                
                save_image(out_test, os.path.join(save_path, f'{i+250}.png'))
                save_image(phi_test, os.path.join(save_path, f'{i+250}_phi.png'))

                # Metrics
                MSE[i] = ((out_test-ground_truth)**2).mean()
                MSE_GAN[i] = ((gan_inverted-ground_truth)**2).mean()
                Mphi[i] = phi_test.mean()
                if 'col' in args.experiment:
                    AUC[i] = compute_auc(out_test.cpu()[0].permute(1,2,0), ground_truth.cpu()[0].permute(1,2,0))
                    AUC_GAN[i] = compute_auc(gan_inverted.cpu()[0].permute(1,2,0), ground_truth.cpu()[0].permute(1,2,0))

        # Save
        np.save(os.path.join(model_path,'MSE'), MSE)
        np.save(os.path.join(model_path,'MSE_GAN'), MSE_GAN)
        np.save(os.path.join(model_path,'Mphi'), Mphi)
        if 'col' in args.experiment:
            np.save(os.path.join(model_path,'AUC'), AUC)
            np.save(os.path.join(model_path,'AUC_GAN'), AUC_GAN)

        print(f'MMSE={MSE.mean():.5f}, mean phi={Mphi.mean():.4f}')
        if 'col' in args.experiment:
            print(f' AUC={AUC.mean():.4f}, GAN AUC={AUC_GAN.mean():.4f}')        
        


if __name__ == "__main__":
            
    parser = argparse.ArgumentParser(description="BIGPrior")

    parser.add_argument("--experiment", type=str, default="inpmask_church", help="Name of the experiment to run")
    parser.add_argument("--extend_input", type=bool, default=False, help="Set to True for using the inversion in training")
    
    # Network
    parser.add_argument("--backbone", type=str, default="D", help="Backbone architecture for confidence; DnCNN (D), MemNet (M), RIDNet (R), RNAN (N)")
    parser.add_argument("--dncnn_layers", type=int, default=17, help="DnCNN number of layers")
    parser.add_argument("--memnet_channels", type=int, default=5, help="MemNet number of channels")
    parser.add_argument("--memnet_memblocks", type=int, default=5, help="MemNet number of memblocks")
    parser.add_argument("--memnet_resblocks", type=int, default=5, help="MemNet number of resblocks")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument("--phi_weight", type=float, default=0.0, help="Weight for phi norm regularization")
    parser.add_argument("--train_count", type=int, default=250, help="Number of images (in the experiment) to be used for training, the rest is for test")
        
    # Testing (all needed train args are taken from the pre-trained experiment)
    parser.add_argument("--test", type=bool, default=False, help="Set to True for running inference")
    parser.add_argument("--test_model", type=int, default=0, help="ID of the test model")
    parser.add_argument("--test_epoch", type=int, default=49, help="Epoch used for the testing")
    #optional:
    parser.add_argument("--test_data", type=str, default="Default", help="Default: (300-train_count) in the train experiment folder, else choose an experiment (inpmask_church)")
    
    
    args = parser.parse_args()

    main(args)