import os
import argparse
import math, time
import torch
import cv2
import json

from utils.file_utils import image_files, load_as_tensor, Tensor2PIL, split_to_batches
from utils.image_precossing import _sigmoid_to_tanh, _tanh_to_sigmoid, _add_batch_one
from derivable_models.derivable_generator import get_derivable_generator
from inversion.inversion_methods import get_inversion
from utils.manipulate import masked_loss, parsing_mask, mask_images, combine_inpainting
from inversion.losses import get_loss
from models.model_settings import MODEL_POOL
from utils.manipulate import convert_array_to_images

from mask_utils import get_var_mask
from torchvision.utils import save_image


def main(args):
    os.makedirs(args.outputs+'/input', exist_ok=True)
    os.makedirs(args.outputs+'/GT', exist_ok=True)
    os.makedirs(args.outputs+'/mGANoutput', exist_ok=True)
    with open(args.outputs+'/mGANargs.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)
    loss = get_loss(args.loss_type, args)
    mask = parsing_mask('code/mganprior/masks/'+args.mask).cuda()
    mask_cpu = parsing_mask('code/mganprior/masks/'+args.mask)
    crop_loss = masked_loss(loss, mask)
    generator.cuda()
    loss.cuda()
    inversion = get_inversion(args.optimization, args)
    image_list = image_files(args.target_images)
    if len(image_list)>300:
        print('Limiting the image set to 300.')
        image_list = image_list[:300]
    frameSize = MODEL_POOL[args.gan_model]['resolution']

    start_time = time.time()
    for i, images in enumerate(split_to_batches(image_list, 1)):
        print('%d: Processing %d images :' % (i, 1), end='')
        pt_image_str = '%s\n'
        print(pt_image_str % tuple(images))

        image_name_list = []
        image_tensor_list = []
        for image in images:
            image_name_list.append(os.path.split(image)[1])
            image_tensor_list.append(_add_batch_one(load_as_tensor(image)))
        y_gt = _sigmoid_to_tanh(torch.cat(image_tensor_list, dim=0)).cuda()
        # Invert
        if args.varmask:
            os.makedirs(args.outputs+'/mask', exist_ok=True)
            mask_cpu = get_var_mask(y_gt.shape[-2:],args.min_p,args.max_p,args.width_mean,args.width_var)
            mask = mask_cpu.cuda()
            save_image(mask, os.path.join(args.outputs + '/mask/%d%s' % (i, '.png')))
            crop_loss = masked_loss(loss, mask)

        latent_estimates, history = inversion.invert(generator, y_gt, crop_loss, batch_size=1, video=args.video)
        # Get Images
        y_estimate_list = torch.split(torch.clamp(_tanh_to_sigmoid(generator(latent_estimates)), min=0., max=1.).cpu(), 1, dim=0)
        # Save
        for img_id, image in enumerate(images):
            y_RGB = Tensor2PIL(image_tensor_list[img_id])
            y_RGB.save(args.outputs + '/GT/%d%s' % (i, image_name_list[img_id][-4:]))
                            
            y_gt_pil = Tensor2PIL(mask_images(image_tensor_list[img_id], mask_cpu))
            y_estimate_pil = Tensor2PIL(y_estimate_list[img_id])
            y_estimate_pil.save(os.path.join(args.outputs + '/mGANoutput/%d%s' % (i, image_name_list[img_id][-4:])))
            
            y_gt_pil.save(os.path.join(args.outputs + '/input/%d%s' % (i, image_name_list[img_id][-4:])))
            # Create video
            if args.video:
                print('Create GAN-Inversion video.')
                video = cv2.VideoWriter(
                    filename=os.path.join(args.outputs, '%s_inpainting_%s.avi' % (image_name_list[img_id][:-4], os.path.split(args.mask[:-4])[1])),
                    fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                    fps=args.fps,
                    frameSize=(frameSize, frameSize))
                print('Save frames.')
                for i, sample in enumerate(history):
                    image = generator(sample)
                    image_cv2 = convert_array_to_images(image.detach().cpu().numpy())[0][:, :, ::-1]
                    video.write(image_cv2)
                video.release()

    print(f'{(time.time()-start_time)/60:.2f}','minutes taken in total;', f'{(time.time()-start_time)/60/len(image_list):.2f}', 'per image.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    # Image Path and Saving Path
    parser.add_argument('-i', '--target_images',
                        default='./examples/inpainting/church',
                        help='Directory with images for encoding')
    parser.add_argument('-m', '--mask',
                        default='mask_center.png',
                        help='Mask name (with masks saved in ./masks/')
    parser.add_argument('-o', '--outputs',
                        default='./inpainting_output',
                        help='Directory for storing generated images')
    # Parameters for Multi-Code GAN Inversion
    parser.add_argument('--inversion_type', default='PGGAN-Multi-Z',
                        help='Inversion type, PGGAN-Multi-Z for Multi-Code-GAN prior.')
    parser.add_argument('--composing_layer', default=4,
                        help='Composing layer in multi-code gan inversion methods.', type=int)
    parser.add_argument('--z_number', default=30,
                        help='Number of the latent codes.', type=int)
    # Loss Parameters
    parser.add_argument('--image_size', default=256,
                        help='Size of images for perceptual model', type=int)
    parser.add_argument('--loss_type', default='Combine',
                        help="['VGG', 'L1', 'L2', 'Combine']. 'Combine' means using L2 and Perceptual Loss.")
    parser.add_argument('--vgg_loss_type', default='L1',
                        help="['L1', 'L2']. The loss used in perceptual loss.")
    parser.add_argument('--vgg_layer', default=16,
                        help='The layer used in perceptual loss.', type=int)
    parser.add_argument('--l1_lambda', default=0.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L1 loss.", type=float)
    parser.add_argument('--l2_lambda', default=1.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L2 loss.", type=float)
    parser.add_argument('--vgg_lambda', default=0.1,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for Perceptual loss.", type=float)
    # Optimization Parameters
    parser.add_argument('--batch_size', default=1,
                        help='Size of images for perceptual loss.', type=int)
    parser.add_argument('--optimization', default='GD',
                        help="['GD', 'Adam']. Optimization method used.")  # inversion_type
    parser.add_argument('--init_type', default='Normal',
                        help="['Zero', 'Normal']. Initialization method. Using zero init or Gaussian random vector.")
    parser.add_argument('--lr', default=1.,
                        help='Learning rate.', type=float)
    parser.add_argument('--iterations', default=3000,
                        help='Number of optimization steps.', type=int)

    # Generator Setting
    parser.add_argument('--gan_model', default='pggan_churchoutdoor', help='The name of model used.', type=str)

    # Video Settings
    parser.add_argument('--video', type=bool, default=False, help='Save video. False for no video.')
    parser.add_argument('--fps', type=int, default=24, help='Frame rate of the created video.')

    # Randomize mask
    parser.add_argument('--varmask', type=bool, default=False, help='Randomize the mask if True. Each mask is saved with image ID.')
    parser.add_argument('--min_p', type=int, default=2, help='Minimum number of patches in a mask.')
    parser.add_argument('--max_p', type=int, default=4, help='Maximum number of patches in a mask.')
    parser.add_argument('--width_mean', type=int, default=64, help='Mean of width/height of a mask patch (Gauss).')
    parser.add_argument('--width_var', type=int, default=32, help='Variance of width/height of a mask patch (Gauss).')
        
    args, other_args = parser.parse_known_args()
    ### RUN
    main(args)
