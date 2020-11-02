import os
import argparse

from PIL import Image

def main(input_dir, output_dir):
    """
    Convert all webp images from an input directory to 256x256 png images, saved in the output folder.
    
    Args:
        input_dir: input directory
        output_dir: output directory   
    """
    filenames = sorted(os.listdir(input_dir))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in filenames:
        img_name = os.path.splitext(filename)[0]
        img = Image.open(os.path.join(input_dir,filename)).convert("RGB")
        img = img.resize((256,256),Image.BICUBIC)
        save_loc = os.path.join(output_dir,img_name)+".png"
        img.save(save_loc,"png")
        print("Saved image to {0}".format(save_loc))
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='webptopng', usage='python3 %(prog)s.py -input_dir [input folder] -output_dir [output folder]', description='Convert webp images to png')
    parser.add_argument('-input_dir', type=str, required=True, help='input directory')
    parser.add_argument('-output_dir', type=str, required=True, help='output directory')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)