import torch

from imageio import imread, imsave
from skimage.transform import resize
import numpy as np
import os
from path import Path
import argparse
from tqdm import tqdm

from models import DispNetS
from models import SemDispNetS
from utils.common import tensor2array

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main(args):
    
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return
    
    if args.with_semantics:
        disp_net = SemDispNetS().to(device)
    else:
        disp_net = DispNetS().to(device)
        
    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    base_output_dir = Path(args.output_dir)
    
    dirs = [dataset_dir/d for d in os.listdir(dataset_dir)]
    for dir_ in dirs:
        print(f"on dir {dir_}")
        
        episodes = os.listdir(dir_)
    
        for i, ep in enumerate(episodes):
            print(f"on episode {ep}")
            
            if i > args.num_episodes:
                break
            
            output_dir = base_output_dir/dir_/ep
            output_dir.makedirs_p()
            
            input_dir = dir_/ep/'rgb'
            test_files = sum([list(input_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])
            
            for file in tqdm(test_files):

                img = imread(file)
                h,w,_ = img.shape
                if (not args.no_resize) and (h != args.img_height or w != args.img_width):
                    img = resize(img, (args.img_height, args.img_width))
                img = np.transpose(img, (2, 0, 1))

                tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
                tensor_img = ((tensor_img - 0.5)/0.5).to(device)

                output = disp_net(tensor_img)[0]

                file_path, file_ext = file.relpath(args.dataset_dir).splitext()
                file_name = '-'.join(file_path.splitall()[1:])

                if args.output_disp:
                    disp = (255 * tensor2array(
                        output, max_value=None, colormap='bone')).astype(np.uint8)
                    imsave(
                        output_dir/'disp_{}_{}'.format(file_name, file_ext), 
                        np.transpose(disp, (1,2,0)))
                    
                if args.output_depth:
                    depth = 1/output
                    depth = (255 * tensor2array(
                        depth, max_value=10, colormap='rainbow')).astype(np.uint8)
                    imsave(output_dir/'depth_{}_{}'.format(
                        file_name, file_ext), np.transpose(depth, (1,2,0)))

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Inference script for the MP3D Sfm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--output-disp", action='store_true', 
        help="save disparity img")
    parser.add_argument("--output-depth", action='store_true', 
        help="save depth img")
    parser.add_argument("--with-semantics", action='store_true', 
        help='use network with semantics')
    parser.add_argument("--pretrained", required=True, type=str, 
        help="pretrained DispNet path")
    parser.add_argument("--img-height", default=128, type=int, 
        help="Image height")
    parser.add_argument("--img-width", default=416, type=int, 
        help="Image width")
    parser.add_argument("--no-resize", action='store_true', 
        help="no resizing is done")
    parser.add_argument("--num-episodes", type=int, default=2, 
        help="number of episodes to run")
    parser.add_argument("--dataset-list", default=None, type=str, 
        help="Dataset list file")
    parser.add_argument("--dataset-dir", default='.', type=str, 
        help="Dataset directory")
    parser.add_argument("--output-dir", default='output', type=str,  
        help="Output directory")
    parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', 
        type=str, help="images extensions to glob")

    args = parser.parse_args()
        
    main(args)
