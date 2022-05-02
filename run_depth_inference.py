import torch
import numpy as np
import os
import argparse

from path import Path
from imageio import imread, mimsave
from tqdm import tqdm
from natsort import natsorted
from models import DispNetS, SemDispNetS
from utils.common import tensor2array, array2tensor

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main(args):
    assert args.with_disp or args.with_depth, f"Need specify DISP or DEPTH"
    
    if args.with_semantics:
        disp_net = SemDispNetS().to(device)
    else:
        disp_net = DispNetS().to(device)
        
    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()
    
    dataset_dir = Path(args.dataset_dir)
    base_output_dir = Path(args.output_dir)
    base_output_dir.makedirs_p()
    
    dirs = [dataset_dir/d for d in os.listdir(dataset_dir)]
    for dir_ in dirs:
        print(f"on dir {dir_}")
        
        episodes = os.listdir(dir_)
    
        for i, ep in enumerate(episodes):
            print(f"on episode {ep}")
            
            if i > args.num_episodes:
                break
            
            input_dir = dir_/ep/'rgb'
            test_files = natsorted(input_dir.walkfiles('*.png'))
           
            img_list = []
            for file in tqdm(test_files):

                file_path, _ = file.relpath(args.dataset_dir).splitext()
                file_name = '-'.join(file_path.splitall()[1:])
                
                out_img = imread(file)
                tgt_img = array2tensor(out_img).to(device)
                
                disp = disp_net(tgt_img)
                if args.with_disp:
                    disp_to_show = tensor2array(
                        disp[0], max_value=None, colormap='magma')
                    disp_to_show = (255 * disp_to_show).astype(np.uint8)
                    disp_to_show = np.transpose(disp_to_show, (1, 2, 0))
                    out_img = np.concatenate((out_img, disp_to_show), axis=1)
                
                if args.with_depth:
                    depth = (1. / disp)
                    depth = tensor2array(
                        depth[0], max_value=None, colormap='rainbow')
                    depth = (255 * depth).astype(np.uint8)
                    depth = np.transpose(depth, (1, 2, 0))
                    out_img = np.concatenate((out_img, depth), axis=1)
                
                img_list.append(out_img)
            
            mimsave(base_output_dir/'ep-{}_{}.gif'.format(ep, file_name), 
                    img_list, duration=0.5)            
            
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Depth inference script for the MP3D Sfm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # I/O
    parser.add_argument("--pretrained", required=True, type=str, 
        help="path to pretrained DispNet")
    parser.add_argument("--dataset-dir", default='./data/mp3d_sfm/val_unseen', 
        type=str, help="Dataset directory")
    parser.add_argument("--output-dir", default='output', type=str,  
        help="Output directory")
    
    # Other parameters
    parser.add_argument("--with-disp", action='store_true', 
        help='append disparities to output video')
    parser.add_argument("--with-depth", action='store_true', 
        help='append depth to output video')
    parser.add_argument("--with-semantics", action='store_true', 
        help='use network with semantics')
    parser.add_argument("--num-episodes", type=int, default=2, 
        help="number of episodes to run")

    args = parser.parse_args()
        
    main(args)
