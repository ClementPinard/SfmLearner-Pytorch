import argparse
import csv
import numpy as np
import os
import torch
from imageio import imread
from path import Path
from natsort import natsorted

from models import PoseExpNet, DispNetS, SemDispNetS
from utils.common import array2tensor

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main(args):
    # If using predicted depth for testing the pose network
    if args.with_semantics:
        disp_net = SemDispNetS().to(device)
    else:
        disp_net = DispNetS().to(device)
    
    disp_weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(disp_weights['state_dict'])
    disp_net.eval()
    
    # Path
    dataset_dir = Path(args.dataset_dir)
    base_output_dir = Path(args.output_dir)
    base_output_dir.makedirs_p()
    
    error_names = ['abs_diff', 'abs_rel','sq_rel','rms','log_rms', 'abs_log', 
        'a1', 'a2', 'a3']
    
    with open(base_output_dir/args.error_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(error_names)

    with open(base_output_dir/args.error_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(error_names)
    
    dirs = [dataset_dir/d for d in os.listdir(dataset_dir)]
    images_considered = 0 
    errors = []
    for dir_ in dirs:
        print(f"on dir {dir_}")
        
        episodes = os.listdir(dir_)
    
        for i, ep in enumerate(episodes):
            if i > args.num_episodes:
                break
            print(f"on episode {ep}")
            
            rgb_input_dir = dir_/ep/'rgb'
            rgb_test_files = natsorted(rgb_input_dir.walkfiles('*.png'))
            
            depth_input_dir = dir_/ep/'depth'
            depth_test_files = natsorted(depth_input_dir.walkfiles('*.png'))
            
            img_list = []
            sequence = [-1, 1]
            for i in range(1, len(rgb_test_files)-1):   
                             
                # target image
                out_img = imread(rgb_test_files[i])
                tgt_img = array2tensor(out_img).to(device)
                    
                # target depth   
                gt_depth = imread(depth_test_files[i])
                gt_depth = np.transpose(gt_depth/255., (2, 0, 1))[0].astype(np.float32)
                gt_depth = gt_depth.clip(0.01, 1.0)
                
                # predicted depth 
                disp = disp_net(tgt_img).cpu().numpy()[0,0]
                pred_depth = (1. / disp).clip(0.01, 10.0)          
                
                scale_factor = np.median(gt_depth) / np.median(pred_depth)
                error = compute_errors(gt_depth, pred_depth*scale_factor)
                images_considered += 1
                errors.append(error)
                
                with open(base_output_dir/args.error_full, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow(error)
        
    mean_errors = np.array(errors).mean(axis=0)
    std_errors = np.array(errors).std(axis=0)
    with open(base_output_dir/args.error_summary, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([f"Mean"])
        writer.writerow(mean_errors)
        writer.writerow([f"Standard deviation"])
        writer.writerow(std_errors)
        writer.writerow([f"Images considerered: {images_considered}"])

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_log = np.mean(np.abs(np.log(gt) - np.log(pred)))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_diff = np.mean(np.abs(gt - pred))

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_diff, abs_rel, sq_rel, rmse, rmse_log, abs_log, a1, a2, a3

                   
            
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pose inference script for the MP3D Sfm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # I/O
    parser.add_argument("--pretrained-disp", required=False, type=str, 
        help="pretrained PoseNet path")
    parser.add_argument("--dataset-dir", default='./data/mp3d_sfm/val_unseen', 
        type=str, help="Dataset directory")
    parser.add_argument("--output-dir", default='output', type=str,  
        help="Output directory")
    parser.add_argument("--error-summary", default='error_summary.csv', type=str,
        help='file to save computed errors')
    parser.add_argument("--error-full", default='error_full.csv', type=str,
        help='file to save computed errors')
    
    # Other parameters
    parser.add_argument("--with-semantics", action='store_true', 
        help='Use SemDispNet or DispNet')
    parser.add_argument("--num-episodes", default=10, type=int,
        help="number of episodes to run")
    
    args = parser.parse_args()
    main(args)