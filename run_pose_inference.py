import argparse
import numpy as np
import os
import torch

from imageio import imread, mimsave
from path import Path
from natsort import natsorted

from models import PoseExpNet, DispNetS, SemDispNetS
from utils.common import convert_depth, tensor2array, array2tensor
from utils.loss_functions import photometric_reconstruction_loss

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main(args):
    # If using predicted depth for testing the pose network
    if args.use_pred_depth:
        if args.with_semantics:
            disp_net = SemDispNetS().to(device)
        else:
            disp_net = DispNetS().to(device)
        
        disp_weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(disp_weights['state_dict'])
        disp_net.eval()
    
    # Initialize pose network
    pose_net = PoseExpNet(output_exp=args.use_exp_mask).to(device)
    weights = torch.load(args.pretrained_pose)
    pose_net.load_state_dict(weights['state_dict'])
    pose_net.eval()
    
    # Path
    dataset_dir = Path(args.dataset_dir)
    base_output_dir = Path(args.output_dir)
    base_output_dir.makedirs_p()
    
    # Load intrinsic matrix
    K = np.genfromtxt(args.intrinsics).astype(np.float32).reshape((3, 3))
    K = torch.from_numpy(K).unsqueeze(0).to(device)
    
    dirs = [dataset_dir/d for d in os.listdir(dataset_dir)]
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
                
                # source images
                seq_imgs = []
                ref_imgs = []
                for s in sequence:
                    simg_ = imread(rgb_test_files[i+s])
                    simg = array2tensor(simg_).to(device)
                    seq_imgs.append(simg_)
                    ref_imgs.append(simg)
                    
                # depth 
                if args.use_gt_depth:
                    # using ground truth depth 
                    depth = imread(depth_test_files[i])
                    depth = convert_depth(depth).to(device)
                else:
                    # using predicted depth 
                    disp = disp_net(tgt_img)
                    depth = (1. / disp)            
                
                # pose
                expmask, pose = pose_net(tgt_img, ref_imgs)
                
                # warped
                _, warped, _ = photometric_reconstruction_loss(
                    tgt_img, ref_imgs, K, depth, expmask, pose, 'euler', 'zeros')
                
                warped = 255 * tensor2array(warped[0][1], max_value=1)
                warped = np.transpose(warped, (1, 2, 0)).astype(np.uint8)
                
                if expmask is not None:
                    expmask = expmask.squeeze(0)[1]
                    expmask = 255 * tensor2array(expmask, max_value=1)
                    expmask = np.transpose(expmask, (1, 2, 0)).astype(np.uint8)
                    
                    exp_mask_ext = np.zeros(warped.shape).astype(np.uint8)
                    exp_mask_ext[:] = expmask
                    
                    output_img = np.concatenate(
                        (out_img, expmask, warped, seq_imgs[1]), axis=1)
                else:
                    output_img = np.concatenate(
                        (out_img, warped, seq_imgs[1]), axis=1)
                    
                img_list.append(output_img)
                
                file_path, _ = rgb_test_files[i].relpath(args.dataset_dir).splitext()
                file_name = '-'.join(file_path.splitall()[1:])
                
            mimsave(base_output_dir/'ep-{}_{}.gif'.format(ep, file_name), 
                img_list, duration=2.0)
                
            
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pose inference script for the MP3D Sfm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # I/O
    parser.add_argument("--pretrained-pose", required=True, type=str, 
        help="pretrained PoseNet path")
    parser.add_argument("--pretrained-disp", required=False, type=str, 
        help="pretrained PoseNet path")
    parser.add_argument("--intrinsics", default='./data/mp3d_sfm/cam.txt', 
        type=str, help='path to intrinsics matrix')
    parser.add_argument("--dataset-dir", default='./data/mp3d_sfm/val_unseen', 
        type=str, help="Dataset directory")
    parser.add_argument("--output-dir", default='output', type=str,  
        help="Output directory")
    
    # Other parameters
    parser.add_argument("--use-exp-mask", action='store_true', 
        help='uses explainability mask')
    parser.add_argument("--use-gt-depth", action='store_true', 
        help='use ground truth depth to test poses')
    parser.add_argument("--use-pred-depth", action='store_true',
        help='use predicted depth')
    parser.add_argument("--with-semantics", action='store_true', 
        help='Use SemDispNet or DispNet')
    parser.add_argument("--num-episodes", default=2, type=int,
        help="number of episodes to run")
    
    args = parser.parse_args()
    main(args)