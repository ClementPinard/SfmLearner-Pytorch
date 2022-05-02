import torch

from imageio import imread, imsave, mimsave
import numpy as np
import os
from path import Path
import argparse
from tqdm import tqdm
from natsort import natsorted

from models import PoseExpNet, DispNetS, SemDispNetS
from utils.common import tensor2array
from utils.loss_functions import photometric_reconstruction_loss

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main(args):
    
    if args.use_pred_depth:
        if args.with_semantics:
            disp_net = SemDispNetS().to(device)
        else:
            disp_net = DispNetS().to(device)
        
        disp_weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(disp_weights['state_dict'])
        disp_net.eval()
    
    # pose network
    pose_net = PoseExpNet().to(device)
    weights = torch.load(args.pretrained_pose)
    pose_net.load_state_dict(weights['state_dict'])
    pose_net.eval()

    dataset_dir = Path(args.dataset_dir)
    base_output_dir = Path(args.output_dir)
    base_output_dir.makedirs_p()
    
    # intrinsic matrix
    hfov = 90
    K = np.array([
        [1 / np.tan(hfov / 2.), 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0.],
        [0., 0.,  1]]).astype(np.float32)
    K = torch.from_numpy(K).unsqueeze(0).to(device)
    
    dirs = [dataset_dir/d for d in os.listdir(dataset_dir)]
    for dir_ in dirs:
        print(f"on dir {dir_}")
        
        episodes = os.listdir(dir_)
    
        for i, ep in enumerate(episodes):
            print(f"on episode {ep}")
            
            if i > args.num_episodes:
                break
            
            rgb_input_dir = dir_/ep/'rgb'
            rgb_test_files = sum([list(
                rgb_input_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])
            rgb_test_files = natsorted(rgb_test_files)
            
            depth_input_dir = dir_/ep/'depth'
            depth_test_files = sum([list(
                depth_input_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])
            depth_test_files = natsorted(depth_test_files)

            img_list = []
            sequence = [-1, 1]
            for i in range(1, len(rgb_test_files)-1):   
                             
                # target image
                out_img = imread(rgb_test_files[i])
                img = np.transpose(out_img / 255., (2, 0, 1))
                tgt_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
                tgt_img = ((tgt_img - 0.5)/0.5).to(device)
                
                # source images
                seq_imgs = []
                ref_imgs = []
                for s in sequence:
                    simg_ = imread(rgb_test_files[i+s])
                    simg = np.transpose(simg_ / 255., (2, 0, 1))
                    simg = torch.from_numpy(simg.astype(np.float32)).unsqueeze(0)
                    simg = ((simg - 0.5)/0.5).to(device)
                    seq_imgs.append(simg_)
                    ref_imgs.append(simg)
                    
                # depth 
                if args.use_gt_depth:
                    # using ground truth depth 
                    depth = imread(depth_test_files[i])
                    depth = np.transpose(depth, (2, 0, 1))[0]
                    depth = torch.from_numpy(
                        depth.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    imn, imx = 0, 255
                    omn, omx = 1.5, 10.01
                    depth = omn + (((depth - imn) / (imx - imn)) * (omx - omn))
                    depth = 1./ depth
                    depth = depth.to(device)
                else:
                    # using predicted depth 
                    disp = disp_net(tgt_img)
                    depth = (1. / disp)            
                
                # pose
                expmask, pose = pose_net(tgt_img, ref_imgs)
                
                # warped
                _, warped, diff = photometric_reconstruction_loss(
                    tgt_img, ref_imgs, K, depth, expmask, pose, 'euler', 'zeros')
                
                warped = 255 * tensor2array(warped[0][1], max_value=1)
                warped = np.transpose(warped, (1, 2, 0)).astype(np.uint8)
                
                output_img = np.concatenate((out_img, warped, seq_imgs[1]), axis=1)
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
    parser.add_argument("--use-gt-depth", action='store_true', 
        help='use ground truth depth to test poses')
    parser.add_argument("--use-pred-depth", action='store_true',
        help='use predicted depth')
    parser.add_argument("--with-semantics", action='store_true', 
        help='Use SemDispNet or DispNet')
    parser.add_argument("--pretrained-disp", required=True, type=str, 
        help="pretrained PoseNet path")
    parser.add_argument("--pretrained-pose", required=True, type=str, 
        help="pretrained PoseNet path")
    parser.add_argument("--num-episodes", type=int, default=2, 
        help="number of episodes to run")
    parser.add_argument("--dataset-dir", default='.', type=str, 
        help="Dataset directory")
    parser.add_argument("--output-dir", default='output', type=str,  
        help="Output directory")
    parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', 
        type=str, help="images extensions to glob")

    args = parser.parse_args()
        
    main(args)