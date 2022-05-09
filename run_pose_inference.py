import argparse
import numpy as np
import os
import torch

from imageio import imread, mimsave
from path import Path
from natsort import natsorted

from models import PoseExpNet, DispNetS, SemDispNetS, SemPoseExpNet
from utils.common import convert_depth, tensor2array, array2tensor
from utils.loss_functions import photometric_reconstruction_loss, semantic_reconstruction_loss

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_ref_images(files, index, sequence = [-1, 1]):
    # source images
    np_imgs = []
    tensor_imgs = []
    for s in sequence:
        simg_ = imread(files[index+s])
        simg = array2tensor(simg_).to(device)
        np_imgs.append(simg_)
        tensor_imgs.append(simg)
    return np_imgs, tensor_imgs
                    
def run(args, K, pose_net, disp_net = None):
    # If using predicted depth for testing the pose network
    if args.use_pred_depth:
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
            for i in range(1, len(rgb_test_files)-1):   
                # target image
                out_img = imread(rgb_test_files[i])
                tgt_img = array2tensor(out_img).to(device)
                
                # source images
                seq_imgs, ref_imgs = get_ref_images(rgb_test_files, i)
                
                # depth 
                if args.use_gt_depth:
                    # using ground truth depth 
                    depth = imread(depth_test_files[i])
                    depth = convert_depth(depth).to(device)
                else:
                    disp = disp_net(tgt_img)
                    expmask, pose = pose_net(tgt_img, ref_imgs)
                    
                depth = (1. / disp)   
                             
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

def run_with_semantics(args, K, pose_net, disp_net = None):
    # Path
    dataset_dir = Path(args.dataset_dir)
    base_output_dir = Path(args.output_dir)
    base_output_dir.makedirs_p()
    
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
            
            sem_input_dir = dir_/ep/'semantics'
            sem_test_files = natsorted(sem_input_dir.walkfiles('*.png'))
            
            depth_input_dir = dir_/ep/'depth'
            depth_test_files = natsorted(depth_input_dir.walkfiles('*.png'))
            
            img_list, sem_img_list = [], []
            for i in range(1, len(rgb_test_files)-1):         
                # target image
                out_img = imread(rgb_test_files[i])
                tgt_img = array2tensor(out_img).to(device)
                
                sem_img = imread(sem_test_files[i])
                sem_tgt_img = array2tensor(sem_img).to(device)
                
                # source images
                seq_imgs, ref_imgs = get_ref_images(rgb_test_files, i)
                sem_seq_imgs, sem_ref_imgs = get_ref_images(sem_test_files, i)
    
                # depth 
                if args.use_gt_depth:
                    # using ground truth depth 
                    depth = imread(depth_test_files[i])
                    depth = convert_depth(depth).to(device)
                else:
                    # using predicted depth 
                    disp = disp_net(tgt_img, sem_tgt_img)
                    depth = (1. / disp)   
                
                expmask, pose = pose_net(
                    tgt_img, ref_imgs, sem_tgt_img, sem_ref_imgs)
                 
                # warps  
                _, sem_warped, _ = semantic_reconstruction_loss(
                    sem_tgt_img, sem_ref_imgs, K, depth, expmask, pose, 
                    'euler', 'zeros')
                             
                # warped
                _, warped, _ = photometric_reconstruction_loss(
                    tgt_img, ref_imgs, K, depth, expmask, pose, 'euler', 'zeros')
                
                warped = 255 * tensor2array(warped[0][1], max_value=1)
                warped = np.transpose(warped, (1, 2, 0)).astype(np.uint8)
                
                sem_warped = 255 * tensor2array(sem_warped[0][1], max_value=1)
                sem_warped = np.transpose(sem_warped, (1, 2, 0)).astype(np.uint8)
                
                if expmask is not None:
                    expmask = expmask.squeeze(0)[1]
                    expmask = 255 * tensor2array(expmask, max_value=1)
                    expmask = np.transpose(expmask, (1, 2, 0)).astype(np.uint8)
                    
                    exp_mask_ext = np.zeros(warped.shape).astype(np.uint8)
                    exp_mask_ext[:] = expmask
                    
                    output_img = np.concatenate(
                        (out_img, expmask, warped, seq_imgs[1]), axis=1)
                    
                    sem_output_img = np.concatenate(
                        (sem_img, expmask, sem_warped, sem_seq_imgs[1]), axis=1)
                else:
                    output_img = np.concatenate(
                        (out_img, warped, seq_imgs[1]), axis=1)
                    
                    sem_output_img = np.concatenate(
                        (sem_img, sem_warped, sem_seq_imgs[1]), axis=1)
                    
                img_list.append(output_img)
                sem_img_list.append(sem_output_img)
                
                file_path, _ = rgb_test_files[i].relpath(args.dataset_dir).splitext()
                file_name = '-'.join(file_path.splitall()[1:])
                
            mimsave(base_output_dir/'ep-{}_{}_rgb.gif'.format(ep, file_name), 
                img_list, duration=2.0)
            
            mimsave(base_output_dir/'ep-{}_{}_sem.gif'.format(ep, file_name), 
                sem_img_list, duration=2.0)
            
@torch.no_grad()
def main(args):
    disp_net = None
    if args.with_semantics:
        disp_net = SemDispNetS().to(device)
        pose_net = SemPoseExpNet(output_exp=args.use_exp_mask).to(device)
    else:
        disp_net = DispNetS().to(device)
        pose_net = PoseExpNet(output_exp=args.use_exp_mask).to(device)
        
    disp_weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(disp_weights['state_dict'])
    disp_net.eval()

    weights = torch.load(args.pretrained_pose)
    pose_net.load_state_dict(weights['state_dict'])
    pose_net.eval()
    
    # Load intrinsic matrix
    K = np.genfromtxt(args.intrinsics).astype(np.float32).reshape((3, 3))
    K = torch.from_numpy(K).unsqueeze(0).to(device)
    
    if args.with_semantics:
        run_with_semantics(args, K, pose_net, disp_net)
    else: 
        run(args, K, pose_net, disp_net) 
            
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