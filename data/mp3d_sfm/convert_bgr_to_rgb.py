import imghdr
import imageio
import os
import numpy as np
import cv2

dirs = [
    'train', 
    # 'val_seen', 
    # 'val_unseen'
]

for dir_ in dirs:
    print(f"Current directory: {dir_}")
    scenes = os.listdir(dir_)
    n_scenes = len(scenes)
    
    for i, scene in enumerate(scenes):
        print(f"\t[{i+1}/{n_scenes}] - Current scene {scene}")
        scene_path = os.path.join(dir_, scene)
        
        episodes = os.listdir(scene_path)
        n_episodes = len(episodes)
        
        for j, episode in enumerate(episodes):
            print(f"\t\t[{j+1}/{n_episodes}] - Current episode {episode}")
            
            episode_path = os.path.join(scene_path, episode, 'rgb')
            
            steps = os.listdir(episode_path)
            n_steps = len(steps)
            for k, step in enumerate(steps):
                print(f"\t\t\t[{k+1}/{n_steps}] - Current step {step}", end='\r')
                
                if ".txt" in step:
                    continue
                
                img_path = os.path.join(episode_path, step)
                img = imageio.imread(img_path)
                
                im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imageio.imwrite(img_path, im_rgb)
                
                # import matplotlib.pyplot as plt 
                # plt.matshow(im_rgb)
                # plt.show()
                # plt.close()
              