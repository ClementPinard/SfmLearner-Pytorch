import enum
import os
import numpy as np

dirs = [
    'train', 
    'val_seen', 
    'val_unseen'
]

for dir_ in dirs:
    print(f"Current directory: {dir_}")
    scenes = os.listdir(dir_)
    n_scenes = len(scenes)
    print(f"\tNumber of scenes: {n_scenes}")
    
    episode_counter = []
    step_counter = []
    for i, scene in enumerate(scenes):
        scene_path = os.path.join(dir_, scene)
        episodes = os.listdir(scene_path)
        episode_counter.append(len(episodes))
        
        for j, ep in enumerate(episodes):
            episode_path = os.path.join(scene_path, ep, 'rgb')
            steps = os.listdir(episode_path)
            step_counter.append(len(steps)-1)
        
    epc = np.asarray(episode_counter)
    print(f"\tEpisode statistics min: {epc.min()} max {epc.max()} avg {int(epc.mean())} tot: {epc.sum()}")
    trj = np.asarray(step_counter)
    print(f"\tStep statistics min: {trj.min()} max {trj.max()} avg {int(trj.mean())} tot: {trj.sum()}")
        
    #     for j, episode in enumerate(episodes):
    #         print(f"\t\t[{j+1}/{n_episodes}] - Current episode {episode}")
            
    #         episode_path = os.path.join(scene_path, episode, 'rgb')
            
    #         steps = os.listdir(episode_path)
    #         n_steps = len(steps)
    #         for k, step in enumerate(steps):
    #             print(f"\t\t\t[{k+1}/{n_steps}] - Current step {step}", end='\r')
                
    #             if ".txt" in step:
    #                 continue
                
    #             img_path = os.path.join(episode_path, step)
    #             img = imageio.imread(img_path)
                
    #             im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             imageio.imwrite(img_path, im_rgb)
           