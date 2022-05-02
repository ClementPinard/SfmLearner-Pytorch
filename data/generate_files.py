import os
import numpy as np

dirs = ['train', 'val_seen', 'val_unseen']

hfov = 90
K = np.array([
    [1 / np.tan(hfov / 2.), 0., 0.],
    [0., 1 / np.tan(hfov / 2.), 0.],
    [0., 0.,  1]])

for dir_ in dirs:
    f = open(f"{dir_}.txt", "w")
    
    scenes = os.listdir(dir_)
    for scene in scenes:
        scene_path = os.path.join(dir_, scene)
        
        episodes = os.listdir(scene_path)
        for episode in episodes:
            episode_path = os.path.join(scene_path, episode, 'rgb')
            f.write(episode_path+'\n')
            
            camera_file = os.path.join(episode_path, 'cam.txt')
            np.savetxt(camera_file, K)