import imageio
import os
import numpy as np

dirs = [
    'train', 
    # 'val_seen', 
    # 'val_unseen'
]

colors_to_label  = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)

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
            
            episode_path = os.path.join(scene_path, episode, 'semantics')
            
            steps = os.listdir(episode_path)
            n_steps = len(steps)
            for k, step in enumerate(steps):
                print(f"\t\t\t[{k+1}/{n_steps}] - Current step {step}", end='\r')
                
                if "npy" in step:
                    continue
                
                img_path = os.path.join(episode_path, step)
                img = imageio.imread(img_path)
                
                # import matplotlib.pyplot as plt
                semantic_mask = np.zeros((img.shape[:2]))
                for l in range(1, colors_to_label.shape[0]):
                    i, j, k = np.where(img[:, :] == colors_to_label[l])
                    semantic_mask[i, j] = l
                    
                    # plt.matshow(semantic_mask)
                    # plt.show()
                    # plt.close()
                mask_file = os.path.join(episode_path, step.split('.')[0])
                np.save(mask_file, semantic_mask)
                