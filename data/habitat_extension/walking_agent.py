
import json
from collections import defaultdict
import os
import numpy as np

np.set_printoptions(suppress=True)
from habitat import Env
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import trange

# Other stuff
from habitat_extension.shortest_path_follower import ShortestPathFollowerCompat
from habitat_sim.utils.common import d3_40_colors_rgb

def parse_env_id(scene_path):
    fname = os.path.basename(scene_path)
    env_id, _ = os.path.splitext(fname)
    return env_id

from habitat.core.utils import try_cv2_import

cv2 = try_cv2_import()

def walk(config: Config) -> None:
    split = config.EVAL.SPLIT
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.freeze()

    env = Env(config=config.TASK_CONFIG)
    env_id = parse_env_id(config.TASK_CONFIG.SIMULATOR.SCENE)

    agent = WalkingAgent()
    
    base_path = f"data/mp3d_sfm/{split}"
    os.makedirs(base_path, exist_ok=True)
    
    stats = defaultdict(float)
    prev_start = np.array([0.0, 0.0, 0.0])
    prev_goal = np.array([0.0, 0.0, 0.0])
    episodes = []
    for i, ep in enumerate(env.episodes):
        if split == 'train' and i == 6512:
            continue
        elif split == 'val_seen' and i == 427:
            continue
        episodes.append(ep)
        
    env.episodes = episodes
    num_episodes = min(config.EVAL.EPISODE_COUNT, len(env.episodes))
    
    for t in trange(num_episodes):
        
        obs = env.reset()
        curr_episode = env.current_episode
        start = np.array(curr_episode.start_position)
        goal = np.array(curr_episode.goals[0].position) 
        if np.array_equal(start, prev_start) and np.array_equal(goal, prev_goal):
            continue
        prev_start = start
        prev_goal = goal
        
        scene_id = curr_episode.scene_id.split('/')[-1].split('.')[0]
        episode_id = curr_episode.episode_id
        episode_path = os.path.join(base_path, f"{scene_id}/{episode_id}")
        os.makedirs(episode_path, exist_ok=True)
        
        agent.reset(env.sim, curr_episode.goals[0].position)
        agent.add_step(obs,  env.get_metrics())

        while not env.episode_over:
            action = agent.act()
            obs = env.step(action)
            info = env.get_metrics()
            agent.add_step(obs, info)

        agent.save(episode_path)


class WalkingAgent(Agent):
    """Agent navigates along the ground-truth path and gathers semantic 
    information. """
    def __init__(self, env_id=None):
        self.reset()
        self.frames = []
        
        # semantics available in the semantic sensor
        with open('data/category_mapping.json') as json_file:
            id2label_map = json.load(json_file)
            
            self.id2label = id2label_map['all_categories']
            self.invalid_labels = id2label_map['categories_to_ignore']


    def reset(self, sim=None, goal=None):
        self.forward_steps = 37
        self.turns = np.random.randint(0, int(360 / 15) + 1)
        
        self.rgb = []
        self.depth = []
        self.semantics = []
        self.semantic_mask = []
        
        self.actions = []
        self.positions = []
        
        self.goal = goal
        self.follower = None

        
        # has to be initialized for every scene
        if not sim == None:
            self.follower = ShortestPathFollowerCompat(
                sim, goal_radius=0.5, return_one_hot=False)
            
            scene = sim.semantic_annotations()
            instance_id_to_label_id = {
                int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
            self.mapping = np.array(
                [instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id))])
    
    def add_step(self, observation, info = {}):
        """Generate image of single frame from observation and info
        returned from a single environment step().

        Args:
            observation: observation returned from an environment step().
            info: info returned from an environment step().

        Returns:
            generated image of a single frame.
        """
        observation_size = -1
        if "rgb" in observation:
            observation_size = observation["rgb"].shape[0]
            rgb = observation["rgb"][:, :, :3]
            self.rgb.append(rgb)

        if "semantic" in observation:
            h, w = observation["semantic"].shape
            semantic_img = np.take(self.mapping, observation["semantic"]).flatten() - 1
            sem_mask = semantic_img.reshape(h, w)
            self.semantic_mask.append(sem_mask)
            
            semantic_img[semantic_img == -1] = 0
            colored_sem = d3_40_colors_rgb[semantic_img].reshape(h, w, 3)
            colored_sem = cv2.resize(
                colored_sem,
                dsize=(observation_size, observation_size),
                interpolation=cv2.INTER_CUBIC,
            )
            self.semantics.append(colored_sem)
            
        # draw depth map if observation has depth info. resize to rgb size.
        if "depth" in observation:
            if observation_size == -1:
                observation_size = observation["depth"].shape[0]
            depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
            depth_map = np.stack([depth_map for _ in range(3)], axis=2)
            depth_map = cv2.resize(
                depth_map,
                dsize=(observation_size, observation_size),
                interpolation=cv2.INTER_CUBIC,
            )
            self.depth.append(depth_map)
            
    def act(self):
        if not self.follower == None:
            action = self.follower.get_next_action(self.goal)
            if action == None:
                action = HabitatSimActions.STOP
        else:
            if self.turns > 0:
                self.turns -= 1
                action = HabitatSimActions.TURN_RIGHT
            if self.forward_steps > 0:
                self.forward_steps -= 1
                action = HabitatSimActions.MOVE_FORWARD

            action = HabitatSimActions.STOP

        if action == HabitatSimActions.STOP:
            euler_action = [0.0,0.0,0.0,0.0,0.0,0.0]
        elif action == HabitatSimActions.MOVE_FORWARD:
            euler_action = [-0.25,0.0,0.0,0.0,0.0,0.0]
        elif action == HabitatSimActions.TURN_RIGHT:
            euler_action = [0.0,0.0,0.0,0.0,0.0,0.26]
        elif action == HabitatSimActions.TURN_LEFT:
            euler_action = [0.0,0.0,0.0,0.0,0.0,-0.26]
            
        self.actions.append(euler_action)
        return {"action": action}

    def save(self, path):
        frames = [self.rgb, self.depth, self.semantics]
        
        for i, d in enumerate(["rgb", "depth", "semantics"]):
            subdir = os.path.join(path, d)
            os.makedirs(subdir, exist_ok=True)
            
            for j, frame in enumerate(frames[i]):
                cv2.imwrite(f"{subdir}/{j}.png", frame)
                
        subdir = os.path.join(path, 'semantics')
        for j, mask in enumerate(self.semantic_mask):
            np.save(f"{subdir}/{j}.npy", mask)
        
        poses = np.array(self.actions)
        np.save(f"{path}/poses.npy", poses)