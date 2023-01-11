import copy

import cv2
import gymnasium as gym
import numpy as np


class DeterministicReplayReset(gym.Wrapper):
    def __init__(self, env, snapshot_in_info=True):
        super().__init__(env)
        self.snapshot_in_info = snapshot_in_info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.snapshot = []
        if self.snapshot_in_info:
            info['snapshot'] = copy.copy(self.snapshot)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.snapshot.append(action)
        if self.snapshot_in_info:
            info['snapshot'] = copy.copy(self.snapshot)
        return obs, reward, terminated, truncated, info

    def restore_snapshot(self, snapshot, *args, **kwargs):
        obs, info = self.reset(*args, **kwargs) # call my own method, not self.env's
        for action in snapshot:
            obs, reward, terminated, truncated, info = self.step(action) # call my own method, not self.env's
        return obs, reward, terminated, truncated, info

def restore_snapshots(envs, snapshots, *args, **kwargs):
    obs, info = envs.reset(*args, **kwargs)
    for env, snapshot in zip(envs.envs, snapshots):
        obs, reward, terminated, truncated, info = env.restore_snapshot(snapshot)
    # for action in snapshot:
        # obs, reward, terminated, truncated, info = self.step(action) # call my own method, not self.env's
    return obs, reward, terminated, truncated, info

class Observation01(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.observation_space
        self.observation_space = gym.spaces.Box(obs_space.low, obs_space.high/255., obs_space.shape, dtype=np.float32)
    def observation(self, obs):
        return (obs/255.).astype(np.float32)

class OneLife(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminated = self.env.ale.lives() < 6
        return obs, reward, terminated, truncated, info

# class DeadScreen(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#     def reset(self):
#         return self.env.reset()
#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         if terminated or truncated:
#             obs = np.zeros_like(obs)
#         return obs, reward, terminated, truncated, info

def get_obs_cell(obs, latent_h=11, latent_w=8, latent_d=20, ret_tuple=True):
    # obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (latent_w, latent_h), interpolation=cv2.INTER_AREA)
    obs = (obs*latent_d).astype(np.uint8, casting='unsafe')
    return tuple(obs.flatten()) if ret_tuple else obs

class ImageCellInfo(gym.Wrapper):
    def __init__(self, env, latent_h=11, latent_w=8, latent_d=20):
        super().__init__(env)
        self.latent_h, self.latent_w, self.latent_d = latent_h, latent_w, latent_d

    def reset(self):
        obs, info = self.env.reset()
        info['cell_img'] = get_obs_cell(obs, self.latent_h, self.latent_w, self.latent_d, ret_tuple=False)
        info['cell'] = tuple(info['cell_img'].flatten())
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['cell_img'] = get_obs_cell(obs, self.latent_h, self.latent_w, self.latent_d, ret_tuple=False)
        info['cell'] = tuple(info['cell_img'].flatten())
        return obs, reward, terminated, truncated, info

class DecisionTransformerEnv(gym.Wrapper):
    def __init__(self, env, n_obs=4):
        super().__init__(env)
        self.n_obs = n_obs

    def reset(self):
        obs, info = self.env.reset()
        self.obs_list = [obs]
        self.action_list = []
        self.reward_list = []
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.obs_list.append(obs)
        self.action_list.append(action)
        self.reward_list.append(reward)

        self.obs_list = self.obs_list[-self.n_obs:]
        self.action_list = self.action_list[-self.n_obs:]
        self.reward_list = self.reward_list[-self.n_obs:]

        info['obs_list'] = np.stack(self.obs_list)
        info['action_list'] = np.array(self.action_list)
        info['reward_list'] = np.array(self.reward_list)

        return obs, reward, terminated, truncated, info
