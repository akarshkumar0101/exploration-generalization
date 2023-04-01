import gymnasium as gym
import numpy as np
import torch


class ToTensor(gym.Wrapper):
    def __init__(self, env, device=None, dtype=None):
        super().__init__(env)
        self.device = device
        self.dtype = dtype

    def set_device(self, device):
        self.device = device

    def set_dtype(self, dtype):
        self.dtype = dtype
    
    def as_tensor(self, obs):
        if isinstance(obs, torch.Tensor):
            return obs.to(device=self.device, dtype=self.dtype)
        elif isinstance(obs, np.ndarray):
            return torch.as_tensor(obs, device=self.device, dtype=self.dtype)
        elif isinstance(obs, list):
            return [self.as_tensor(o) for o in obs]
        elif isinstance(obs, tuple):
            return tuple(self.as_tensor(o) for o in obs)
        elif isinstance(obs, dict):
            return obs.__class__({k: self.as_tensor(v) for k, v in obs.items()})

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        obs = self.as_tensor(obs)
        return obs, info

    def restore_snapshot(self, snapshots, *args, **kwargs):
        assert isinstance(snapshots, list)
        obs, reward, terminated, truncated, info = self.env.restore_snapshot(snapshots, *args, **kwargs)
        obs = self.as_tensor(obs)
        reward = self.as_tensor(reward)
        terminated = self.as_tensor(terminated)
        truncated = self.as_tensor(truncated)
        return obs, reward, terminated, truncated, info
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.tolist()
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.as_tensor(obs)
        reward = self.as_tensor(reward)
        terminated = self.as_tensor(terminated)
        truncated = self.as_tensor(truncated)
        return obs, reward, terminated, truncated, info