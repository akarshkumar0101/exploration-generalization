import numpy as np
import torch


def count_params(net):
    return np.sum([p.numel() for p in net.parameters()])

def normalize(a, devs=None):
    b = a
    if devs is not None:
        # mask = ((a-a.mean())/a.std()).abs()<devs
        mask = ((a-a.median())/a.std()).abs()<devs
        b = a[mask]
    return (a-b.mean())/(a.std()+1e-9)



from collections import OrderedDict

import gymnasium as gym


class ToTensor(gym.Wrapper):
    def __init__(self, env, device=None, dtype=None):
        super().__init__(env)
        self.device = device
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

    def reset(self):
        obs, info = self.env.reset()
        obs = self.as_tensor(obs)
        return obs, info
    
    def step(self, action):
        action = action.tolist()
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self.as_tensor(obs)
        reward = torch.as_tensor(reward, device=self.device, dtype=self.dtype)
        terminated = torch.as_tensor(terminated, device=self.device, dtype=self.dtype)
        truncated = torch.as_tensor(truncated, device=self.device, dtype=self.dtype)
        return obs, reward, terminated, truncated, info

class DictObservation(gym.wrappers.TransformObservation):
    def __init__(self, env):
        super().__init__(env, lambda obs: OrderedDict(obs=obs))
        self.observation_space = gym.spaces.Dict(obs=self.env.observation_space)
