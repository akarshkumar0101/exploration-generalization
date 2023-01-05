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

import gym


class ToTensorWrapper(gym.Wrapper):
    def __init__(self, env, device=None, dtype=None):
        super().__init__(env)
        self.device = device
        self.dtype = dtype

    def reset(self):
        obs, info = self.env.reset()
        obs = torch.as_tensor(obs, device=self.device, dtype=self.dtype)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, device=self.device, dtype=self.dtype)
        elif isinstance(obs, list):
            obs = [torch.as_tensor(o, device=self.device, dtype=self.dtype) for o in obs]
        elif isinstance(obs, tuple):
            obs = tuple(torch.as_tensor(o, device=self.device, dtype=self.dtype) for o in obs)
        elif isinstance(obs, dict):
            obs = obs.__class__({k: torch.as_tensor(v, device=self.device, dtype=self.dtype) for k, v in obs.items()})

        reward = torch.as_tensor(reward, device=self.device, dtype=self.dtype)
        terminated = torch.as_tensor(terminated, device=self.device, dtype=self.dtype)
        truncated = torch.as_tensor(truncated, device=self.device, dtype=self.dtype)
        return obs, reward, terminated, truncated, info

class DictObservationWrapper(gym.wrappers.TransformObservation):
    def __init__(self, env):
        super().__init__(env, lambda obs: OrderedDict(obs=obs))
        self.observation_space = gym.spaces.Dict(obs=self.env.observation_space)
