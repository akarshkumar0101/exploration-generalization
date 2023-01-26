import copy

import cv2
import gymnasium as gym
import numpy as np
import torch


class MRDomainCellInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.y, self.x = 0, 0
        self.roomx, self.roomy = 0, 0
        self.inventory = None

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        self.y, self.x = self.get_agent_yx(info['obs_ori'])
        self.roomx, self.roomy = 0, 0
        self.inventory = None

        self.update_info(info['obs_ori'], info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.update_info(info['obs_ori'], info)
        return obs, reward, terminated, truncated, info

    def update_info(self, obs, info):
        y, x = self.get_agent_yx(obs)
        if (x-self.x)>5:
            self.roomx -= 1
        elif (x-self.x)<-5:
            self.roomx += 1
        if (y-self.y)>5:
            self.roomy -= 1
        elif (y-self.y)<-5:
            self.roomy += 1
        self.y, self.x = y, x
        self.inventory = self.get_cell_inventory(obs)
        info['cell'] = (self.y, self.x, self.roomy, self.roomx)+tuple(self.inventory.flatten())
        info['y'], info['x'] = self.y, self.x
        info['roomy'], info['roomx'] = self.roomy, self.roomx
        info['inventory'] = self.inventory

    def get_agent_yx(self, obs):
        h, w, c = obs.shape
        y, x = np.where((obs[:, :, 0]==228))
        if len(y)>0:
            y, x = np.mean(y), np.mean(x)
            y, x = int(y/h*16), int(x/w*16)
            return y, x
        else:
            return self.y, self.x

    def get_cell_inventory(self, obs):
        y, x, c = obs.shape
        # obs_key = obs[int(.47*y): int(.55*y), int(.05*x): int(.15*x), ...] # location of first key
        obs_inventory = obs[int(.1*y): int(.22*y), int(.3*x): int(.65*x), ...]
        cell = cv2.cvtColor(obs_inventory, cv2.COLOR_RGB2GRAY)
        cell = cv2.resize(cell, (6, 3), interpolation=cv2.INTER_AREA)
        cell = (cell/255. * 8).astype(np.uint8, casting='unsafe')
        return cell

class ImageCellInfo(gym.Wrapper):
    def __init__(self, env, latent_h=11, latent_w=8, latent_d=20):
        super().__init__(env)
        self.latent_h, self.latent_w, self.latent_d = latent_h, latent_w, latent_d

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info['cell_img'] = self.get_cell_obs(obs, self.latent_h, self.latent_w, self.latent_d, ret_tuple=False)
        info['cell'] = tuple(info['cell_img'].flatten())
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['cell_img'] = self.get_cell_obs(obs, self.latent_h, self.latent_w, self.latent_d, ret_tuple=False)
        info['cell'] = tuple(info['cell_img'].flatten())
        return obs, reward, terminated, truncated, info
    def get_cell_obs(self, obs, latent_h=11, latent_w=8, latent_d=20, ret_tuple=True):
        # obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (latent_w, latent_h), interpolation=cv2.INTER_AREA)
        obs = (obs*latent_d).astype(np.uint8, casting='unsafe')
        return tuple(obs.flatten()) if ret_tuple else obs

class StoreObsInfo(gym.Wrapper):
    def __init__(self, env, key_name='obs_ori'):
        super().__init__(env)
        self.key_name = key_name

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info[self.key_name] = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info[self.key_name] = obs
        return obs, reward, terminated, truncated, info

class ObservationDivide(gym.ObservationWrapper):
    def __init__(self, env, divide=255.):
        super().__init__(env)
        obs_space = self.observation_space
        self.divide = divide
        self.observation_space = gym.spaces.Box(obs_space.low, obs_space.high/self.divide, obs_space.shape, dtype=np.float64)
    def observation(self, obs):
        return (obs/self.divide).astype(np.float32)

class AtariOneLife(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.terminated_data = None
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.terminated_data = None
        return obs, info
    def step(self, action):
        if self.terminated_data is None:
            obs, reward, terminated, truncated, info = self.env.step(action)
            terminated = self.env.ale.lives() < 6
            if terminated:
                self.terminated_data = obs, reward, terminated, truncated, info
        else:
            obs, reward, terminated, truncated, info = self.terminated_data
        # we need deep copy because future envs will alter it, and we want to maintain original
        return obs, reward, terminated, truncated, copy.deepcopy(info)

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
    
class ZeroReward(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = 0
        return obs, reward, terminated, truncated, info

class TerminationReward(gym.Wrapper):
    def __init__(self, env, termination_reward=-1):
        super().__init__(env)
        self.termination_reward = termination_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward = reward + self.termination_reward
        return obs, reward, terminated, truncated, info

# class CurrentTrajectoryStats(gym.Wrapper):
class CumulativeReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cumulative_reward, self.len_traj = 0, 0

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.cumulative_reward, self.len_traj = 0, 0
        info['cumulative_reward'], info['len_traj'] = self.cumulative_reward, self.len_traj
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.cumulative_reward += reward
        self.len_traj += 1
        info['cumulative_reward'], info['len_traj'] = self.cumulative_reward, self.len_traj
        return obs, reward, terminated, truncated, info

class ActionHistory(gym.Wrapper):
    def __init__(self, env, max_len=None):
        super().__init__(env)
        self.action_history = []
        self.max_len = max_len
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.action_history = []
        info['action_history'] = np.array(self.action_history)
        return obs, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.action_history.append(action)
        if self.max_len is not None:
            self.action_history = self.action_history[-self.max_len:]
        info['action_history'] = np.array(self.action_history)
        return obs, reward, terminated, truncated, info

class RewardHistory(gym.Wrapper):
    def __init__(self, env, max_len=None):
        super().__init__(env)
        self.reward_history = []
        self.max_len = max_len
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.reward_history = []
        info['reward_history'] = np.array(self.reward_history)
        return obs, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.reward_history.append(reward)
        if self.max_len is not None:
            self.reward_history = self.reward_history[-self.max_len:]
        info['reward_history'] = np.array(self.reward_history)
        return obs, reward, terminated, truncated, info

class ObservationHistory(gym.Wrapper):
    def __init__(self, env, max_len=None):
        super().__init__(env)
        self.observation_history = []
        self.max_len = max_len
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.observation_history = [obs]
        info['observation_history'] = np.stack(self.observation_history)
        return obs, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.observation_history.append(obs)
        if self.max_len is not None:
            self.observation_history = self.observation_history[-self.max_len:]
        info['observation_history'] = np.stack(self.observation_history)
        return obs, reward, terminated, truncated, info


class DecisionTransformerEnv(gym.Wrapper):
    def __init__(self, env, n_obs=4):
        self.n_obs = n_obs
        self.n_rewards = n_obs-1
        if not hasattr(env, 'action_history'):
            self.env = ActionHistory(env, max_len=n_obs)
        if not hasattr(env, 'reward_history'):
            self.env = RewardHistory(env, max_len=n_obs)
        if not hasattr(env, 'observation_history'):
            self.env = ObservationHistory(env, max_len=n_obs)
        super().__init__(env)
        # obs_space = self.observation_space
        # new_shape = (n_obs,)+obs_space.shape
        # self.observation_space = gym.spaces.Box(obs_space.low, obs_space.high/255., new_shape, dtype=np.float32)
    
    def update_info(self, info):
        pass
        # obs_list = self.obs_list
        # action_list = self.action_list
        # reward_list = self.reward_list

        # padding = self.n_obs-len(obs_list)
        # obs_list = obs_list + [np.zeros_like(obs_list[0]) for _ in range(padding)]
        # action_list = action_list + [-1 for _ in range(padding)]
        # reward_list = reward_list + [np.nan for _ in range(padding)]

        # mask = [True for _ in range(self.n_obs-padding)] + [False for _ in range(padding)]

        # info['obs_list'] = np.stack(obs_list)
        # info['action_list'] = np.array(action_list)
        # info['reward_list'] = np.array(reward_list)
        # info['mask'] = np.array(mask)
        
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.update_info(info)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.update_info(info)
        return obs, reward, terminated, truncated, info

class EasierMRActionSpace(gym.ActionWrapper):
    # NOOP FIRE UP RIGHT LEFT DOWN RIGHTFIRE LEFTFIRE
    underlying_actions = [0, 1, 2, 3, 4, 5, 11, 12]
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.underlying_actions))
    def action(self, action):
        return self.underlying_actions[action]

class DeterministicReplayReset(gym.Wrapper):
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.snapshot = []
        info['snapshot'] = self.get_snapshot()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.snapshot.append(action)
        info['snapshot'] = self.get_snapshot()
        return obs, reward, terminated, truncated, info

    def restore_snapshot(self, snapshot, *args, **kwargs):
        obs, info = self.reset(*args, **kwargs) # call my own method, not self.env's
        reward, terminated, truncated = 0, False, False
        for action in snapshot:
            obs, reward, terminated, truncated, info = self.step(action) # call my own method, not self.env's
        return obs, reward, terminated, truncated, info

    def get_snapshot(self):
        return np.array(self.snapshot)

class ALEState(gym.Wrapper):
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        del info['frame_number']
        return obs, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        del info['frame_number']
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        if name=='ale_state':
            return self.env.clone_state(include_rng=True)
        else:
            return super().__getattr__(name)
    def __setattr__(self, name, value):
        if name=='ale_state':
            self.env.restore_state(value)
        else:
            super().__setattr__(name, value)

class DeterministicRestoreReset(gym.Wrapper):
    def __init__(self, env, class2keys):
        super().__init__(env)
        self.class2keys = class2keys
        self.env_vars, self.action = {}, None
        self.create_class2env()

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.env_vars, self.action = {}, None
        info['snapshot'] = self.get_snapshot()
        return obs, info

    def step(self, action):
        self.env_vars = self.get_env_variables()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.action = action
        info['snapshot'] = self.get_snapshot()
        return obs, reward, terminated, truncated, info

    def restore_snapshot(self, snapshot, *args, **kwargs):
        if snapshot['action'] is None:
            obs, info = self.reset(*args, **kwargs) # call my own method, not self.env's
            return obs, 0, False, False, info
        else:
            self.set_env_variables(snapshot)
            return self.step(snapshot['action']) # call my own method, not self.env's

    def get_snapshot(self):
        snapshot = self.env_vars
        snapshot['action'] = self.action
        return snapshot

    def create_class2env(self):
        self.class2env = {}
        env = self
        while hasattr(env, 'env'):
            env = env.env
            classname = type(env).__name__
            if classname in self.class2keys:
                self.class2env[classname] = env
                if self.class2keys[classname] is None:
                    keys = [key for key in env.__dict__.keys() if not key.startswith('_') and key!='env']
                    self.class2keys[classname] = keys

    def get_env_variables(self):
        env_vars = {}
        for classname, keys in self.class2keys.items():
            env = self.class2env[classname]
            env_vars[classname] = {key: copy.deepcopy(getattr(env, key)) for key in keys}
        return env_vars

    def set_env_variables(self, env_vars):
        for classname, keys in self.class2keys.items():
            env = self.class2env[classname]
            for key in keys:
                setattr(env, key, copy.deepcopy(env_vars[classname][key]))

    def print_env_variables(self):
        env = self
        while hasattr(env, 'env'):
            env = env.env
            keys = env.__dict__.keys()
            keys = [i for i in keys if not i.startswith('_') and i!='env'] 
            print(f'{type(env).__name__:25s}: {keys}')


# altered from https://github.com/Farama-Foundation/Gymnasium/blob/c52ef43a1bc56bec7c2adea5130f8119a4d58065/gymnasium/vector/sync_vector_env.py#L67
from copy import deepcopy

from gymnasium.vector.utils import concatenate, create_empty_array, iterate


class RestorableSyncVectorEnv(gym.vector.SyncVectorEnv):
    def __init__(self, *args, auto_reset=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_reset = auto_reset

    def restore_snapshot(self, snapshots, *args, **kwargs):
        obs, info = self.reset(*args, **kwargs)
        observations, infos = [], {}
        for i, (env, snapshot) in enumerate(zip(self.envs, snapshots)):

            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.restore_snapshot(snapshot)

            if self.auto_reset and (self._terminateds[i] or self._truncateds[i]):
                old_observation, old_info = observation, info
                observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )

    def step_wait(self):
        """Steps through each of the environments returning the batched results.
        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):

            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            if self.auto_reset and (self._terminateds[i] or self._truncateds[i]):
                old_observation, old_info = observation, info
                observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )




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

from collections import OrderedDict


class DictObservation(gym.wrappers.TransformObservation):
    def __init__(self, env):
        super().__init__(env, lambda obs: OrderedDict(obs=obs))
        self.observation_space = gym.spaces.Dict(obs=self.env.observation_space)
