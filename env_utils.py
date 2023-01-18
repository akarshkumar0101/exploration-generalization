import copy

import cv2
import gymnasium as gym
import numpy as np


class MRDomainCellInfo(gym.Wrapper):
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
        reward, terminated, truncated = 0, False, False
        for action in snapshot:
            obs, reward, terminated, truncated, info = self.step(action) # call my own method, not self.env's
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
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.obs, self.reward, self.terminated, self.info = obs, 0, False, info
        return obs, info
    def step(self, action):
        if self.terminated:
            obs, reward, terminated, truncated, info = self.obs, self.reward, self.terminated, False, self.info
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
            terminated = self.env.ale.lives() < 6
            self.obs, self.reward, self.info = obs, reward, info
            self.terminated = self.terminated or terminated
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

class CurrentTrajectoryStats(gym.Wrapper):
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

class DecisionTransformerEnv(gym.Wrapper):
    def __init__(self, env, n_obs=4):
        super().__init__(env)
        self.n_obs = n_obs
        self.n_rewards = n_obs-1
        # obs_space = self.observation_space
        # new_shape = (n_obs,)+obs_space.shape
        # self.observation_space = gym.spaces.Box(obs_space.low, obs_space.high/255., new_shape, dtype=np.float32)
    
    def update_info(self, info):
        obs_list = self.obs_list
        action_list = self.action_list
        reward_list = self.reward_list

        padding = self.n_obs-len(obs_list)
        obs_list = obs_list + [np.zeros_like(obs_list[0]) for _ in range(padding)]
        action_list = action_list + [-1 for _ in range(padding)]
        reward_list = reward_list + [np.nan for _ in range(padding)]

        mask = [True for _ in range(self.n_obs-padding)] + [False for _ in range(padding)]

        info['obs_list'] = np.stack(obs_list)
        info['action_list'] = np.array(action_list)
        info['reward_list'] = np.array(reward_list)
        info['mask'] = np.array(mask)
        
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.obs_list = [obs]
        self.action_list = []
        self.reward_list = []
        self.update_info(info)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.obs_list.append(obs)
        self.action_list.append(action)
        self.reward_list.append(reward)

        self.obs_list = self.obs_list[-self.n_obs:]
        self.action_list = self.action_list[-self.n_rewards:]
        self.reward_list = self.reward_list[-self.n_rewards:]

        self.update_info(info)
        return obs, reward, terminated, truncated, info

# class MyVectorEnv(gym.vector.SyncVectorEnv):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def reset(self, *args, **kwargs):
#         o, i = [], []
#         for env in self.envs:
#             obs, info = env.reset()
#             o.append(obs)
#             i.append(info)
#         return np.stack(o), i
        
#     def restore_snapshots(self, snapshots):
#         o, r, t, tr, i = [], [], [], [], []
#         for env, snapshot in zip(self.envs, snapshots):
#             obs, reward, terminated, truncated, info = env.restore_snapshot(snapshot)
#             o.append(obs)
#             r.append(reward)
#             t.append(terminated)
#             tr.append(truncated)
#             i.append(info)
#         return np.stack(o), np.array(r), np.array(t), np.array(tr), i

#     def step(self, action):
#         o, r, t, tr, i = [], [], [], [], []
#         for env, a in zip(self.envs, action):
#             obs, reward, terminated, truncated, info = env.step(a)
#             o.append(obs)
#             r.append(reward)
#             t.append(terminated)
#             tr.append(truncated)
#             i.append(info)
#         return np.stack(o), np.array(r), np.array(t), np.array(tr), i



# def restore_snapshots(envs, snapshots, *args, **kwargs):
#     obs, info = envs.reset(*args, **kwargs)
#     for env, snapshot in zip(envs.envs, snapshots):
#         obs, reward, terminated, truncated, info = env.restore_snapshot(snapshot)
#     # for action in snapshot:
#         # obs, reward, terminated, truncated, info = self.step(action) # call my own method, not self.env's
#     return obs, reward, terminated, truncated, info

# altered from https://github.com/Farama-Foundation/Gymnasium/blob/c52ef43a1bc56bec7c2adea5130f8119a4d58065/gymnasium/vector/sync_vector_env.py#L67
from copy import deepcopy

from gymnasium.vector.utils import concatenate, create_empty_array, iterate


class RestorableSyncVectorEnv(gym.vector.SyncVectorEnv):
    def __init__(self, *args, auto_reset=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_reset = auto_reset

    # def reset(self, *args, **kwargs):
    #     o, i = [], []
    #     for env in self.envs:
    #         obs, info = env.reset()
    #         o.append(obs)
    #         i.append(info)
    #     return np.stack(o), i
        
    # def restore_snapshots(self, snapshots):
    #     o, r, t, tr, i = [], [], [], [], []
    #     for env, snapshot in zip(self.envs, snapshots):
    #         obs, reward, terminated, truncated, info = env.restore_snapshot(snapshot)
    #         o.append(obs)
    #         r.append(reward)
    #         t.append(terminated)
    #         tr.append(truncated)
    #         i.append(info)
    #     return np.stack(o), np.array(r), np.array(t), np.array(tr), i

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

    # def step(self, action):
    #     o, r, t, tr, i = [], [], [], [], []
    #     for env, a in zip(self.envs, action):
    #         obs, reward, terminated, truncated, info = env.step(a)
    #         o.append(obs)
    #         r.append(reward)
    #         t.append(terminated)
    #         tr.append(truncated)
    #         i.append(info)
    #     return np.stack(o), np.array(r), np.array(t), np.array(tr), i

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





import torch


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