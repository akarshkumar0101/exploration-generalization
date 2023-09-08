from functools import partial

import gymnasium as gym
import numpy as np
import torch
import collections

import envpool

from einops import rearrange


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        assert hasattr(env, "num_envs")
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.traj_rets = collections.deque(maxlen=deque_size)
        self.traj_lens = collections.deque(maxlen=deque_size)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.episode_returns += info["reward"]  # use info["reward"] instead of rew to get true atari rewards
        self.episode_lengths += 1
        self.traj_rets.extend(self.episode_returns[info["terminated"]])
        self.traj_lens.extend(self.episode_lengths[info["terminated"]])
        self.episode_returns *= 1 - info["terminated"]  # use info["terminated"] instead of term to get true atari terminated
        self.episode_lengths *= 1 - info["terminated"]
        return obs, rew, term, trunc, info


class MyEnvpool(gym.Env):
    def __init__(self, env_id, *args, **kwargs):
        self.env = envpool.make_gymnasium(env_id, *args, **kwargs)
        self.env_id = env_id
        self.n_envs, self.num_envs = kwargs["num_envs"], kwargs["num_envs"]
        self.single_observation_space = self.env.observation_space
        self.single_action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.n_envs,) + self.single_observation_space.shape, dtype=np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([self.single_action_space.n for _ in range(self.n_envs)])

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset()
        info["players"] = info["players"]["env_id"]
        info["terminated"] = info["terminated"].astype(bool)
        return obs, info

    def reset_subenvs(self, ids):
        return self.env.reset(ids)

    def step(self, action):
        self.send(action)
        return self.recv()

    def send(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, list):
            action = np.array(action)
        self.env.send(action)

    def recv(self, *args, **kwargs):
        obs, rew, term, trunc, info = self.env.recv(*args, **kwargs)
        info["players"] = info["players"]["env_id"]
        info["terminated"] = info["terminated"].astype(bool)
        return obs, rew, term, trunc, info


class ConcatEnv:
    def __init__(self, envs):
        self.envs = envs
        self.n_envs = self.num_envs = sum([env.num_envs for env in envs])
        self.single_observation_space = envs[0].single_observation_space
        self.single_action_space = envs[0].single_action_space
        self.action_space = gym.spaces.MultiDiscrete([envs[0].single_action_space.n for _ in range(self.num_envs)])

    def reset(self):
        obss, infos = zip(*[env.reset() for env in self.envs])  # lists
        obs = rearrange(list(obss), "n b ... -> (n b) ...")
        info = {k: rearrange([info[k] for info in infos], "n b ... -> (n b) ...") for k in infos[0].keys()}
        return obs, info

    def step(self, action, async_=True):
        action = rearrange(action, "(n b) ... -> n b ...", n=len(self.envs))
        if async_:
            [env.send(a) for env, a in zip(self.envs, action)]
            obss, rews, terms, truncs, infos = zip(*[env.recv() for env, a in zip(self.envs, action)])
        else:
            obss, rews, terms, truncs, infos = zip(*[env.step(a) for env, a in zip(self.envs, action)])
        obs = rearrange(list(obss), "n b ... -> (n b) ...")
        rew = rearrange(list(rews), "n b ... -> (n b) ...")
        term = rearrange(list(terms), "n b ... -> (n b) ...")
        trunc = rearrange(list(truncs), "n b ... -> (n b) ...")
        info = {k: rearrange([info[k] for info in infos], "n b ... -> (n b) ...") for k in infos[0].keys()}
        return obs, rew, term, trunc, info

    def reset_subenvs(self, ids):
        i_env = ids // len(self.envs)
        i_subenv = ids % len(self.envs)

        obss, infos = [], []
        for i, env in enumerate(self.envs):
            ids_ = i_subenv[i_env == i]
            if len(ids_) > 0:
                obs, info = env.reset_subenvs(ids_)
                obss.append(obs)
                infos.append(info)
        if isinstance(obs, np.ndarray):
            obs = np.concatenate(obss)
            info = {k: np.concatenate([info[k] for info in infos]) for k in infos[0].keys()}
        else:
            obs = torch.cat(obss)
            info = {k: torch.cat([info[k] for info in infos]) for k in infos[0].keys()}
        return obs, info


# def make_concat_env(env_ids, *args, **kwargs):
#     return ConcatEnv([MyEnvpool(env_id, *args, **kwargs) for env_id in env_ids])


class ToTensor(gym.Wrapper):
    def __init__(self, env, device=None):
        super().__init__(env)
        self.device = device

    def info_to_tensor(self, info):
        for key in info:
            if isinstance(info[key], np.ndarray):
                info[key] = torch.from_numpy(info[key]).to(self.device)
        return info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        obs = torch.from_numpy(obs).to(self.device)
        info = self.info_to_tensor(info)
        return obs, info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, list):
            action = np.array(action)
        obs, rew, term, trunc, info = self.env.step(action)
        obs = torch.from_numpy(obs).to(device=self.device)
        rew = torch.from_numpy(rew).to(dtype=torch.float32, device=self.device)
        term = torch.from_numpy(term).to(dtype=torch.bool, device=self.device)
        trunc = torch.from_numpy(trunc).to(dtype=torch.bool, device=self.device)
        info = self.info_to_tensor(info)
        return obs, rew, term, trunc, info


if __name__ == "__main__":
    import timers

    timer = timers.Timer()
    env = MyEnvpool("MontezumaRevenge-v5", num_envs=8)
    env = gym.wrappers.NormalizeReward(env)
    env = RecordEpisodeStatistics(env)
    env = ToTensor(env)
    env.reset()
    with timer.add_time("steps"):
        for i in range(1000):
            env.step(env.action_space.sample())
    print((env.n_envs * 1000) / timer.key2time["steps"])
