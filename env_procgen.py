import numpy as np
import torch

import gym
from procgen import ProcgenEnv

class NormalGym(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def reset(self):
        obs = self.env.reset()
        return obs, {}
    def step(self, action):
        obs, rew, done, infos = self.env.step(action)
        return obs, rew, done, {}

class StoreObs(gym.Wrapper):
    def __init__(self, env, n_envs=25, store_limit=1000):
        super().__init__(env)
        self.n_envs = n_envs
        self.store_limit = store_limit
        self.past_obs = []

    def reset(self):
        obs, info = self.env.reset()
        self.past_obs.append(obs[:self.n_envs])
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.past_obs.append(obs[:self.n_envs])
        self.past_obs = self.past_obs[-self.store_limit:]
        info['past_obs'] = self.past_obs
        return obs, rew, done, info

class ToTensor(gym.Wrapper):
    def __init__(self, env, device=None):
        super().__init__(env)
        self.device = device

    def reset(self):
        obs, info = self.env.reset()
        info['obs'] = torch.from_numpy(obs).to(self.device)
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info['obs'] = torch.from_numpy(obs).to(self.device)
        info['rew'] = torch.from_numpy(rew).to(self.device)
        info['done'] = torch.from_numpy(done).to(self.device)
        return obs, rew, done, info

class E3BReward(gym.Wrapper):
    def __init__(self, env, e3b):
        super().__init__(env)
        self.e3b = e3b

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        e3b_rew = self.e3b.calc_reward(info['obs'], done=info['done'])
        info['e3b'] = e3b_rew
        return obs, rew, done, info
        
class StoreReturns(gym.Wrapper):
    def __init__(self, env, store_limit=1000):
        super().__init__(env)
        # running returns
        self._ret_ext, self._ret_e3b, self._traj_len = None, None, None
        self.rets_ext, self.rets_e3b, self.traj_lens = [], [], []
        self.store_limit = store_limit

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if self._ret_ext is None:
            self._ret_ext = torch.zeros_like(info['rew'])
            self._ret_e3b = torch.zeros_like(info['rew'])
            self._traj_len = torch.zeros_like(info['rew']).to(torch.int)

        self._ret_ext += info['rew']
        if 'e3b' in info:
            self._ret_e3b += info['e3b']
        self._traj_len += 1

        self.rets_ext.append(self._ret_ext[info['done']])
        self.rets_e3b.append(self._ret_e3b[info['done']])
        self.traj_lens.append(self._traj_len[info['done']])

        self.rets_ext = self.rets_ext[-self.store_limit:]
        self.rets_e3b = self.rets_e3b[-self.store_limit:]
        self.traj_lens = self.traj_lens[-self.store_limit:]

        self._ret_ext[info['done']] = 0.
        self._ret_e3b[info['done']] = 0.
        self._traj_len[info['done']] = 0

        return obs, rew, done, info

class ReturnTracker(gym.Wrapper):
    def __init__(self, env, device='cpu'):
        super().__init__(env)
        self._ret_ext, self._ret_eps, self._traj_len = None, None, None
        self.device = device

    def reset(self):
        obs, info = self.env.reset()
        self._ret_ext = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ret_e3b = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._traj_len = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self._ret_ext += info['rew']
        if 'e3b' in info:
            self._ret_e3b += info['e3b']
        self._traj_len += 1

        info['ret_ext'] = self._ret_ext[info['done']]
        info['ret_e3b'] = self._ret_e3b[info['done']]
        info['traj_len'] = self._traj_len[info['done']]

        self._ret_ext[info['done']] = 0.
        self._ret_e3b[info['done']] = 0.
        self._traj_len[info['done']] = 0

        return obs, rew, done, info

class RewardSelector(gym.Wrapper):
    def __init__(self, env, obj='rew'):
        super().__init__(env)
        self.obj = obj
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        rew = info[self.obj].detach().cpu().numpy()
        return obs, rew, done, info


def make_env(env_id, obj, num_envs, start_level, num_levels, distribution_mode, gamma, e3b=None, device='cpu'):
    envs = ProcgenEnv(num_envs=num_envs, env_name=env_id, num_levels=num_levels,
                      start_level=start_level, distribution_mode=distribution_mode)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.multi_action_space = gym.spaces.MultiDiscrete([envs.action_space.n] * num_envs)
    envs.observation_space = envs.observation_space["rgb"]
    envs.single_observation_space = envs.observation_space
    envs.is_vector_env = True
    envs = NormalGym(envs)
    envs = StoreObs(envs)
    envs = ToTensor(envs, device=device)
    if e3b is not None:
        envs = E3BReward(envs, e3b)
    envs = StoreReturns(envs)
    envs = RewardSelector(envs, obj)
    envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    return envs


import argparse
import time
from tqdm.auto import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--e3b", default=False, action='store_true')

if __name__=='__main__':
    args = parser.parse_args()
    from agent_procgen import E3B
    obs_shape = (64, 64, 3)
    n_actions = 15
    e3b = E3B(64, obs_shape, n_actions, 100).to(args.device) if args.e3b else None
    envs = make_env('miner', 'rew', 64, 0, 0, 'hard', 0.999, e3b=e3b, device=args.device)
    obs, info = envs.reset()
    n_steps = 10000
    start = time.time()
    for i in tqdm(range(n_steps)):
        obs, rew, done, info = envs.step(np.random.randint(0, 5, size=64))
    print((n_steps*64)/(time.time() - start), 'SPS')

    # print(torch.cat(envs.rets_ext))
    # print(torch.cat(envs.rets_e3b))
    # print(torch.cat(envs.traj_lens))
    # print(torch.cat(envs.traj_lens).sum())

    # a = torch.cat(envs.traj_lens)
    # b = torch.cat(envs.rets_e3b)

    # import matplotlib.pyplot as plt
    # plt.scatter(a.cpu().numpy(), b.cpu().numpy())
    # plt.show()
