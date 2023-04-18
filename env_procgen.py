import numpy as np
import torch

import gym
from procgen import ProcgenEnv


class ProcgenWrapper(gym.Wrapper):
    def __init__(self, env):
        # env = gym.wrappers.TransformObservation(env, lambda obs: obs["rgb"])
        super().__init__(env)
        self.single_action_space = env.action_space
        self.action_space = gym.spaces.MultiDiscrete([self.single_action_space.n] * self.num_envs)
        self.single_observation_space = env.observation_space["rgb"]
        self.observation_space = None  # TODO implement this
        self.is_vector_env = True
        self.action_meanings = ["leftdown", "left", "leftup", "down", "noop", "up", "rightdown", "right", "rightup", "d", "a", "w", "s", "q", "e"]

    def reset(self):
        obs = self.env.reset()
        obs = obs["rgb"]
        return obs, {}

    def step(self, action):
        obs, rew, done, infos = self.env.step(action)
        obs = obs["rgb"]
        return obs, rew, done, {}


class StoreObs(gym.Wrapper):
    def __init__(self, env, n_envs=25, store_limit=1000):
        super().__init__(env)
        self.n_envs = n_envs
        self.store_limit = store_limit
        self.past_obs = []

    def reset(self):
        obs, info = self.env.reset()
        self.past_obs.append(obs[: self.n_envs])
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.past_obs.append(obs[: self.n_envs])
        self.past_obs = self.past_obs[-self.store_limit :]
        info["past_obs"] = self.past_obs
        return obs, rew, done, info


class ToTensor(gym.Wrapper):
    def __init__(self, env, device=None):
        super().__init__(env)
        self.device = device

    def reset(self):
        obs, info = self.env.reset()
        info["obs"] = torch.from_numpy(obs).to(self.device)
        info["done"] = torch.ones(self.num_envs, dtype=bool, device=info["obs"].device)
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info["obs"] = torch.from_numpy(obs).to(self.device)
        info["rew"] = torch.from_numpy(rew).to(self.device)
        info["ext"] = torch.from_numpy(rew).to(self.device)
        info["done"] = torch.from_numpy(done).to(self.device)
        return obs, rew, done, info


class E3BReward(gym.Wrapper):
    def __init__(self, env, encoder, lmbda=0.1):
        super().__init__(env)
        self.set_encoder(encoder, lmbda)

    def set_encoder(self, encoder, lmbda=0.1):
        self.encoder = encoder
        if encoder is not None:
            n_features = self.encoder.n_features
            self.Il = torch.eye(n_features) / lmbda  # d, d
            self.Cinv = torch.zeros(self.num_envs, n_features, n_features)  # b, d, d

    def reset(self):
        obs, info = self.env.reset()
        if self.encoder is not None:
            info["e3b"] = self.step_e3b(info["obs"], info["done"])
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.encoder is not None:
            info["e3b"] = self.step_e3b(info["obs"], info["done"])
        return obs, rew, done, info

    @torch.no_grad()  # this is required
    def step_e3b(self, obs, done):
        self.Il, self.Cinv = self.Il.to(obs.device), self.Cinv.to(obs.device)
        assert done.dtype == torch.bool
        self.Cinv[done] = self.Il
        v = self.encoder.calc_features(obs)[..., :, None]  # b, d, 1
        u = self.Cinv @ v  # b, d, 1
        b = v.mT @ u  # b, 1, 1
        self.Cinv = self.Cinv - u @ u.mT / (1.0 + b)  # b, d, d
        rew_e3b = b[..., 0, 0].detach()
        rew_e3b[done] = 0.0
        return rew_e3b


class StoreReturns(gym.Wrapper):
    def __init__(self, env, store_limit=1000):
        super().__init__(env)
        # running returns
        self.ret_ext, self.ret_e3b, self.ret_cov, self.traj_len = None, None, None, None
        # list (past store_limit timesteps) of tensors (envs that were done) of returns
        self.rets_ext, self.rets_e3b, self.rets_cov, self.traj_lens = [], [], [], []
        self.store_limit = store_limit

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if self.ret_ext is None:
            self.ret_ext = torch.zeros_like(info["ext"])
            self.ret_e3b = torch.zeros_like(info["ext"])
            self.ret_cov = torch.zeros_like(info["ext"])
            self.traj_len = torch.zeros_like(info["ext"]).to(torch.int)

        self.ret_ext += info["ext"]
        if "e3b" in info:
            self.ret_e3b += info["e3b"]
        if "cov" in info:
            self.ret_cov += info["cov"]
        self.traj_len += 1

        self.rets_ext.append(self.ret_ext[info["done"]])
        self.rets_e3b.append(self.ret_e3b[info["done"]])
        self.rets_cov.append(self.ret_cov[info["done"]])
        self.traj_lens.append(self.traj_len[info["done"]])

        self.rets_ext = self.rets_ext[-self.store_limit :]
        self.rets_e3b = self.rets_e3b[-self.store_limit :]
        self.rets_cov = self.rets_cov[-self.store_limit :]
        self.traj_lens = self.traj_lens[-self.store_limit :]

        self.ret_ext[info["done"]] = 0.0
        self.ret_e3b[info["done"]] = 0.0
        self.ret_cov[info["done"]] = 0.0
        self.traj_len[info["done"]] = 0

        return obs, rew, done, info


class RewardSelector(gym.Wrapper):
    def __init__(self, env, obj="ext"):
        super().__init__(env)
        self.obj = obj

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        rew = info[self.obj].detach().cpu().numpy()
        return obs, rew, done, info


class OrdinalActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        am_udlr = ["noop", "up", "down", "left", "right"]
        self.i_actions = np.array([i for i, am in enumerate(self.action_meanings) if am in am_udlr])
        self.action_meanings = [self.action_meanings[i] for i in self.i_actions]
        self.single_action_space = gym.spaces.Discrete(len(self.action_meanings))
        self.action_space = gym.spaces.MultiDiscrete([self.single_action_space.n] * self.num_envs)

    def step(self, action):
        action = self.i_actions[action]
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info


class MinerCoverageReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.pobs, self.mask_episodic = None, None

    def reset(self):
        obs, info = self.env.reset()
        self.pobs = info["obs"][:, ::2, ::2, :]  # n_envs, h, w, c
        self.mask_episodic = (self.pobs != self.pobs).any(dim=-1)
        self.mask_episodic_single = self.mask_episodic[0].clone()
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        o = info["obs"][:, ::2, ::2, :]
        dmask = (self.pobs != o).any(dim=-1)
        rew_eps = (dmask & ~self.mask_episodic).any(dim=-1).any(dim=-1).float()
        rew_eps[info["done"]] = 0.0
        info["cov"] = rew_eps
        self.mask_episodic = self.mask_episodic | dmask
        self.pobs = o
        self.mask_episodic[done] = self.mask_episodic_single
        return obs, rew, done, info


def make_env(env_id, obj, num_envs, start_level, num_levels, distribution_mode, gamma, encoder=None, device="cpu", cov=True, actions="all"):
    env = ProcgenEnv(num_envs=num_envs, env_name=env_id, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    env = ProcgenWrapper(env)
    env = StoreObs(env)
    env = ToTensor(env, device=device)
    env = E3BReward(env, encoder=encoder, lmbda=0.1)
    env = MinerCoverageReward(env)
    env = StoreReturns(env)
    env = RewardSelector(env, obj)
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    if actions == "ordinal":
        env = OrdinalActions(env)
    return env


import argparse
import time
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--e3b", default=False, action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    from agent_procgen import E3B

    obs_shape = (64, 64, 3)
    n_actions = 15
    e3b = E3B(64, obs_shape, n_actions, 100).to(args.device) if args.e3b else None
    envs = make_env("miner", "rew", 64, 0, 0, "hard", 0.999, e3b=e3b, device=args.device)
    obs, info = envs.reset()
    n_steps = 10000
    start = time.time()
    for i in tqdm(range(n_steps)):
        obs, rew, done, info = envs.step(np.random.randint(0, 5, size=64))
    print((n_steps * 64) / (time.time() - start), "SPS")

    # print(torch.cat(envs.rets_ext))
    # print(torch.cat(envs.rets_e3b))
    # print(torch.cat(envs.traj_lens))
    # print(torch.cat(envs.traj_lens).sum())

    # a = torch.cat(envs.traj_lens)
    # b = torch.cat(envs.rets_e3b)

    # import matplotlib.pyplot as plt
    # plt.scatter(a.cpu().numpy(), b.cpu().numpy())
    # plt.show()
