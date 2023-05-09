import gym
import numpy as np
import torch
from procgen import ProcgenEnv
from torch import nn


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
    def __init__(self, env, n_envs=25, buf_size=1000):
        super().__init__(env)
        self.n_envs = n_envs
        self.buf_size = buf_size
        self.past_obs = []

    def reset(self):
        obs, info = self.env.reset()
        self.past_obs.append(obs[: self.n_envs])
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.past_obs.append(obs[: self.n_envs])
        self.past_obs = self.past_obs[-self.buf_size :]
        info["past_obs"] = self.past_obs
        return obs, rew, done, info


class ToTensor(gym.Wrapper):
    def __init__(self, env, device=None):
        super().__init__(env)
        self.device = device

    def reset(self):
        obs, info = self.env.reset()
        info["obs"] = torch.from_numpy(obs).to(self.device)
        info["rew_ext"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        info["rew_alive"] = torch.ones_like(info["rew_ext"])
        info["done"] = torch.ones(self.num_envs, dtype=bool, device=self.device)

        self.timestep = torch.zeros(self.num_envs, dtype=int, device=self.device)
        info["timestep"] = self.timestep
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info["obs"] = torch.from_numpy(obs).to(self.device)
        info["rew_ext"] = torch.from_numpy(rew).to(self.device)
        info["rew_alive"] = torch.ones_like(info["rew_ext"])
        info["done"] = torch.from_numpy(done).to(self.device)

        self.timestep += 1
        self.timestep[info["done"]] = 0
        info["timestep"] = self.timestep
        return obs, rew, done, info


class StoreReturns(gym.Wrapper):
    def __init__(self, env, buf_size=1000):
        super().__init__(env)
        self.buf_size = buf_size
        # running returns
        self.key2running_ret = {}
        self.key2running_ret_dsc = {}  # discounted
        # list of past returns
        self.key2past_rets = {}

    def reset(self):
        obs, info = self.env.reset()
        self.update_rets(info, resetting=True)
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.update_rets(info)
        return obs, rew, done, info

    def update_rets(self, info, resetting=False):
        for key in info:
            if not key.startswith("rew"):
                continue
            keyr = key.replace("rew", "ret")
            if keyr not in self.key2running_ret:
                self.key2running_ret[keyr] = torch.zeros_like(info[key])
                self.key2past_rets[keyr] = []
            self.key2running_ret[keyr] += info[key].to(self.device)
            if not resetting:
                self.key2past_rets[keyr].append(self.key2running_ret[keyr][info["done"]])
            self.key2past_rets[keyr] = self.key2past_rets[keyr][-self.buf_size :]
            self.key2running_ret[keyr][info["done"]] = 0.0


class RewardSelector(gym.Wrapper):
    def __init__(self, env, obj="ext"):
        super().__init__(env)
        self.obj = obj

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        rew = info[f"rew_{self.obj}"].detach().cpu().numpy()
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
    def __init__(self, env, res=1):
        super().__init__(env)
        self.pobs, self.mask_episodic = None, None
        self.res = res

    def reset(self):
        obs, info = self.env.reset()
        self.pobs = info["obs"][:, :: self.res, :: self.res, :]  # n_envs, h, w, c
        self.mask_episodic = (self.pobs != self.pobs).any(dim=-1)
        self.mask_episodic_single = self.mask_episodic[0].clone()
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        o = info["obs"][:, :: self.res, :: self.res, :]
        dmask = (self.pobs != o).any(dim=-1)
        rew_cov = (dmask & ~self.mask_episodic).any(dim=-1).any(dim=-1).float()
        rew_cov[info["done"]] = 0.0
        info["rew_cov"] = rew_cov
        self.mask_episodic = self.mask_episodic | dmask
        self.pobs = o
        self.mask_episodic[done] = self.mask_episodic_single
        return obs, rew, done, info

class MinerOracleExplorationReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info['rew_mex'] = info["rew_nov_xy"] + torch.sign(info["rew_ext"])
        return obs, rew, done, info

class ObservationEncoder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.key2encoder = {}

    def add_encoder(self, key, encoder):
        self.key2encoder[key] = encoder

    def reset(self):
        obs, info = self.env.reset()
        self.update_info(info)
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.update_info(info)
        return obs, rew, done, info

    @torch.no_grad()  # this is required
    def update_info(self, info):
        for key, encoder in self.key2encoder.items():
            encoder = encoder.to(self.device)
            info[key] = encoder.encode(info["obs"])


class NoveltyReward(gym.Wrapper):
    def __init__(self, env, latent_key=None, buf_size=100):
        super().__init__(env)
        self.latent_key = latent_key
        self.buf_size = buf_size

    def reset(self):
        obs, info = self.env.reset()
        # episodic archive of all latents seen in this episode
        if self.latent_key in info:
            self.archive = torch.full((self.num_envs, self.buf_size, info[self.latent_key].shape[-1]), torch.inf, device=self.device)
            info[f"rew_nov_{self.latent_key}"] = self.step_nov(info[self.latent_key], info["done"], info["timestep"])
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.latent_key in info:
            info[f"rew_nov_{self.latent_key}"] = self.step_nov(info[self.latent_key], info["done"], info["timestep"])
        return obs, rew, done, info

    @torch.no_grad()  # this is required
    def step_nov(self, latent, done, timestep):
        assert done.dtype == torch.bool
        self.archive[done, :, :] = torch.inf  # reset archive for envs starting in a new obs
        rew_nov = (self.archive - latent[:, None, :]).norm(dim=-1).min(dim=-1).values  # compute novelty of this latent
        rew_nov[done] = 0.0
        self.archive[range(len(timestep)), timestep, :] = latent  # update archive with this latent
        return rew_nov


class E3BReward(gym.Wrapper):
    def __init__(self, env, latent_key, lmbda=0.1):
        super().__init__(env)
        self.latent_key = latent_key
        self.lmbda = lmbda

    def reset(self):
        obs, info = self.env.reset()
        if self.latent_key in info:
            latent = info[self.latent_key]
            self.Il = torch.eye(latent.shape[-1]) / self.lmbda  # d, d
            self.Cinv = torch.zeros(self.num_envs, latent.shape[-1], latent.shape[-1])  # b, d, d
            info[f"rew_e3b_{self.latent_key}"] = self.step_e3b(info[self.latent_key], info["done"])
        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.latent_key in info:
            info[f"rew_e3b_{self.latent_key}"] = self.step_e3b(info[self.latent_key], info["done"])
        return obs, rew, done, info

    @torch.no_grad()  # this is required
    def step_e3b(self, latent, done):
        self.Il, self.Cinv = self.Il.to(self.device), self.Cinv.to(self.device)
        assert done.dtype == torch.bool
        self.Cinv[done] = self.Il
        v = latent[..., :, None]  # b, d, 1
        u = self.Cinv @ v  # b, d, 1
        b = v.mT @ u  # b, 1, 1
        rew_e3b = b[..., 0, 0]
        rew_e3b[done] = 0.0
        self.Cinv = self.Cinv - u @ u.mT / (1.0 + b)  # b, d, d
        return rew_e3b


class MinerXYEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent_color = torch.tensor([111, 196, 169])

    def encode(self, obs):
        self.agent_color = self.agent_color.to(obs.device)
        col_dist = (obs - self.agent_color).float().norm(dim=-1)
        vals, x = col_dist.min(dim=-1)
        y = vals.argmin(dim=-1)
        x = x[range(len(y)), y]
        x = torch.stack([x, y], dim=-1)
        latent = 10*x.float() / obs.shape[-2]
        return latent


def make_env(env_id="miner", obj="ext", num_envs=64, start_level=0, num_levels=0, distribution_mode="hard", gamma=0.999, latent_keys=[], device="cpu", cov=True, actions="all"):
    env = ProcgenEnv(num_envs=num_envs, env_name=env_id, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    env = ProcgenWrapper(env)
    env = StoreObs(env)
    env = ToTensor(env, device=device)

    env = ObservationEncoder(env)

    for latent_key in latent_keys:
        # env = E3BReward(env, latent_key=latent_key, lmbda=0.1)
        env = NoveltyReward(env, latent_key=latent_key, buf_size=1000)
    if cov:
        env = MinerCoverageReward(env)
    env = MinerOracleExplorationReward(env)

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

    obs_shape = (64, 64, 3)
    n_actions = 15
    # e3b = E3B(64, obs_shape, n_actions, 100).to(args.device) if args.e3b else None
    # envs = make_env("miner", "rew", 64, 0, 0, "hard", 0.999, e3b=e3b, device=args.device)
    # obs, info = envs.reset()
    # n_steps = 10000
    # start = time.time()
    # for i in tqdm(range(n_steps)):
    #     obs, rew, done, info = envs.step(np.random.randint(0, 5, size=64))
    # print((n_steps * 64) / (time.time() - start), "SPS")
