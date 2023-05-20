from functools import partial

import gymnasium as gym
import numpy as np
import torch

# def make_env_envpool(env_id, num_envs, ):
#     import envpool
#     seed = 0
#     envs = envpool.make(env_id, env_type="gym", num_envs=num_envs, episodic_life=True, reward_clip=True, seed=seed)
#     envs.num_envs = num_envs
#     envs.single_action_space = envs.action_space
#     envs.action_space = gym.spaces.Discrete(envs.action_space.nvec.sum())
#     envs.single_observation_space = envs.observation_space
#     envs = RecordEpisodeStatistics(envs)
#     assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"


def make_env_single(env_id="Breakout"):
    env = gym.make(f"ALE/{env_id}-v5", frameskip=1, full_action_space=True)
    # TODO: reduce space of actions
    env = gym.wrappers.AtariPreprocessing(env, terminal_on_life_loss=True)

    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


def make_env(env_id="Breakout", n_envs=8, obj="ext", gamma=0.999, device="cpu", seed=0):
    make_fn = partial(make_env_single, env_id=env_id)
    make_fns = [make_fn for _ in range(n_envs)]
    env = gym.vector.SyncVectorEnv(make_fns)

    env = StoreObs(env, n_envs=25, buf_size=1000)

    # env = gym.wrappers.FrameStack(env, num_stack=4, lz4_compress=False)

    env = ToTensor(env, device=device)
    env = StoreReturns(env, buf_size=128)

    env = RewardSelector(env, obj=obj)

    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    for i, envi in enumerate(env.envs):
        envi.seed(seed + i * 1000)
        envi.action_space.seed(seed + i * 1000)
        envi.observation_space.seed(seed + i * 1000)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


class StoreObs(gym.Wrapper):
    def __init__(self, env, n_envs=25, buf_size=1000):
        super().__init__(env)
        self.n_envs, self.buf_size = n_envs, buf_size
        self.past_obs = []

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.past_obs.append(obs[: self.n_envs])
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.past_obs.append(obs[: self.n_envs])
        self.past_obs = self.past_obs[-self.buf_size :]
        info["past_obs"] = self.past_obs
        return obs, rew, term, trunc, info


class ToTensor(gym.Wrapper):
    def __init__(self, env, device=None):
        super().__init__(env)
        self.device = device

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info["obs"] = torch.from_numpy(obs).to(self.device)
        # info["rew_ext"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # info["rew_alive"] = torch.ones_like(info["rew_ext"])
        # info["term"] = torch.ones(self.num_envs, dtype=bool, device=self.device)
        # info["trunc"] = torch.ones(self.num_envs, dtype=bool, device=self.device)
        # info["done"] = info["term"] | info["trunc"]

        self.timestep = torch.zeros(self.num_envs, dtype=int, device=self.device)
        info["timestep"] = self.timestep
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        rew = rew.astype(np.float32)
        info["obs"] = torch.from_numpy(obs).to(self.device)
        info["rew_ext"] = torch.from_numpy(rew).to(self.device)
        info["rew_alive"] = torch.ones_like(info["rew_ext"])
        info["term"] = torch.from_numpy(term).to(self.device)
        info["trunc"] = torch.from_numpy(trunc).to(self.device)
        info["done"] = info["term"] | info["trunc"]

        self.timestep += 1
        self.timestep[info["done"]] = 0
        info["timestep"] = self.timestep
        return obs, rew, term, trunc, info


class RewardSelector(gym.Wrapper):
    def __init__(self, env, obj="ext"):
        super().__init__(env)
        self.obj = obj

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        rew = info[f"rew_{self.obj}"].detach().cpu().numpy()
        return obs, rew, term, trunc, info


class StoreReturns(gym.Wrapper):
    def __init__(self, env, buf_size=128):
        super().__init__(env)
        self.buf_size = buf_size
        self.key2running_ret = {}  # key -> running return
        self.key2past_rets = {}  # key -> list of arrays of past returns

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)

        for key in info:
            if not key.startswith("rew"):
                continue
            keyr = key.replace("rew", "ret")
            if keyr not in self.key2running_ret:
                self.key2running_ret[keyr] = torch.zeros_like(info[key])
                self.key2past_rets[keyr] = []
            self.key2running_ret[keyr] += info[key]

            self.key2past_rets[keyr].append(self.key2running_ret[keyr][info["done"]])
            self.key2past_rets[keyr] = self.key2past_rets[keyr][-self.buf_size :]
            self.key2running_ret[keyr][info["done"]] = 0.0

        return obs, rew, term, trunc, info
