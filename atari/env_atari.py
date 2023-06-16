from functools import partial

import gymnasium as gym
import numpy as np
import torch
import collections

try:
    import envpool

    has_envpool = True
except ModuleNotFoundError:
    has_envpool = False
    print("WARNING: envpool not found.")

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


def make_env_single(env_id="Breakout", frame_stack=4):
    env = gym.make(f"ALE/{env_id}-v5", frameskip=1, full_action_space=True)
    # TODO: reduce space of actions
    env = gym.wrappers.AtariPreprocessing(env, terminal_on_life_loss=True)
    env = gym.wrappers.TransformReward(env, lambda reward: np.sign(reward))
    env = gym.wrappers.FrameStack(env, num_stack=frame_stack)
    return env


def make_env(env_id="Breakout", n_envs=8, frame_stack=4, obj="ext", e3b_encode_fn=None, gamma=0.999, device="cpu", seed=0, buf_size=128):
    make_fn = partial(make_env_single, env_id=env_id, frame_stack=frame_stack)
    make_fns = [make_fn for _ in range(n_envs)]
    env = gym.vector.SyncVectorEnv(make_fns)

    env = StoreObs(env, n_envs_store=25, buf_size=1000)

    env = ToTensor(env, device=device)

    env = E3BReward(env, encode_fn=e3b_encode_fn, lmbda=0.1)

    env = StoreReturns(env, buf_size=buf_size)

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


class NoArgsReset(gym.Wrapper):
    def reset(self, *args, **kwargs):
        return self.env.reset()


def make_env(env_id="Breakout", n_envs=8, obj="ext", e3b_encode_fn=None, gamma=0.99, full_action_space=True, device=None, seed=0):
    if has_envpool:
        env = envpool.make_gymnasium(
            task_id=f"{env_id}-v5",
            num_envs=n_envs,
            # batch_size=None,
            # num_threads=None,
            seed=seed,  # default: 42
            # max_episode_steps=27000,  # default: 27000
            # img_height=84,  # default: 84
            # img_width=84,  # default: 84
            stack_num=1,  # default: 4
            # gray_scale=True,  # default: True
            # frame_skip=4,  # default: 4
            # noop_max=30,  # default: 30
            episodic_life=True,  # default: False
            # zero_discount_on_life_loss=False,  # default: False
            reward_clip=True,  # default: False
            # repeat_action_probability=0,  # default: 0
            # use_inter_area_resize=True,  # default: True
            # use_fire_reset=True,  # default: True
            full_action_space=full_action_space,  # default: False
        )
        env = NoArgsReset(env)
        env.num_envs = n_envs
        env.single_observation_space = env.observation_space
        env.single_action_space = env.action_space
        env.observation_space = gym.spaces.Box(low=0, high=255, shape=(n_envs,) + env.single_observation_space.shape, dtype=np.uint8)
        env.action_space = gym.spaces.MultiDiscrete([env.single_action_space.n for _ in range(n_envs)])
    else:
        make_fn = partial(make_env_single, env_id=env_id, frame_stack=1)
        make_fns = [make_fn for _ in range(n_envs)]
        env = gym.vector.SyncVectorEnv(make_fns)
        for i, envi in enumerate(env.envs):
            envi.seed(seed + i * 1000)
            envi.action_space.seed(seed + i * 1000)
            envi.observation_space.seed(seed + i * 1000)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    env = StoreObs(env, n_envs_store=4, buf_size=450)
    env = ToTensor(env, device=device)
    # env = E3BReward(env, encode_fn=e3b_encode_fn, lmbda=0.1)
    env = StoreReturns(env)
    env = RewardSelector(env, obj=obj)

    # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    return env


class StoreObs(gym.Wrapper):
    def __init__(self, env, n_envs_store=4, buf_size=1000):
        super().__init__(env)
        self.n_envs_store = n_envs_store
        self.past_obs = collections.deque(maxlen=buf_size)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.past_obs.append(obs[: self.n_envs_store])
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.past_obs.append(obs[: self.n_envs_store])
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
        # info["rew_traj"] = torch.ones_like(info["rew_ext"])
        # info["term"] = torch.ones(self.num_envs, dtype=bool, device=self.device)
        # info["trunc"] = torch.ones(self.num_envs, dtype=bool, device=self.device)
        # info["done"] = info["term"] | info["trunc"]
        info["done"] = torch.ones(self.num_envs, dtype=bool, device=self.device)

        self.timestep = torch.zeros(self.num_envs, dtype=int, device=self.device)
        info["timestep"] = self.timestep
        return obs, info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, list):
            action = np.array(action)
        obs, rew, term, trunc, info = self.env.step(action)
        rew = rew.astype(np.float32)
        info["obs"] = torch.from_numpy(obs).to(self.device)
        info["rew_ext"] = torch.from_numpy(rew).to(self.device)
        info["rew_score"] = torch.from_numpy(info["reward"]).to(self.device)
        del info["reward"]
        info["rew_traj"] = torch.ones_like(info["rew_ext"])
        info["term"] = torch.from_numpy(term).to(self.device, torch.bool)
        info["trunc"] = torch.from_numpy(trunc).to(self.device, torch.bool)
        info["term_atari"] = torch.from_numpy(info["terminated"]).to(self.device, torch.bool)
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
        info["rew"] = info[f"rew_{self.obj}"]
        return obs, rew, term, trunc, info


class StoreReturns(gym.Wrapper):
    def __init__(self, env, key_ret="ret_", key_term="term_atari", buf_size=512):
        super().__init__(env)
        self.key_ret, self.key_term, self.buf_size = key_ret, key_term, buf_size
        self.key2running_ret = {}  # key -> running return
        self.key2past_rets = {}  # key -> list of arrays of past returns

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        term = info[self.key_term]
        for key in [key for key in info if key.startswith("rew_")]:
            keyr = key.replace("rew_", self.key_ret)
            if keyr not in self.key2running_ret:
                self.key2running_ret[keyr] = torch.zeros_like(info[key])
                self.key2past_rets[keyr] = collections.deque(maxlen=self.buf_size)

            self.key2running_ret[keyr] += info[key]
            self.key2past_rets[keyr].append(self.key2running_ret[keyr][term].clone())
            # info[keyr] = self.key2running_ret[keyr].clone()
            self.key2running_ret[keyr][term] = 0.0
        return obs, rew, term, trunc, info

    def get_past_returns(self):
        return {key: torch.cat(list(val), dim=0) for key, val in self.key2past_rets.items()}


class E3BReward(gym.Wrapper):
    def __init__(self, env, encode_fn, lmbda=0.1):
        super().__init__(env)
        self.encode_fn = encode_fn
        self.lmbda = lmbda

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        # if self.latent_key in info:
        # latent, done = info[self.latent_key], info["done"]
        if self.encode_fn is not None:
            latent, done = self.encode_fn(info["obs"]), info["done"]

            self.Il = torch.eye(latent.shape[-1]) / self.lmbda  # d, d
            self.Cinv = torch.zeros(self.num_envs, latent.shape[-1], latent.shape[-1])  # b, d, d

            # info[f"rew_e3b_{self.latent_key}"] = self.step_e3b(latent, done)
            self.step_e3b(latent, done)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        # if self.latent_key in info:
        # latent, done = info[self.latent_key], info["done"]
        if self.encode_fn is not None:
            latent, done = self.encode_fn(info["obs"]), info["done"]

            # info[f"rew_e3b_{self.latent_key}"] = self.step_e3b(latent, done)
            info[f"rew_e3b"] = self.step_e3b(latent, done)

        return obs, rew, term, trunc, info

    @torch.no_grad()  # this is required
    def step_e3b(self, latent, done):
        self.Il, self.Cinv = self.Il.to(self.device), self.Cinv.to(self.device)
        assert done.dtype == torch.bool
        self.Cinv[done] = self.Il
        v = latent[..., :, None]  # b, d, 1
        u = self.Cinv @ v  # b, d, 1
        b = v.mT @ u  # b, 1, 1
        rew_e3b = b[..., 0, 0].clone()  # cloning is important, otherwise b will be written to
        rew_e3b[done] = 0.0
        self.u, self.v, self.b = u, v, b
        self.Cinv = self.Cinv - u @ u.mT / (1.0 + b)  # b, d, d
        return rew_e3b
