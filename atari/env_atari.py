from functools import partial

import gymnasium as gym
import numpy as np
import torch
import collections

import normalize

import envpool

from einops import rearrange


# def make_env_single(env_id="Breakout", frame_stack=4):
#     env = gym.make(f"ALE/{env_id}-v5", frameskip=1, full_action_space=True)
#     # TODO: reduce space of actions
#     env = gym.wrappers.AtariPreprocessing(env, terminal_on_life_loss=True)
#     env = gym.wrappers.TransformReward(env, lambda reward: np.sign(reward))
#     env = gym.wrappers.FrameStack(env, num_stack=frame_stack)
#     return env


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

    def step(self, action):
        action = rearrange(action, "(n b) ... -> n b ...", n=len(self.envs))
        obss, rews, terms, truncs, infos = zip(*[env.step(a) for env, a in zip(self.envs, action)])
        obs = rearrange(list(obss), "n b ... -> (n b) ...")
        rew = rearrange(list(rews), "n b ... -> (n b) ...")
        term = rearrange(list(terms), "n b ... -> (n b) ...")
        trunc = rearrange(list(truncs), "n b ... -> (n b) ...")
        info = {k: rearrange([info[k] for info in infos], "n b ... -> (n b) ...") for k in infos[0].keys()}
        return obs, rew, term, trunc, info


# class NoArgsReset(gym.Wrapper):
#     def reset(self, *args, **kwargs):
#         return self.env.reset()


# def make_env_envpool(n_envs, env_id, episodic_life=True, full_action_space=True, seed=0):
#     env = envpool.make_gymnasium(
#         task_id=f"{env_id}-v5",
#         num_envs=n_envs,
#         # batch_size=None,
#         # num_threads=None,
#         seed=seed,  # default: 42
#         max_episode_steps=2700,  # default: 27000
#         # img_height=84,  # default: 84
#         # img_width=84,  # default: 84
#         stack_num=1,  # default: 4
#         # gray_scale=True,  # default: True
#         # frame_skip=4,  # default: 4
#         # noop_max=30,  # default: 30
#         episodic_life=episodic_life,  # default: False
#         # zero_discount_on_life_loss=False,  # default: False
#         # reward_clip=False,  # default: False
#         # repeat_action_probability=0,  # default: 0
#         # use_inter_area_resize=True,  # default: True
#         # use_fire_reset=True,  # default: True
#         full_action_space=full_action_space,  # default: False
#     )
#     env = NoArgsReset(env)
#     env.num_envs = n_envs
#     env.single_observation_space = env.observation_space
#     env.single_action_space = env.action_space
#     env.observation_space = gym.spaces.Box(low=0, high=255, shape=(n_envs,) + env.single_observation_space.shape, dtype=np.uint8)
#     env.action_space = gym.spaces.MultiDiscrete([env.single_action_space.n for _ in range(n_envs)])
#     return env


class MyEnvpool(gym.Env):
    def __init__(self, *args, **kwargs):
        self.env = envpool.make_gymnasium(*args, **kwargs)
        self.n_envs, self.num_envs = kwargs["num_envs"], kwargs["num_envs"]
        self.single_observation_space = self.env.observation_space
        self.single_action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.n_envs,) + self.single_observation_space.shape, dtype=np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([self.single_action_space.n for _ in range(self.n_envs)])

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset()
        info["players"] = info["players"]["env_id"]
        return obs, info

    def reset_subenvs(self, ids):
        return self.env.reset(ids)

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, list):
            action = np.array(action)
        obs, rew, term, trunc, info = self.env.step(action)
        info["players"] = info["players"]["env_id"]
        return obs, rew, term, trunc, info


def make_env_single(env_id, episodic_life=True, full_action_space=True):
    env = gym.make(f"ALE/{env_id}-v5", frameskip=1, repeat_action_probability=0, full_action_space=True)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=episodic_life, grayscale_obs=True, scale_obs=False)
    env = gym.wrappers.TransformObservation(env, lambda obs: obs[None, :, :])
    env.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, *env.observation_space.shape), dtype=np.uint8)
    return env


def make_env_gymnasium(n_envs, env_id, episodic_life=True, full_action_space=True, seed=0):
    env_fns = [partial(make_env_single, env_id=env_id, episodic_life=episodic_life, full_action_space=full_action_space) for _ in range(n_envs)]
    env = gym.vector.SyncVectorEnv(env_fns)
    env.get_ram = lambda: np.stack([e.unwrapped.ale.getRAM() for e in env.envs])
    return env


def make_env(env_id="Breakout", n_envs=8, obj="ext", norm_rew=True, gamma=0.99, episodic_life=True, full_action_space=True, device=None, seed=0, lib="envpool"):
    if lib == "envpool":
        # env = make_env_envpool(n_envs, env_id, episodic_life=episodic_life, full_action_space=full_action_space, seed=seed)
        env = MyEnvpool(
            task_id=f"{env_id}-v5",
            num_envs=n_envs,
            seed=seed,
            # max_episode_steps=27000,
            stack_num=1,
            # noop_max=1,
            # use_fire_reset=False,
            episodic_life=episodic_life,
            full_action_space=full_action_space,
        )

    elif lib == "gymnasium":
        env = make_env_gymnasium(n_envs, env_id, episodic_life=episodic_life, full_action_space=full_action_space, seed=seed)
    else:
        raise ValueError(f"Unknown lib: {lib}")

    env = StoreObs(env, n_envs_store=4, buf_size=450)
    env = ToTensor(env, device=device)
    env = EpisodicReward(env)
    env = RNDReward(env)
    env = StoreReturns(env)
    env = normalize.NormalizeReward(env, key_rew=obj, gamma=gamma, eps=1e-5)
    env = RewardSelector(env, obj=f"{obj}_norm" if norm_rew else obj)
    return env


def make_concat_env(env_ids, *args, **kwargs):
    return ConcatEnv([make_env(env_id, *args, **kwargs) for env_id in env_ids])


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
        return obs, rew, term, trunc, info

    def get_past_obs(self):
        return self.past_obs


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
        info["rew_score"] = torch.from_numpy(rew).to(self.device)
        info["rew_ext"] = torch.sign(info["rew_score"])
        if "reward" in info:
            del info["reward"]
        info["rew_traj"] = torch.ones_like(info["rew_ext"])
        info["term"] = torch.from_numpy(term).to(self.device, torch.bool)
        info["trunc"] = torch.from_numpy(trunc).to(self.device, torch.bool)
        if "terminated" in info:
            info["term_atari"] = torch.from_numpy(info["terminated"]).to(self.device, torch.bool)
        else:
            info["term_atari"] = info["term"] | info["trunc"]
        info["done"] = info["term"] | info["trunc"]

        self.timestep += 1
        self.timestep[info["done"]] = 0
        info["timestep"] = self.timestep
        return obs, rew, term, trunc, info


class RewardSelector(gym.Wrapper):
    def __init__(self, env, obj="ext", clip_min=None, clip_max=None):
        super().__init__(env)
        self.obj = obj
        self.clip_min, self.clip_max = clip_min, clip_max

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        # rew = info[f"rew_{self.obj}"].detach().cpu().numpy()
        # info["rew"] = info[f"rew_{self.obj}"].clamp(self.clip_min, self.clip_max)
        info["rew"] = info[f"rew_{self.obj}"]
        return obs, rew, term, trunc, info


# class StoreReturns(gym.Wrapper):
#     def __init__(self, env, key_ret="ret_", key_term="term_atari", buf_size=512):
#         super().__init__(env)
#         self.key_ret, self.key_term, self.buf_size = key_ret, key_term, buf_size
#         self.key2running_ret = {}  # key -> running return
#         self.key2past_rets = {}  # key -> list of arrays of past returns

#     def step(self, action):
#         obs, rew, term, trunc, info = self.env.step(action)
#         done = info[self.key_term]
#         for key in [key for key in info if key.startswith("rew_")]:
#             keyr = key.replace("rew_", self.key_ret)
#             if keyr not in self.key2running_ret:
#                 self.key2running_ret[keyr] = torch.zeros_like(info[key])
#                 self.key2past_rets[keyr] = collections.deque(maxlen=self.buf_size)

#             self.key2running_ret[keyr] += info[key]
#             self.key2past_rets[keyr].append(self.key2running_ret[keyr][done].clone())
#             # info[keyr] = self.key2running_ret[keyr].clone()
#             self.key2running_ret[keyr][done] = 0.0
#         return obs, rew, term, trunc, info

#     def get_past_returns(self):
#         return {key: torch.cat(list(val), dim=0) for key, val in self.key2past_rets.items()}


# This is correct implementation of StoreReturns


class StoreReturns(gym.Wrapper):
    def __init__(self, env, key_term="term_atari", past_k_rets=5):
        super().__init__(env)
        self.key_term = key_term
        self.past_k_rets = past_k_rets
        self.key2running_ret = {}  # key -> running return
        self.key2past_rets = {}  # key -> list (over env) of deque (over time) of past returns

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        done = info[self.key_term]
        for key in [key for key in info if key.startswith("rew_")]:
            keyr = key.replace("rew_", "ret_")
            if keyr not in self.key2running_ret:
                self.key2running_ret[keyr] = torch.zeros_like(info[key])
                self.key2past_rets[keyr] = [collections.deque([np.nan] * self.num_envs, maxlen=self.past_k_rets) for _ in range(self.num_envs)]
            self.key2running_ret[keyr] += info[key]

            donelist = done.tolist()
            retlist = self.key2running_ret[keyr].tolist()
            for i in range(self.num_envs):
                if donelist[i]:  # env finished
                    self.key2past_rets[keyr][i].append(retlist[i])
            self.key2running_ret[keyr][done] = 0.0
        return obs, rew, term, trunc, info

    def get_past_returns(self):
        return {key: np.array(val) for key, val in self.key2past_rets.items()}


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


# class EpisodicReward(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.encode_fn = None

#     def configure_eps_reward(self, encode_fn=None, ctx_len=10, k=1):
#         self.encode_fn = encode_fn
#         assert ctx_len >= k
#         self.memory = collections.deque(maxlen=ctx_len)  # m b d
#         self.k = k

#     @torch.no_grad()  # this is required
#     def reset(self, *args, **kwargs):
#         obs, info = self.env.reset(*args, **kwargs)
#         if self.encode_fn is not None:
#             latent = self.encode_fn(info["obs"])
#             self.memory.append(latent)
#         return obs, info

#     @torch.no_grad()  # this is required
#     def step(self, action):
#         obs, rew, term, trunc, info = self.env.step(action)
#         if self.encode_fn is not None:
#             latent = self.encode_fn(info["obs"])  # b d
#             memory = torch.stack(list(self.memory), dim=1)  # b m d
#             self.memory.append(latent)
#             if memory.shape[1] >= self.k:
#                 d = (latent[:, None, :] - memory).norm(dim=-1)  # b m
#                 dk = d.topk(k=self.k, dim=-1, largest=False).values  # b k
#                 info["rew_eps"] = dk.mean(dim=-1)  # b. TODO: is mean the right thing to do?
#                 assert info["rew_eps"].shape == info["rew_ext"].shape
#             else:
#                 info["rew_eps"] = torch.zeros_like(info["rew_ext"])
#         return obs, rew, term, trunc, info


from einops import repeat


class EpisodicReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.encode_fn = None

    def state_encoder(self, obs):
        return torch.nn.functional.avg_pool2d(obs.float(), kernel_size=(3, 3)).to(torch.uint8).reshape(-1, 28 * 28)

    def configure_eps_reward(self, encode_fn=None, ctx_len=10, k=1, p=2, obj="eps"):
        assert ctx_len >= k
        self.encode_fn, self.ctx_len, self.k, self.p = encode_fn, ctx_len, k, p
        self.myobj = obj

    @torch.no_grad()  # this is required
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if self.encode_fn is not None:
            latent = self.encode_fn(info["obs"])  # b d
            self.memory = repeat(latent, "b d -> b m d", m=self.ctx_len).clone()  # clone makes items independent
            self.i_mem = 1
        return obs, info

    @torch.no_grad()  # this is required
    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        done = info["done"]
        if self.encode_fn is not None:
            latent = self.encode_fn(info["obs"])  # b d
            # whereever done is true, replace the entire memory with the current latent
            self.memory[done, :, :] = latent[done, None, :]  # bp m d

            d = (latent[:, None, :] - self.memory).float().norm(dim=-1, p=self.p)  # b m
            dk = d.topk(k=self.k, dim=-1, largest=False).values  # b k
            info[f"rew_{self.myobj}"] = dk.mean(dim=-1)  # b. TODO: is mean the right thing to do?

            self.memory[:, self.i_mem, :] = latent
            self.i_mem = (self.i_mem + 1) % self.ctx_len

        return obs, rew, term, trunc, info


# class BufferEpisodicReward:
#     def __init__(self, encode_fn=None, ctx_len=10, k=1):
#         assert ctx_len >= k
#         self.encode_fn, self.ctx_len, self.k = encode_fn, ctx_len, k

#     @torch.no_grad()
#     def calc_rewards(self, obss, dones):
#         n_envs, n_steps = obss.shape[:2]
#         rews = torch.zeros(n_envs, n_steps, device=obss.device)
#         if getattr(self, "memory", None) is None:
#             self.i_mem = 0
#             latent = self.encode_fn(obss[:, 0])
#             self.memory = repeat(latent, "b d -> b m d", m=self.ctx_len)

#         for i_step in range(n_steps):
#             latent = self.encode_fn(obss[:, i_step])  # b d
#             # TODO: MEMORY ASSINGMENT SHOULD BE AFER COMPUTING REWARD
#             self.memory[:, self.i_mem, :] = latent
#             self.i_mem = (self.i_mem + 1) % self.ctx_len
#             self.memory[dones[:, i_step], :, :] = latent[dones[:, i_step], None, :]  # bp m d

#             d = (latent[:, None, :] - self.memory).norm(dim=-1)  # b m
#             dk = d.topk(k=self.k, dim=-1, largest=False).values  # b k
#             rews[:, i_step] = dk.mean(dim=-1)  # b. TODO: is mean the right thing to do?
#         return rews


class RNDReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.rnd_model = None

    def configure_rnd_reward(self, rnd_model=None):
        self.rnd_model = rnd_model

    @torch.no_grad()  # this is required
    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if self.rnd_model is not None:
            rnd_student, rnd_teacher = self.rnd_model(info["obs"], update_rms_obs=True)  # b d
            info["rew_rnd"] = (rnd_student - rnd_teacher).pow(2).mean(dim=-1)  # b
            assert info["rew_rnd"].shape == info["rew_ext"].shape
        return obs, rew, term, trunc, info


class BufferRNDReward:
    def __init__(self, rnd_model=None):
        self.rnd_model = rnd_model

    @torch.no_grad()
    def calc_rewards(self, obss):
        n_envs, n_steps = obss.shape[:2]
        rews = torch.zeros(n_envs, n_steps, device=obss.device)
        for i_step in range(n_steps):
            rnd_student, rnd_teacher = self.rnd_model(obss[:, i_step], update_rms_obs=True)  # b d
            rews[:, i_step] = (rnd_student - rnd_teacher).pow(2).mean(dim=-1)  # b
        return rews


import normalize


# class BufferNormalizeReward:
#     def __init__(self, gamma, eps=1e-8):
#         self.rms_returns, self.returns = normalize.RunningMeanStd(), None
#         self.gamma, self.eps = gamma, eps

#     def normalize_rews(self, rews):
#         n_envs, n_steps = rews.shape
#         for i_step in range(n_steps):
#             self.returns = self.returns * self.gamma + rews[:, i_step]
#             self.rms_returns.update(self.returns)
#         return rews / (self.rms_returns.var + self.eps).sqrt()


"""
specialist_ext_{env_id}
specialist_eps_{env_id}
specialist_rnd_{env_id}

generalist_pre_ext
generalist_pre_eps
generalist_pre_rnd

generalist_pre_ext_ft__bc_{env_id}
generalist_pre_ext_ft_ppo_{env_id}
generalist_pre_eps_ft__bc_{env_id}
generalist_pre_eps_ft_ppo_{env_id}
generalist_pre_rnd_ft__bc_{env_id}
generalist_pre_rnd_ft_ppo_{env_id}

"""
