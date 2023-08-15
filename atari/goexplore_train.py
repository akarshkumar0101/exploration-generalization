# Adapted from CleanRL's ppo_atari_envpool.py
import argparse
import glob
import os
import random
import time
from collections import defaultdict
from distutils.util import strtobool
from functools import partial

import agent_atari
import atari_data
import buffers
import gymnasium as gym
import normalize
import numpy as np
import timers
import torch
import torchinfo
import utils
import wandb
from buffers import Buffer, MultiBuffer
from einops import rearrange, repeat
from env_atari import MyEnvpool, ToTensor, make_concat_env
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--entity", type=str, default=None)
parser.add_argument("--project", type=str, default=None)
parser.add_argument("--name", type=str, default=None)
# parser.add_argument("--log-video", type=lambda x: bool(strtobool(x)), default=False)
# parser.add_argument("--log-hist", type=lambda x: bool(strtobool(x)), default=False)

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)

# Algorithm arguments
parser.add_argument("--env_ids", type=str, nargs="+", default=["MontezumaRevenge"])
parser.add_argument("--strategy", type=str, default="best")
parser.add_argument("--n_iters", type=lambda x: int(float(x)), default=int(1e4))
parser.add_argument("--n_envs", type=int, default=1)
parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_updates", type=int, default=16)

parser.add_argument("--ctx_len", type=int, default=32)
parser.add_argument("--save_agent", type=str, default=None)

parser.add_argument("--lr", type=float, default=2.5e-4)
# parser.add_argument("--lr-warmup", type=lambda x: bool(strtobool(x)), default=True)
# parser.add_argument("--lr-decay", type=str, default="none")
# parser.add_argument("--max-grad-norm", type=float, default=1.0, help="the maximum norm for the gradient clipping")

parser.add_argument("--n_archives", type=int, default=1)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    args.n_envs_per_id = args.n_envs // len(args.env_ids)
    return args


class GEBuffer(Buffer):
    def __init__(self, env, n_steps, sample_traj_fn, device=None):
        super().__init__(env, n_steps, device=device)
        self.trajs = [None for _ in range(self.env.num_envs)]
        self.traj_lens = np.zeros(self.env.num_envs, dtype=int)
        self.i_locs = np.zeros(self.env.num_envs, dtype=int)
        self.sample_traj_fn = sample_traj_fn

        self.buf_obs = np.zeros((env.num_envs, n_steps, 84, 84), dtype=np.uint8)
        self.buf_act = np.zeros((env.num_envs, n_steps), dtype=np.uint8)
        self.next_obs = np.zeros_like(env.observation_space.sample())
        env.envs[0].reset(seed=0)
        self.ram_start = env.envs[0].ale.getRAM().copy()

    def traj_over_subenv(self, id):
        self.trajs[id] = self.sample_traj_fn()
        self.traj_lens[id] = len(self.trajs[id])
        self.i_locs[id] = 0
        obs, _ = self.env.envs[id].reset(seed=0)
        assert np.array_equal(self.env.envs[id].ale.getRAM(), self.ram_start), "env reset to seed=0"
        return obs

    def gecollect(self, pbar=None):
        if self.first_collect:
            self.first_collect = False
            # obs, info = self.env.reset()
            # self.next_obs = info["obs"]
            # self.sample_new_traj(np.arange(self.env.num_envs))
            for id in range(self.env.num_envs):
                self.next_obs[id] = self.traj_over_subenv(id)

        for t in range(self.n_steps):
            self.buf_obs[:, t] = self.next_obs
            action = np.array([traj[i_loc] for traj, i_loc in zip(self.trajs, self.i_locs)])
            self.i_locs += 1
            self.buf_act[:, t] = action
            self.next_obs, _, term, trunc, _ = self.env.step(action)
            if any(term) or any(trunc):
                print(self.env.envs[0])
                print(term, trunc)
            assert not any(term) and not any(trunc), "found a done in the ge buffer"
            # self.dones[:, t] = info["done"]
            # self.next_obs = info["obs"]
            for id in np.where(self.i_locs >= self.traj_lens)[0]:
                self.next_obs[id] = self.traj_over_subenv(id)
        self.obss[:, :] = torch.from_numpy(self.buf_obs).to(self.device)
        self.acts[:, :] = torch.from_numpy(self.buf_act).to(self.device)


def make_ge_env_single(env_id):
    env = gym.make(f"ALE/{env_id}-v5", frameskip=1, repeat_action_probability=0.0, full_action_space=True)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=1, frame_skip=4, screen_size=84, grayscale_obs=True)
    return env


def make_ge_env(env_id, n_envs):
    make_fn = partial(make_ge_env_single, env_id=env_id)
    return gym.vector.SyncVectorEnv([make_fn for _ in range(n_envs)])


def load_env_id2archives(env_ids, n_archives):
    env_id2archives = {}
    for env_id in tqdm(env_ids):
        env_id2archives[env_id] = [np.load(f, allow_pickle=True).item() for f in sorted(glob.glob(f"./data/goexplore/{env_id}*"))[:n_archives]]
    return env_id2archives


def get_env_id2trajs(env_id2archives, strategy="best"):
    env_id2trajs = {}
    for env_id, archives in tqdm(env_id2archives.items()):
        env_id2trajs[env_id] = []
        for archive in archives:
            trajs, rets, scores = archive["trajs"], archive["rets"], archive["scores"]
            if strategy == "best":
                idx = [np.argmax(rets)]
            elif strategy == "random":
                idx = np.arange(len(trajs))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            trajs, rets, scores = trajs[idx], rets[idx], scores[idx]
            env_id2trajs[env_id].extend(trajs)
    return env_id2trajs


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading archives")
    env_id2archives = load_env_id2archives(args.env_ids, args.n_archives)
    print("Creating trajs")
    env_id2trajs = get_env_id2trajs(env_id2archives, strategy=args.strategy)
    for env_id, trajs in env_id2trajs.items():
        print(f"env_id: {env_id}, #trajs: {len(trajs)}")

    print("Creating buffers")
    bufs = []
    for env_id in args.env_ids:
        env = make_ge_env(env_id, args.n_envs)

        def sample_traj_fn(env_id):
            trajs = env_id2trajs[env_id]
            return trajs[np.random.choice(len(trajs))]

        buf = GEBuffer(env, args.n_steps, sample_traj_fn=partial(sample_traj_fn, env_id=env_id), device=args.device)
        bufs.append(buf)
    mbuf = MultiBuffer(bufs)

    print("Creating agent")
    agent = utils.create_agent("gpt", 18, args.ctx_len, load_agent=None, device=args.device)
    opt = agent.create_optimizer(lr=args.lr)

    print("Starting learning")
    for i_iter in tqdm(range(args.n_iters)):
        for buf in bufs:
            buf.gecollect()

        for i_update in tqdm(range(args.n_updates)):
            assert args.batch_size % (len(args.env_ids) * args.n_envs) == 0
            batch = mbuf.generate_batch(args.batch_size, args.ctx_len)
            obs, act = batch["obs"], batch["act"]
            obs = obs[:, :, None, :, :]
            logits, values = agent(None, obs, act, None)

            loss_bc = torch.nn.functional.cross_entropy(rearrange(logits, "b t d -> (b t) d"), rearrange(act, "b t -> (b t)"), reduction="none")
            loss_bc = rearrange(loss_bc, "(b t) -> b t", b=128)
            loss = loss_bc.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            # if i % 50 == 0:
            # print(np.e ** loss.item())


# data["trajs"] = np.array([np.array(cell.trajectory, dtype=np.uint8) for cell in archive.values()], dtype=object)
# data["rets"] = np.array([cell.running_ret for cell in archive.values()])
# data["scores"] = np.array([cell.score for cell in archive.values()])


if __name__ == "__main__":
    main(parse_args())

# TODO do goexplore runs with different seeds
