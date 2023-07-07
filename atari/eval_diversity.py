import argparse
import os
import random
import time
from distutils.util import strtobool

import agent_atari
import atari_data
import buffers
import lpips
import numpy as np
import timers
import torch
import torchinfo
import utils
from einops import rearrange
from env_atari import make_env
from torch.distributions import Categorical
from tqdm.auto import tqdm

from einops import repeat
import wandb


def main(args):
    env = make_env(env_id="MontezumaRevenge", n_envs=8, obj="ext")
    buffer = buffers.Buffer(env, 128)
    agent = agent_atari.RandomAgent(18)

    loss_fn_alex = lpips.LPIPS(net="alex")

    encoders = [agent_atari.NatureCNN(1, n_dim=128, normalize=False) for _ in range(2)]

    scores_lpips = []
    scores_random = [[] for _ in range(len(encoders))]

    for i in range(1):
        buffer.collect(agent, 1)
        for _ in tqdm(range(16)):
            batch1 = buffer.generate_batch(256, 1)
            batch2 = buffer.generate_batch(256, 1)  # b 1 1 84 84
            obs1 = repeat(batch1["obs"], "b 1 1 ... -> b 3 ...")
            obs2 = repeat(batch2["obs"], "b 1 1 ... -> b 3 ...")

            with torch.no_grad():
                d = loss_fn_alex(obs1 / 255.0 * 2.0 - 1.0, obs2 / 255.0 * 2.0 - 1.0).detach().cpu().numpy().flatten()
                scores_lpips.append(d)
                for i_encoder, encoder in enumerate(encoders):
                    d = (encoder(obs1[:, [0]]) - encoder(obs2[:, [0]])).norm(dim=-1).detach().cpu().numpy()
                    scores_random[i_encoder].append(d)

    scores_lpips = np.array(scores_lpips)
    scores_random = np.array(scores_random)

    print(scores_lpips.shape)
    print(scores_random.shape)
    print(scores_lpips.mean(), scores_lpips.std())
    print(scores_random.mean(axis=(-1, -2)), scores_random.std(axis=(-1, -2)))


loss_fn_alex = lpips.LPIPS(net="alex")


@torch.no_grad()
def calc_diversity(buffer, n_iters=16, batch_size=256, device=None):
    loss_fn_alex.to(device)
    scores = []
    for _ in range(n_iters):
        batch1 = buffer.generate_batch(batch_size, 1)  # b 1 1 84 84
        batch2 = buffer.generate_batch(batch_size, 1)  # b 1 1 84 84
        obs1 = repeat(batch1["obs"], "b 1 1 ... -> b 3 ...") / 255.0 * 2.0 - 1.0
        obs2 = repeat(batch2["obs"], "b 1 1 ... -> b 3 ...") / 255.0 * 2.0 - 1.0

        d = loss_fn_alex(obs1, obs2).flatten()
        scores.append(d)
    scores = torch.stack(scores)
    return scores


env_id2ram_var = {}


def register_envid_ram(env_id):
    if env_id not in env_id2ram_var:
        env = make_env(env_id, n_envs=16, lib="gymnasium")
        buffer_random = buffers.Buffer(env, 16)
        agent_random = agent_atari.RandomAgent(18)
        buffer_random.collect(agent_random, 1)
        ram_var = buffer_random.rams.float().var(dim=(0, 1)).clamp(1, None)
        env_id2ram_var[env_id] = ram_var


@torch.no_grad()
def calc_div_traj_ram(buffer, env_id):
    register_envid_ram(env_id)
    ram_var = env_id2ram_var[env_id].to(buffer.device)
    batch = buffer.generate_batch(buffer.env.num_envs, buffer.n_steps)
    return (batch["ram"].float().var(dim=1) / ram_var).mean(dim=0).mean().item()


@torch.no_grad()
def calc_div_buff_ram(buffer, env_id):
    register_envid_ram(env_id)
    ram_var = env_id2ram_var[env_id].to(buffer.device)
    batch = buffer.generate_batch(buffer.env.num_envs, buffer.n_steps)
    return (batch["ram"].float().var(dim=(0, 1)) / ram_var).mean().item()


@torch.no_grad()
def calc_div_traj_pix(buffer):
    batch = buffer.generate_batch(buffer.env.num_envs, buffer.n_steps)
    return (batch["obs"].float().var(dim=1)).mean(dim=0).mean().item()


@torch.no_grad()
def calc_div_buff_pix(buffer):
    batch = buffer.generate_batch(buffer.env.num_envs, buffer.n_steps)
    return (batch["obs"].float().var(dim=(0, 1))).mean().item()


@torch.no_grad()
def calc_div_traj_lpips(buffer):
    batch = buffer.generate_batch(buffer.env.num_envs, buffer.n_steps)
    i1, i2 = torch.randint(low=0, high=buffer.n_steps, size=(2,))
    obs1 = repeat(batch["obs"][:, i1], "b 1 ... -> b 3 ...") / 255.0 * 2.0 - 1.0
    obs2 = repeat(batch["obs"][:, i2], "b 1 ... -> b 3 ...") / 255.0 * 2.0 - 1.0
    loss_fn_alex = loss_fn_alex.to(obs1.device)
    return loss_fn_alex(obs1, obs2).flatten().mean().item()


@torch.no_grad()
def calc_div_buff_lpips(buffer):
    batch = buffer.generate_batch(buffer.env.num_envs, buffer.n_steps)
    i1, i2 = torch.randint(low=0, high=buffer.env.num_envs * buffer.n_steps, size=(2, buffer.env.num_envs))
    obs1 = repeat(batch["obs"].flatten(0, 1)[i1], "b 1 ... -> b 3 ...") / 255.0 * 2.0 - 1.0
    obs2 = repeat(batch["obs"].flatten(0, 1)[i2], "b 1 ... -> b 3 ...") / 255.0 * 2.0 - 1.0
    loss_fn_alex = loss_fn_alex.to(obs1.device)
    return loss_fn_alex(obs1, obs2).flatten().mean().item()


if __name__ == "__main__":
    main(None)
