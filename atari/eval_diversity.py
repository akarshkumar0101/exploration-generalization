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

    for i in range(10):
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


if __name__ == "__main__":
    main(None)
