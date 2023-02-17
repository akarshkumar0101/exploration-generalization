# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy
import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from functools import partial

import gymnasium as gym
import matplotlib.pyplot as plt
import miner
import models
import numpy as np
import ppo_rnd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from einops import rearrange

# TODO rename to pretrain.py

def viz_stuff(args, env, pbar, **kwargs):
    data = dict()
    data['charts/learning_rate'] = kwargs['optimizer'].param_groups[0]["lr"]
    data['losses/value_loss'] = kwargs['v_loss'].item()
    data['losses/policy_loss'] = kwargs['pg_loss'].item()
    data['losses/entropy'] = kwargs['entropy_loss'].item()
    data['losses/old_approx_kl'] = kwargs['old_approx_kl'].item()
    data['losses/approx_kl'] = kwargs['approx_kl'].item()
    data['losses/clipfrac'] = np.mean(kwargs['clipfracs'])
    # data['losses/explained_variance'] = kwargs['explained_var']
    data['charts/SPS'] = kwargs['sps']

    first_obs = env.envs[0].first_obs.copy()
    calc_traj_cov = lambda o:(o.std(axis=0).mean(axis=-1)>0).sum()/first_obs.mean(axis=-1).size
    
    if np.all([e.past_traj_obs is not None for e in env.envs]):
        traj_cov = np.array([calc_traj_cov(e.past_traj_obs) for e in env.envs])
        full_cov = calc_traj_cov(np.concatenate([e.past_traj_obs for e in env.envs]))
        data['avg coverage w/ one trajectory'] = traj_cov.mean()
        data[f'coverage w/ {env.num_envs} trajectories'] = full_cov
        traj_returns = np.array([e.past_returns[-1] for e in env.envs])
        traj_lens = np.array([len(e.past_traj_obs) for e in env.envs])
        data['traj_returns_hist'] = wandb.Histogram(traj_returns)
        data['traj_lens_hist'] = wandb.Histogram(traj_lens)
        data['traj_returns_mean'] = traj_returns.mean()
        data['traj_lens_mean'] = traj_lens.mean()

    data['rewards.mean'] = kwargs['rewards'].mean().item()
    data['curiosity_rewards.mean'] = kwargs['curiosity_rewards'].mean().item()

    if args.track and kwargs['update']%10==0:
        plt.figure(figsize=(10, 7))
        plt.subplot(221)
        plt.imshow(env.envs[0].first_obs)
        plt.subplot(222)
        o = kwargs['b_obs'][:, -1].cpu().numpy().std(axis=0).mean(axis=-1)
        plt.imshow(o)
        plt.tight_layout()
        data['state distribution'] = wandb.Image(plt.gcf())
        plt.close('all')
        
    video = env.envs[0].past_traj_obs
    if args.track and video is not None and kwargs['update']%10==0:
        data['video'] = wandb.Video(rearrange(video, 't h w c->t c h w'), fps=15)
        
    pbar.set_postfix({key: val for key, val in data.items() if isinstance(val, (int, float))})
    if args.track:
        wandb.log(data)

def main():
    parser = ppo_rnd.parse_args()
    args = parser.parse_args()
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        
    env = miner.make_env(args.num_envs, level_id=args.seed, video_folder=None)
    agent = models.Agent(env)
    rnd = models.RNDModel(env, (64, 64, 3))
    n_params = np.sum([p.numel() for p in agent.parameters()])
    print(f'Agent # parameters: {n_params:012d}')
    n_params = np.sum([p.numel() for p in rnd.parameters()])
    print(f'RND   # parameters: {n_params:012d}')

    pbar = tqdm(total=int(args.total_timesteps//(args.num_envs*args.num_steps)))
    callback = partial(viz_stuff, args=args, env=env, pbar=pbar)
    ppo_rnd.run(agent, rnd, env, args, callback_fn=callback)

if __name__=="__main__":
    main()