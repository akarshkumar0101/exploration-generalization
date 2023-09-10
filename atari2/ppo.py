# Adapted from CleanRL's ppo_atari_envpool.py
import argparse
import os
import random
import time

import hns
import numpy as np
import torch
from my_agents import *
from my_buffers import *
from my_envs import *
from torch import nn
from tqdm.auto import tqdm

import wandb

mystr = lambda x: None if x.lower() == "none" else x
mybool = lambda x: x.lower() == "true"
myint = lambda x: int(float(x))

parser = argparse.ArgumentParser()
# Wandb arguments
parser.add_argument("--track", type=mybool, default=False)
parser.add_argument("--entity", type=mystr, default=None)
parser.add_argument("--project", type=mystr, default=None)
parser.add_argument("--name", type=mystr, default=None)

# General arguments
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)

# Model arguments
parser.add_argument("--model", type=str, default="cnn_4")
parser.add_argument("--load_ckpt", type=mystr, default=None)
parser.add_argument("--save_ckpt", type=mystr, default=None)
parser.add_argument("--n_ckpts", type=int, default=1)

# Optimizer arguments
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--clip_grad_norm", type=float, default=1.0)

# Algorithm arguments
parser.add_argument("--env_ids", type=str, nargs="+", default=["Pong"])
parser.add_argument("--n_iters", type=myint, default=10000)
parser.add_argument("--n_envs", type=int, default=8)
parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_updates", type=int, default=16)

# PPO arguments
parser.add_argument("--episodic_life", type=mybool, default=True)
parser.add_argument("--norm_rew", type=mybool, default=True)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--norm_adv", type=mybool, default=True)
parser.add_argument("--ent_coef", type=float, default=0.001)
parser.add_argument("--clip_coef", type=float, default=0.1)
parser.add_argument("--clip_vloss", type=lambda x: mybool, default=True)
parser.add_argument("--vf_coef", type=float, default=0.5)
# parser.add_argument("--max_kl_div", type=lambda x: None if x.lower == "none" else float(x), default=None)


def parse_args(*args, **kwargs):
    args, uargs = parser.parse_known_args(*args, **kwargs)
    for k, v in dict([tuple(uarg.replace("--", "").split("=")) for uarg in uargs]).items():
        setattr(args, k, v)
    return args


def calc_ppo_policy_loss(dist, dist_old, act, adv, norm_adv=True, clip_coef=0.1):
    # can be called with dist or logits
    if isinstance(dist, torch.Tensor):
        dist = torch.distributions.Categorical(logits=dist)
    if isinstance(dist_old, torch.Tensor):
        dist_old = torch.distributions.Categorical(logits=dist_old)
    if norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = (dist.log_prob(act) - dist_old.log_prob(act)).exp()
    loss_pg1 = -adv * ratio
    loss_pg2 = -adv * ratio.clamp(1 - clip_coef, 1 + clip_coef)
    loss_pg = torch.max(loss_pg1, loss_pg2)
    return loss_pg


def calc_ppo_value_loss(val, val_old, ret, clip_coef=0.1):
    if clip_coef is not None:
        loss_v_unclipped = (val - ret) ** 2
        v_clipped = val_old + (val - val_old).clamp(-clip_coef, clip_coef)
        loss_v_clipped = (v_clipped - ret) ** 2
        loss_v_max = torch.max(loss_v_unclipped, loss_v_clipped)
        loss_v = 0.5 * loss_v_max
    else:
        loss_v = 0.5 * ((val - ret) ** 2)
    return loss_v


def make_env(args):
    envs = []
    for env_id in args.env_ids:
        envi = MyEnvpool(f"{env_id}-v5", num_envs=args.n_envs, stack_num=1, episodic_life=True, reward_clip=True, seed=args.seed, full_action_space=True)
        if args.norm_rew:
            envi = MyNormalizeReward(envi, gamma=args.gamma)
        envi = RecordEpisodeStatistics(envi, deque_size=32)
        envs.append(envi)
    env = ConcatEnv(envs)
    env = ToTensor(env, device=args.device)
    return env


def main(args):
    print("Running PPO with args: ", args)
    print("Starting wandb...")
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    print("Seeding...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Creating environment...")
    env = make_env(args)

    print("Creating agent...")
    agent = make_agent(args.model, 18).to(args.device)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    if args.load_ckpt is not None:
        print("Loading checkpoint...")
        ckpt = torch.load(args.load_ckpt, map_location=args.device)
        agent.load_state_dict(ckpt["agent"])

    print("Creating buffer...")
    buffer = Buffer(env, agent, args.n_steps, device=args.device)

    print("Warming up buffer...")
    for i_iter in tqdm(range(40), leave=False):
        buffer.collect()

    start_time = time.time()
    print("Starting learning...")
    for i_iter in tqdm(range(args.n_iters)):
        buffer.collect()
        buffer.calc_gae(gamma=args.gamma, gae_lambda=args.gae_lambda, episodic=True)

        for _ in range(args.n_updates):
            batch = buffer.generate_batch(args.batch_size, ctx_len=agent.ctx_len)

            logits, val = agent(done=batch["done"], obs=batch["obs"], act=batch["act"], rew=batch["rew"])
            dist, batch_dist = torch.distributions.Categorical(logits=logits), torch.distributions.Categorical(logits=batch["logits"])

            loss_p = calc_ppo_policy_loss(dist, batch_dist, batch["act"], batch["adv"], norm_adv=args.norm_adv, clip_coef=args.clip_coef)
            loss_v = calc_ppo_value_loss(val, batch["val"], batch["ret"], clip_coef=args.clip_coef if args.clip_vloss else None)
            loss_e = dist.entropy()

            if not agent.train_per_token:
                loss_p, loss_v, loss_e = loss_p[:, [-1]], loss_v[:, [-1]], loss_e[:, [-1]]
            loss = 1.0 * loss_p.mean() + args.vf_coef * loss_v.mean() - args.ent_coef * loss_e.mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.clip_grad_norm)
            opt.step()

        if args.save_ckpt is not None and args.n_ckpts > 0 and (i_iter + 1) % (args.n_iters // args.n_ckpts) == 0:
            print(f"Saving Checkpoint at {i_iter}/{args.n_iters} iterations")
            file = args.save_ckpt.format(i_iter=i_iter)
            ckpt = dict(args=args, i_iter=i_iter, agent=agent.state_dict())
            os.makedirs(os.path.dirname(file), exist_ok=True)
            torch.save(ckpt, file)

        viz_slow = i_iter % np.clip(args.n_iters // 10, 1, None) == 0
        viz_midd = i_iter % np.clip(args.n_iters // 100, 1, None) == 0 or viz_slow
        viz_fast = i_iter % np.clip(args.n_iters // 1000, 1, None) == 0 or viz_midd

        data = {}
        if viz_fast:
            for envi in env.envs:
                data[f"charts/{envi.env_id}_score"] = np.mean(envi.traj_rets)
                data[f"charts/{envi.env_id}_tlen"] = np.mean(envi.traj_lens)
                data[f"charts/{envi.env_id}_score_max"] = np.max(envi.traj_rets)
                low, high = hns.atari_human_normalized_scores[envi.env_id]
                data["charts/hns"] = (np.mean(envi.traj_rets) - low) / (high - low)

            env_steps = (i_iter + 1) * len(args.env_ids) * args.n_envs * args.n_steps
            grad_steps = (i_iter + 1) * args.n_updates
            data["env_steps"] = env_steps
            data["grad_steps"] = grad_steps
            sps = int(env_steps / (time.time() - start_time))
            data["meta/SPS"] = sps
        if args.track and viz_fast:
            wandb.log(data)


if __name__ == "__main__":
    main(parse_args())
