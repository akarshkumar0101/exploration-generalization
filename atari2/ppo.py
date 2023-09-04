# Adapted from CleanRL's ppo_atari_envpool.py
import argparse
import os
import random
import time

import envpool
import gym
import numpy as np
import torch
from my_agents import *
from my_buffers import *
from my_envs import *
import hns
from torch import nn
from tqdm.auto import tqdm

import wandb

mystr = lambda x: None if x.lower() == "none" else x
mybool = lambda x: x.lower() == "true"
myint = lambda x: int(float(x))

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=mybool, default=False)
parser.add_argument("--entity", type=mystr, default=None)
parser.add_argument("--project", type=mystr, default=None)
parser.add_argument("--name", type=mystr, default=None)

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)

# Algorithm arguments
parser.add_argument("--env_ids", type=str, nargs="+", default=["Pong"])
parser.add_argument("--n_iters", type=myint, default=10000)
parser.add_argument("--n_envs", type=int, default=8)
parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--n_updates", type=int, default=16)

parser.add_argument("--model", type=str, default="stacked_cnn")
parser.add_argument("--ctx_len", type=int, default=4)
# parser.add_argument("--load_agent_history", type=lambda x: bool(strtobool(x)), default=False)
# parser.add_argument("--load_agent", type=nonestr, help="file to load the agent from")
# parser.add_argument("--save_agent", type=nonestr, help="file to periodically save the agent to")
# parser.add_argument("--full_action_space", type=lambda x: bool(strtobool(x)), default=True)

parser.add_argument("--lr", type=float, default=2.5e-4)
# parser.add_argument("--lr_warmup", type=lambda x: bool(strtobool(x)), default=True)
# parser.add_argument("--lr_decay", type=str, default="none")
parser.add_argument("--max_grad_norm", type=float, default=1.0)

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

# parser.add_argument("--pre_obj", type=str, default="ext")
# parser.add_argument("--train_klbc", type=lambda x: bool(strtobool(x)), default=False)
# parser.add_argument("--model_teacher", type=str, default="cnn")
# parser.add_argument("--ctx_len_teacher", type=int, default=4)
# parser.add_argument("--load_agent_teacher", type=lambda x: None if x.lower() == "none" else x, default=None)
# parser.add_argument("--teacher_last_k", type=int, default=1)
# parser.add_argument("--pre_teacher_last_k", type=int, default=1)


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


def main(args):
    print("Running PPO with args: ", args)
    print("Starting wandb...")
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    print("Seeding...")
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    print("Creating environment...")
    envs = []
    for env_id in args.env_ids:
        envi = MyEnvpool(f"{env_id}-v5", num_envs=args.n_envs, stack_num=1, episodic_life=True, seed=args.seed, full_action_space=True)
        envi = gym.wrappers.NormalizeReward(envi, gamma=args.gamma)
        envi = RecordEpisodeStatistics(envi)
        envi = ToTensor(envi)
        envs.append(envi)
    env = ConcatEnv(envs)

    print("Creating agent...")
    agent = create_agent(args.model, 18, args.ctx_len)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    print("Creating buffer...")
    buffer = Buffer(env, agent, args.n_steps, device="cpu")

    start_time = time.time()
    pbar_iters = tqdm(total=args.n_iters)
    pbar_steps = tqdm(total=args.n_steps)
    pbar_updates = tqdm(total=args.n_updates)

    for i_iter in range(args.n_iters):
        pbar_steps.reset()
        buffer.collect(pbar_steps)
        buffer.calc_gae(gamma=args.gamma, gae_lambda=args.gae_lambda, episodic=True)

        pbar_updates.reset()
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
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            opt.step()
            pbar_updates.update(1)

        viz_slow = i_iter % np.clip(args.n_iters // 10, 1, None) == 0
        viz_midd = i_iter % np.clip(args.n_iters // 100, 1, None) == 0 or viz_slow
        viz_fast = i_iter % np.clip(args.n_iters // 1000, 1, None) == 0 or viz_midd
        to_np = lambda x: x.detach().cpu().numpy()

        data = {}
        if viz_fast:
            print("logging! ", i_iter, args.n_iters)
            for envi in envs:
                data["charts/avg_episodic_return"] = np.mean(envi.traj_rets)
                data["charts/episodic_length"] = np.mean(envi.traj_lens)
                low, high = hns.atari_human_normalized_scores[envi.env_id]
                data["charts/hns"] = (np.mean(envi.traj_rets) - low) / (high - low)

            env_steps = len(args.env_ids) * args.n_steps * args.n_envs * i_iter
            data["charts/env_steps"] = env_steps
            sps = int(env_steps / (time.time() - start_time))
            data["charts/SPS"] = sps
        if args.track and viz_fast:
            wandb.log(data)
        pbar_iters.update(1)


if __name__ == "__main__":
    main(parse_args())
