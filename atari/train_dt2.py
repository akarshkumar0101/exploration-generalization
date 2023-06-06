import argparse
import os
import random
import time
from distutils.util import strtobool

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchinfo
from agent_atari import CNNAgent
from buffers import Buffer, MultiBuffer
from decision_transformer import DecisionTransformer
from einops import rearrange
from env_atari import make_env
from tqdm.auto import tqdm

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--project", type=str, default=None, help="the wandb's project name")
parser.add_argument("--name", type=str, default=None, help="the name of this experiment")

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")

parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
parser.add_argument("--log-video", type=lambda x: bool(strtobool(x)), default=False)

# Algorithm arguments
parser.add_argument("--env-ids", type=str, nargs="+", default=["Pong"], help="the id of the environment")
parser.add_argument("--env-ids-test", type=str, nargs="+", default=[], help="the id of the environment")
parser.add_argument("--obj", type=str, default="ext", help="the objective of the agent, either ext or e3b")
parser.add_argument("--ctx-len", type=int, default=4, help="agent's context length")
parser.add_argument("--total-steps", type=lambda x: int(float(x)), default=10000000, help="total timesteps of the experiments")
parser.add_argument("--n-envs", type=int, default=64, help="the number of parallel game environments")
parser.add_argument("--n-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--batch-size", type=int, default=1024, help="the batch size for training")
parser.add_argument("--n-updates", type=int, default=16, help="gradient updates per collection")
parser.add_argument("--lr", type=float, default=6e-4, help="the learning rate of the optimizer")
parser.add_argument("--lr-schedule", type=lambda x: bool(strtobool(x)), default=True, help="Toggle learning rate annealing for policy and value networks")
# parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, help="Toggle learning rate annealing for policy and value networks")
parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
# parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
# parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, help="Toggles advantages normalization")
parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
# parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
# parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
# parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
# parser.add_argument("--max-kl-div", type=float, default=None, help="the target KL divergence threshold")

parser.add_argument("--expert-agent", type=str, default=None, help="file to load the expert agent from")
parser.add_argument("--load-agent", type=str, default=None, help="file to load the agent from")
parser.add_argument("--save-agent", type=str, default=None, help="file to periodically save the agent to")


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    if args.project is not None:
        args.project = args.project.format(**vars(args))
    if args.name is not None:
        args.name = args.name.format(**vars(args))
    if args.expert_agent is not None:
        args.expert_agent = args.expert_agent.format(**vars(args))
    if args.load_agent is not None:
        args.load_agent = args.load_agent.format(**vars(args))
    if args.save_agent is not None:
        args.save_agent = args.save_agent.format(**vars(args))

    args.n_envs_per_id = args.n_envs // len(args.env_ids)

    args.collect_size = args.n_envs * args.n_steps
    args.total_steps = args.total_steps // args.collect_size * args.collect_size
    args.n_iters = args.total_steps // args.collect_size * args.n_updates
    args.freq_collect = args.n_updates
    return args


def get_lr(args, i_iter):
    if not args.lr_schedule:
        return args.lr

    n_iters_warmup = args.n_iters // 100
    if i_iter < n_iters_warmup:
        return args.lr * i_iter / n_iters_warmup
    if i_iter > args.n_iters:
        return args.lr_min
    decay_ratio = (i_iter - n_iters_warmup) / (args.n_iters - n_iters_warmup)
    coeff = 0.5 * (1.0 + np.math.cos(np.pi * decay_ratio))  # coeff ranges 0..1
    assert 0 <= decay_ratio <= 1 and 0 <= coeff <= 1
    return args.lr_min + coeff * (args.lr - args.lr_min)


def main(args):
    print("Running distillation with args: ", args)
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env = make_env(args.env_id, n_envs=args.n_envs, frame_stack=args.frame_stack, obj=args.obj, e3b_encode_fn=None, gamma=args.gamma, device=args.device, seed=args.seed)
    # env_test = make_env(args.env_id, n_envs=args.n_envs, frame_stack=args.frame_stack, obj=args.obj, e3b_encode_fn=None, gamma=args.gamma, device=args.device, seed=args.seed)

    # agent = Agent(env.single_observation_space.shape, env.single_action_space.n).to(args.device)
    # if args.load_agent is not None:
    # agent.load_state_dict(torch.load(args.load_agent))
    # print("Printing Base Agent Summary...")
    # torchinfo.summary(agent, input_size=(args.batch_size,) + env.single_observation_space.shape, device=args.device)

    agent = DecisionTransformer(n_steps=args.ctx_len, n_acts=18, n_layers=4, n_heads=4, n_embd=4 * 64, dropout=0.0, bias=True).to(args.device)
    print("Printing DTGPT Summary...")
    torchinfo.summary(
        agent,
        input_size=[(args.batch_size, args.ctx_len, 1, 84, 84), (args.batch_size, args.ctx_len), (args.batch_size, args.ctx_len)],
        dtypes=[torch.float, torch.float, torch.long],
        device=args.device,
    )

    # opt = torch.optim.Adam(dtgpt.parameters(), lr=args.lr)
    opt = agent.configure_optimizers(0.0, args.lr, (0.9, 0.95), args.device)

    experts = []
    mbuffer, mbuffer_test = MultiBuffer(), MultiBuffer()
    for env_id in args.env_ids:
        env = make_env(env_id, n_envs=args.n_envs_per_id, obj=args.obj, e3b_encode_fn=None, gamma=args.gamma, device=args.device, seed=args.seed, buf_size=args.n_steps)
        mbuffer.buffers.append(Buffer(args.n_steps, args.n_envs_per_id, env, device=args.device))
        expert = CNNAgent((args.ctx_len, 1, 84, 84), 18).to(args.device)
        # print("Loading expert from: ", {args.expert_agent.format(env_id=env_id)})
        # expert.load_state_dict(torch.load(args.expert_agent.format(env_id=env_id)))
        experts.append(expert)
    for env_id in args.env_ids_test:
        env = make_env(env_id, n_envs=args.n_envs_per_id, obj=args.obj, e3b_encode_fn=None, gamma=args.gamma, device=args.device, seed=args.seed, buf_size=args.n_steps)
        mbuffer_test.buffers.append(Buffer(args.n_steps, args.n_envs_per_id, env, device=args.device))

    viz_slow = set(np.linspace(0, args.n_iters - 1, 10).astype(int))
    viz_midd = set(np.linspace(0, args.n_iters - 1, 100).astype(int)).union(viz_slow)
    viz_fast = set(np.linspace(0, args.n_iters - 1, 1000).astype(int)).union(viz_midd)

    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        data = {}
        if i_iter % args.freq_collect == 0:
            print("Collecting expert data...")
            mbuffer.collect(experts, args.ctx_len)
            for env_id, env in zip(args.env_ids, mbuffer.envs):
                data[f"{env_id}_expert_ret_ext"] = torch.cat(env.key2past_rets["ret_ext"]).mean().item()

            print("Collecting student data...")
            mbuffer_test.collect(agent, args.ctx_len)
            for env_id, env in zip(args.env_ids, mbuffer_test.envs):
                data[f"{env_id}_student_ret_ext"] = torch.cat(env.key2past_rets["ret_ext"]).mean().item()

        lr = get_lr(args, i_iter)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        agent.train()
        batch = mbuffer.generate_batch(args.batch_size, args.ctx_len)
        obs, dist_teacher, act, done = batch["obs"], batch["dist"], batch["act"], batch["done"]
        dist_student, values = agent.act(obs=obs, act=act, done=done)

        kl_div = torch.distributions.kl.kl_divergence(dist_student, dist_teacher)
        entropy = dist_student.entropy()
        loss_kl, loss_entropy = kl_div.mean(), entropy.mean()
        loss = loss_kl - args.ent_coef * loss_entropy

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        opt.step()

        if i_iter in viz_fast:
            data["loss_kl"] = loss_kl.item()
            data["loss_entropy"] = loss_entropy.item()
            data["loss_kl_0"] = kl_div[:, 0].mean().item()
            data["loss_kl_-1"] = kl_div[:, -1].mean().item()
            data["lr"] = lr
        if i_iter in viz_midd:
            pass
        if i_iter in viz_slow:
            pass

        if args.track and i_iter in viz_fast:
            wandb.log(data, step=i_iter)

        keys_tqdm = ["loss_kl", "loss_entropy", "Breakout_student_ret_ext", "Breakout_student_dtgpt_ext"]
        pbar.set_postfix({k.split("/")[-1]: data[k] for k in keys_tqdm if k in data})


if __name__ == "__main__":
    main(parse_args())
