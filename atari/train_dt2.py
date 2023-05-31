import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from agent_atari import Agent, Encoder
from buffers import Buffer, MultiBuffer
from decision_transformer import Config, DecisionTransformer
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

# Algorithm specific arguments
parser.add_argument("--env-id", type=str, default="Pong", help="the id of the environment")
parser.add_argument("--n-envs", type=int, default=64, help="the number of parallel game environments")
parser.add_argument("--n-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--batch-size", type=int, default=1024, help="the number of mini-batches")

parser.add_argument("--ctx-len", type=int, default=4, help="context length of the transformer")
parser.add_argument("--n-iters", type=int, default=1000, help="the K epochs to update the policy")
parser.add_argument("--freq-collect", type=int, default=10, help="context length of the transformer")

parser.add_argument("--lr", type=float, default=6e-4, help="the learning rate of the optimizer")
parser.add_argument("--lr-min", type=float, default=6e-5, help="the learning rate of the optimizer")
parser.add_argument("--lr-schedule", type=lambda x: bool(strtobool(x)), default=True, help="Toggle learning rate annealing for policy and value networks")

parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")

parser.add_argument("--frame-stack", type=int, default=4, help="the number of frames to stack as input to the model")
parser.add_argument("--obj", type=str, default="ext", help="the objective of the agent, either ext or e3b")

parser.add_argument("--load-agent", type=str, default=None, help="file to load the agent from")
parser.add_argument("--save-agent", type=str, default=None, help="file to periodically save the agent to")


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    if args.project is not None:
        args.project = args.project.format(**vars(args))
    if args.name is not None:
        args.name = args.name.format(**vars(args))
    if args.load_agent is not None:
        args.load_agent = args.load_agent.format(**vars(args))
    if args.save_agent is not None:
        args.save_agent = args.save_agent.format(**vars(args))
    args.collect_size = int(args.n_envs * args.n_steps)
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

    config = Config(n_steps_max=args.ctx_len, n_actions=18, n_layer=4, n_head=4, n_embd=4 * 16, dropout=0.0, bias=False)
    dtgpt = DecisionTransformer(config).to(args.device)
    print("Printing DTGPT Summary...")
    torchinfo.summary(
        dtgpt,
        input_size=[(args.batch_size, args.ctx_len), (args.batch_size, args.ctx_len, 1, 84, 84), (args.batch_size, args.ctx_len)],
        dtypes=[torch.float, torch.float, torch.long],
        device=args.device,
    )

    # print(sum(p.numel() for p in agent.parameters()))
    print(sum(p.numel() for p in dtgpt.parameters()))

    # opt = torch.optim.Adam(dtgpt.parameters(), lr=args.lr)
    opt = dtgpt.configure_optimizers(0.0, args.lr, (0.9, 0.95), args.device)

    env_ids = ["Boxing", "Breakout", "Pong", "SpaceInvaders"]
    agents = []
    envs = []
    envs2 = []
    for i in range(4):
        agent = Agent((4, 84, 84), 18).to(args.device)
        agent.load_state_dict(torch.load(f"../data/egb-atari-1/{env_ids[i]}_ext_0/agent.pt"))
        agents.append(agent)
        env = make_env(env_ids[i], n_envs=128 // 4, frame_stack=1, obj=args.obj, e3b_encode_fn=None, gamma=args.gamma, device=args.device, seed=args.seed, buf_size=args.n_steps)
        envs.append(env)
        env = make_env(env_ids[i], n_envs=32 // 4, frame_stack=1, obj=args.obj, e3b_encode_fn=None, gamma=args.gamma, device=args.device, seed=args.seed, buf_size=args.n_steps)
        envs2.append(env)
    multibuffer = MultiBuffer(args.n_steps, 128, args.ctx_len, envs=envs, device=args.device)
    multibuffer_test = MultiBuffer(args.n_steps, 32, args.ctx_len, envs=envs2, device=args.device)

    viz_slow = set(np.linspace(0, args.n_iters - 1, 10).astype(int))
    viz_midd = set(np.linspace(0, args.n_iters - 1, 100).astype(int)).union(viz_slow)
    viz_fast = set(np.linspace(0, args.n_iters - 1, 1000).astype(int)).union(viz_midd)

    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        data = {}
        lr = get_lr(args, i_iter)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        if i_iter % args.freq_collect == 0:
            print("Collecting expert data...")
            multibuffer.collect(agents)
            for i_env, env_id in enumerate(env_ids):
                data[f"{env_id}_student_ret_ext"] = torch.cat(multibuffer.envs[i_env].key2past_rets["ret_ext"]).mean().item()

            print("Collecting student data...")
            multibuffer_test.collect([dtgpt for _ in range(4)])
            for i_env, env_id in enumerate(env_ids):
                data[f"{env_id}_dtgpt_ret_ext"] = torch.cat(multibuffer_test.envs[i_env].key2past_rets["ret_ext"]).mean().item()

        batch = multibuffer.generate_batch(args.batch_size)

        dtgpt.train()
        obs, dist_teacher, act = batch["obs"], batch["dist"], batch["act"]
        dist_student, values = dtgpt.forward(rtg=None, obs=obs, act=act)

        kl_div = torch.distributions.kl.kl_divergence(dist_student, dist_teacher)
        entropy = dist_student.entropy()
        loss_kl, loss_entropy = kl_div.mean(), entropy.mean()
        loss = loss_kl + args.ent_coef * loss_entropy

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(dtgpt.parameters(), args.max_grad_norm)
        opt.step()

        data["loss_kl"] = loss_kl.item()
        data["loss_entropy"] = loss_entropy.item()
        data["loss_kl_0"] = kl_div[:, 0].mean().item()
        data["loss_kl_-1"] = kl_div[:, -1].mean().item()
        data["lr"] = lr

        keys_tqdm = ["loss_kl", "loss_entropy", "Breakout_student_ret_ext", "Breakout_student_dtgpt_ext"]
        pbar.set_postfix({k.split("/")[-1]: data[k] for k in keys_tqdm if k in data})
        if args.track and i_iter in viz_fast:
            wandb.log(data, step=i_iter)


if __name__ == "__main__":
    args_hehe = parse_args()
    print(args_hehe)
    main(args_hehe)
