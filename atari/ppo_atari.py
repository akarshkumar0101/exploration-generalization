# Adapted from CleanRL's ppo_atari_envpool.py
import argparse
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
from einops import rearrange
from env_atari import make_env
from time_contrastive import calc_contrastive_loss, sample_contrastive_batch
from tqdm.auto import tqdm

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--project", type=str, default=None, help="the wandb's project name")
parser.add_argument("--name", type=str, default=None, help="the name of this experiment")

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")

parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
parser.add_argument("--log-video", type=lambda x: bool(strtobool(x)), default=False)

# Algorithm specific arguments
parser.add_argument("--env-id", type=str, default="Pong", help="the id of the environment")
parser.add_argument("--total-steps", type=lambda x: int(float(x)), default=10000000, help="total timesteps of the experiments")
parser.add_argument("--n-envs", type=int, default=8, help="the number of parallel game environments")
parser.add_argument("--n-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--batch-size", type=int, default=256, help="the number of mini-batches")
parser.add_argument("--n-epochs", type=int, default=4, help="the K epochs to update the policy")
parser.add_argument("--lr", type=float, default=2.5e-4, help="the learning rate of the optimizer")
parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, help="Toggle learning rate annealing for policy and value networks")
parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, help="Toggles advantages normalization")
parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")

parser.add_argument("--frame-stack", type=int, default=4, help="the number of frames to stack as input to the model")
parser.add_argument("--obj", type=str, default="ext", help="the objective of the agent, either ext or e3b")
parser.add_argument("--lr-tc", type=float, default=3e-4, help="learning rate for time contrastive encoder")


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)

    args.track = args.project is not None
    if args.project is not None:
        args.project = args.project.format(**vars(args))
    if args.name is not None:
        args.name = args.name.format(**vars(args))

    args.collect_size = int(args.n_envs * args.n_steps)
    args.n_updates = args.total_steps // args.collect_size
    return args


def main(args):
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    encoder = Encoder((1, 84, 84), 64).to(args.device)
    e3b_encode_fn = lambda obs: encoder.encode(obs[:, [-1]])  # encode only the latest frame
    env = make_env(args.env_id, n_envs=args.n_envs, frame_stack=args.frame_stack, obj=args.obj, e3b_encode_fn=e3b_encode_fn, gamma=args.gamma, device=args.device, seed=args.seed)
    assert isinstance(env.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(env.single_observation_space.shape, env.single_action_space.n).to(args.device)
    torchinfo.summary(agent, input_size=(args.batch_size,) + env.single_observation_space.shape, device=args.device)
    # optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    optimizer = optim.Adam([{"params": agent.parameters(), "lr": args.lr, "eps": 1e-5}, {"params": encoder.parameters(), "lr": args.lr_tc, "eps": 1e-8}])

    obs = torch.zeros((args.n_steps, args.n_envs) + env.single_observation_space.shape, dtype=torch.uint8, device=args.device)
    actions = torch.zeros((args.n_steps, args.n_envs) + env.single_action_space.shape, dtype=torch.uint8, device=args.device)
    logprobs = torch.zeros((args.n_steps, args.n_envs), device=args.device)
    rewards = torch.zeros((args.n_steps, args.n_envs), device=args.device)
    dones = torch.zeros((args.n_steps, args.n_envs), dtype=torch.bool, device=args.device)
    values = torch.zeros((args.n_steps, args.n_envs), device=args.device)

    start_time = time.time()
    _, info = env.reset()
    next_obs = info["obs"]
    next_done = torch.zeros(args.n_envs, dtype=torch.bool, device=args.device)

    viz_slow = set(np.linspace(0, args.n_updates - 1, 10).astype(int))
    viz_midd = set(np.linspace(0, args.n_updates - 1, 100).astype(int)).union(viz_slow)
    viz_fast = set(np.linspace(0, args.n_updates - 1, 1000).astype(int)).union(viz_midd)

    pbar = tqdm(range(args.n_updates))
    for i_update in pbar:
        if args.anneal_lr:  # Annealing the rate if instructed to do so.
            frac = 1.0 - (i_update) / args.n_updates
            lrnow = frac * args.lr
            optimizer.param_groups[0]["lr"] = lrnow

        for i_step in range(args.n_steps):
            obs[i_step] = next_obs
            dones[i_step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[i_step] = value.flatten()
            actions[i_step] = action
            logprobs[i_step] = logprob

            _, reward, _, _, info = env.step(action.cpu().numpy())
            next_obs, next_done = info["obs"], info["done"]
            rewards[i_step] = torch.as_tensor(reward).to(args.device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(args.device)
            lastgaelam = 0
            for t in reversed(range(args.n_steps)):
                if t == args.n_steps - 1:
                    nextnonterminal = ~next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = ~dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.collect_size)
        clipfracs = []
        for i_epoch in range(args.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.collect_size, args.batch_size):
                end = start + args.batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + (newvalue - b_values[mb_inds]).clamp(-args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                obs_anc, obs_pos, obs_neg = sample_contrastive_batch(obs[:, :, -1, :, :], p=0.1, batch_size=args.batch_size)
                loss_tc = calc_contrastive_loss(encoder, obs_anc, obs_pos, obs_neg)
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + loss_tc

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        data = {}
        if i_update in viz_fast:  # log ex. points with fast frequency
            data["details/lr"] = optimizer.param_groups[0]["lr"]
            data["details/value_loss"] = v_loss.item()
            data["details/policy_loss"] = pg_loss.item()
            data["details/entropy"] = entropy_loss.item()
            data["details/old_approx_kl"] = old_approx_kl.item()
            data["details/approx_kl"] = approx_kl.item()
            data["details/clipfrac"] = np.mean(clipfracs)
            data["details/explained_variance"] = explained_var
            data["meta/SPS"] = int(i_update * args.collect_size / (time.time() - start_time))
            data["meta/global_step"] = i_update * args.collect_size

            data["details/loss_tc"] = loss_tc.item()

            for key, val in env.key2past_rets.items():
                rets = torch.cat(val).tolist()
                if len(rets) > 0:
                    data[f"charts/{key}"] = np.mean(rets)
                    data[f"charts_hist/{key}"] = wandb.Histogram(rets)  # TODO move to viz_midd
        if i_update in viz_midd:  # log ex. histograms with midd frequency
            data["details_hist/entropy"] = wandb.Histogram(entropy.detach().cpu().numpy())
            # data["details_hist/action"] = wandb.Histogram(b_actions.detach().cpu().numpy())
        if i_update in viz_slow:  # log ex. videos with slow frequency
            vid = np.stack(env.past_obs).copy()
            vid[:, :, -1, :] = 0
            vid[:, :, :, -1] = 0
            vid = rearrange(vid, "t (H W) h w -> t (H h) (W w)", H=2, W=4)
            data["media/vid"] = wandb.Video(rearrange(vid, "t h w -> t 1 h w"), fps=15)

        keys_tqdm = ["charts/ret_ext", "charts/ret_e3b", "meta/SPS"]
        pbar.set_postfix({k.split("/")[-1]: data[k] for k in keys_tqdm if k in data})
        if args.track:
            wandb.log(data, step=i_update * args.collect_size)
        plt.close("all")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
