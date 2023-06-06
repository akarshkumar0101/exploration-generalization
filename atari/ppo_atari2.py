# Adapted from CleanRL's ppo_atari_envpool.py
import argparse
import os
import random
import time
from distutils.util import strtobool

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from agent_atari import CNNAgent
from buffers import Buffer, MultiBuffer
from einops import rearrange
from env_atari import make_env

# from time_contrastive import calc_contrastive_loss, sample_contrastive_batch
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
parser.add_argument("--env-id", type=str, default="Pong", help="the id of the environment")
# parser.add_argument("--env-ids", type=str, nargs="+", default=["Pong"], help="the id of the environment")
# parser.add_argument("--env-ids-test", type=str, nargs="+", default=[], help="the id of the environment")
parser.add_argument("--obj", type=str, default="ext", help="the objective of the agent, either ext or e3b")
parser.add_argument("--ctx-len", type=int, default=4, help="agent's context length")
parser.add_argument("--total-steps", type=lambda x: int(float(x)), default=10000000, help="total timesteps of the experiments")
parser.add_argument("--n-envs", type=int, default=8, help="the number of parallel game environments")
parser.add_argument("--n-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--batch-size", type=int, default=256, help="the batch size for training")
parser.add_argument("--n-updates", type=int, default=16, help="gradient updates per collection")
parser.add_argument("--lr", type=float, default=2.5e-4, help="the learning rate of the optimizer")
# parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, help="Toggle learning rate annealing for policy and value networks")
parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, help="Toggles advantages normalization")
parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
parser.add_argument("--max-kl-div", type=float, default=None, help="the target KL divergence threshold")

parser.add_argument("--load-agent", type=str, default=None, help="file to load the agent from")
parser.add_argument("--save-agent", type=str, default=None, help="file to periodically save the agent to")

# parser.add_argument("--lr-tc", type=float, default=3e-4, help="learning rate for time contrastive encoder")


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

    # args.n_envs_per_id = args.n_envs // len(args.env_ids)

    args.collect_size = args.n_envs * args.n_steps
    # args.total_steps = args.total_steps // args.collect_size * args.collect_size
    args.n_collects = args.total_steps // args.collect_size

    args.last_token_only = True  # if cnnagent, else False

    return args


# steps, updates, iterations
# timesteps in env, updates to policy, iterations of training


def main(args):
    print("Running PPO with args: ", args)
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # encoder = Encoder((1, 84, 84), 64).to(args.device)
    # e3b_encode_fn = lambda obs: encoder.encode(obs[:, [-1]])  # encode only the latest frame
    # env = make_env(args.env_id, n_envs=args.n_envs, frame_stack=args.frame_stack, obj=args.obj, e3b_encode_fn=e3b_encode_fn, gamma=args.gamma, device=args.device, seed=args.seed)

    mbuffer, mbuffer_test = MultiBuffer(), MultiBuffer()
    for env_id in [args.env_id]:
        env = make_env(env_id, n_envs=args.n_envs_per_id, obj=args.obj, e3b_encode_fn=None, gamma=args.gamma, device=args.device, seed=args.seed, buf_size=args.n_steps)
        mbuffer.buffers.append(Buffer(args.n_steps, args.n_envs_per_id, env, device=args.device))
    # for env_id in args.env_ids_test:
    # env = make_env(env_id, n_envs=args.n_envs_per_id, obj=args.obj, e3b_encode_fn=None, gamma=args.gamma, device=args.device, seed=args.seed, buf_size=args.n_steps)
    # mbuffer_test.buffers.append(Buffer(args.n_steps, args.n_envs_per_id, env, device=args.device))

    agent = CNNAgent((args.ctx_len, 1, 84, 84), 18).to(args.device)
    torchinfo.summary(agent, input_size=[(args.batch_size, args.ctx_len, 1, 84, 84)], dtypes=[torch.float32], device=args.device)
    if args.load_agent is not None:
        agent.load_state_dict(torch.load(args.load_agent))
    # torchinfo.summary(agent, input_size=(args.batch_size,) + env.single_observation_space.shape, device=args.device)
    # opt = optim.Adam([{"params": agent.parameters(), "lr": args.lr, "eps": 1e-5}, {"params": encoder.parameters(), "lr": args.lr_tc, "eps": 1e-8}])
    opt = optim.Adam([{"params": agent.parameters(), "lr": args.lr, "eps": 1e-5}])

    start_time = time.time()
    dtime_env = 0.0
    dtime_inference = 0.0
    dtime_learning = 0.0

    viz_slow = set(np.linspace(0, args.n_collects - 1, 10).astype(int))
    viz_midd = set(np.linspace(0, args.n_collects - 1, 100).astype(int)).union(viz_slow)
    viz_fast = set(np.linspace(0, args.n_collects - 1, 1000).astype(int)).union(viz_midd)

    pbar = tqdm(range(args.n_collects))
    for i_collect in pbar:
        mbuffer.collect(agent, args.ctx_len)
        mbuffer.calc_gae(args.gamma, args.gae_lambda)
        # mbuffer_test.collect(agent, args.ctx_len)

        # if args.anneal_lr:  # Annealing the rate if instructed to do so.
        #     frac = 1.0 - i_iter / args.n_updates
        #     lrnow = frac * args.lr
        #     opt.param_groups[0]["lr"] = lrnow

        agent.train()
        for i_update in range(args.n_updates):
            batch = mbuffer.generate_batch(args.batch_size, args.ctx_len)
            obs, act, done, ret, adv = batch["obs"], batch["act"], batch["done"], batch["ret"], batch["adv"]
            dist_old, val_old = batch["dist"], batch["val"]
            dist, val = agent.act(obs=obs, act=act, done=done)

            if args.last_token_only:
                dist = torch.distributions.Categorical(logits=dist.logits[:, -1])
                dist_old = torch.distributions.Categorical(logits=dist_old.logits[:, -1])
                val, val_old = val[:, -1], val_old[:, -1]
                obs, act, done, ret, adv = obs[:, -1], act[:, -1], done[:, -1], ret[:, -1], adv[:, -1]

            if args.norm_adv:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            ratio = (dist.log_prob(act) - dist_old.log_prob(act)).exp()
            loss_pg1 = -adv * ratio
            loss_pg2 = -adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
            loss_pg = torch.max(loss_pg1, loss_pg2).mean()

            if args.clip_vloss:
                loss_v_unclipped = (val - ret) ** 2
                v_clipped = val_old + (val - val_old).clamp(-args.clip_coef, args.clip_coef)
                loss_v_clipped = (v_clipped - ret) ** 2
                loss_v_max = torch.max(loss_v_unclipped, loss_v_clipped)
                loss_v = 0.5 * loss_v_max.mean()
            else:
                loss_v = 0.5 * ((val - ret) ** 2).mean()

            entropy = dist.entropy()
            loss_entropy = entropy.mean()

            loss = loss_pg + loss_v * args.vf_coef - args.ent_coef * loss_entropy

            # obs_anc, obs_pos, obs_neg = sample_contrastive_batch(obs[:, :, -1, :, :], p=0.1, batch_size=args.batch_size)
            # loss_tc = calc_contrastive_loss(encoder, obs_anc, obs_pos, obs_neg)
            # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + loss_tc

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            opt.step()

            kl_div = torch.distributions.kl_divergence(dist_old, dist).mean().item()
            if args.max_kl_div is not None and kl_div > args.max_kl_div:
                break

            y_pred, y_true = val_old.flatten().cpu().numpy(), ret.flatten().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        data = {}
        if i_collect in viz_fast:  # fast logging, ex: scalars
            data["details/lr"] = opt.param_groups[0]["lr"]
            data["details/value_loss"] = loss_v.item()
            data["details/policy_loss"] = loss_pg.item()
            data["details/entropy"] = loss_entropy.item()
            # data["details/kl_div"] = kl_div
            # data["details/clipfrac"] = np.mean(clipfracs)
            data["details/explained_variance"] = explained_var
            data["meta/SPS"] = i_collect * args.collect_size / (time.time() - start_time)
            data["meta/global_step"] = i_collect * args.collect_size
            # data["details/loss_tc"] = loss_tc.item()
            for key, val in env.key2past_rets.items():
                rets = torch.cat(val).tolist()
                if len(rets) > 0:
                    data[f"charts/{key}"] = np.mean(rets)
                    data[f"charts_hist/{key}"] = wandb.Histogram(rets)  # TODO move to viz_midd
        if i_collect in viz_midd:  # midd loggin, ex: histograms
            data["details_hist/entropy"] = wandb.Histogram(entropy.detach().cpu().numpy())
            # data["details_hist/action"] = wandb.Histogram(b_actions.detach().cpu().numpy())
        if i_collect in viz_slow:  # slow logging, ex: videos
            vid = np.stack(env.past_obs).copy()
            vid[:, :, -1, :] = 0
            vid[:, :, :, -1] = 0
            vid = rearrange(vid, "t (H W) h w -> t (H h) (W w)", H=2, W=4)
            data["media/vid"] = wandb.Video(rearrange(vid, "t h w -> t 1 h w"), fps=15)
            if args.save_agent is not None:  # save agent
                print("Saving agent...")
                os.makedirs(os.path.dirname(args.save_agent), exist_ok=True)
                torch.save(agent.state_dict(), f"{args.save_agent}")
        if args.track and i_collect in viz_fast:  # tracking
            wandb.log(data, step=i_collect * args.collect_size)
            plt.close("all")

        keys_tqdm = ["charts/ret_ext", "meta/SPS"]
        pbar.set_postfix({k.split("/")[-1]: data[k] for k in keys_tqdm if k in data})

    env.close()


if __name__ == "__main__":
    main(parse_args())
