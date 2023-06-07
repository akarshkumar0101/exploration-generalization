# Adapted from CleanRL's ppo_atari_envpool.py
import argparse
import os
import random
import time
from distutils.util import strtobool

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchinfo
from agent_atari import CNNAgent, RandomAgent
from buffers import Buffer, MultiBuffer
from decision_transformer import DecisionTransformer
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
parser.add_argument("--lr", type=float, default=6e-4, help="the learning rate of the optimizer")
parser.add_argument("--lr-schedule", type=lambda x: bool(strtobool(x)), default=True, help="use lr schedule warmup and cosine decay")
parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, help="Toggles advantages normalization")
parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
parser.add_argument("--max-kl-div", type=float, default=None, help="the target KL divergence threshold")

parser.add_argument("--arch", type=str, default="cnn", help="either cnn or gpt")
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

    args.n_envs_per_id = args.n_envs // 1  # len(args.env_ids)

    args.collect_size = args.n_envs * args.n_steps
    # args.total_steps = args.total_steps // args.collect_size * args.collect_size
    args.n_collects = args.total_steps // args.collect_size

    args.last_token_only = True if args.arch == "cnn" else False
    return args


# steps, updates, iterations
# timesteps in env, updates to policy, iterations of training


def get_lr(lr, i_collect, n_collects, lr_schedule=True):
    assert i_collect <= n_collects
    if not lr_schedule:
        return lr
    lr_min = lr / 10.0

    n_warmup = n_collects // 100
    if i_collect < n_warmup:
        return lr * i_collect / n_warmup
    decay_ratio = (i_collect - n_warmup) / (n_collects - n_warmup)
    coeff = 0.5 * (1.0 + np.math.cos(np.pi * decay_ratio))  # coeff ranges 0..1
    assert 0 <= decay_ratio <= 1 and 0 <= coeff <= 1
    return lr_min + coeff * (lr - lr_min)


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

    if args.arch == "cnn":
        agent = CNNAgent((args.ctx_len, 1, 84, 84), 18).to(args.device)
    elif args.arch == "gpt":
        agent = DecisionTransformer(args.ctx_len, 18, 4, 4, 4 * 64, 0.0, True).to(args.device)
    elif args.arch == "rand":
        agent = RandomAgent(18).to(args.device)
    print("Agent Summary: ")
    torchinfo.summary(
        agent,
        input_size=[(args.batch_size, args.ctx_len, 1, 84, 84), (args.batch_size, args.ctx_len), (args.batch_size, args.ctx_len)],
        dtypes=[torch.float, torch.long, torch.bool],
        device=args.device,
    )
    if args.load_agent is not None:
        agent.load_state_dict(torch.load(args.load_agent))

    if args.arch == "cnn" or args.arch == "rand":
        # opt = optim.Adam([{"params": agent.parameters(), "lr": args.lr, "eps": 1e-5}, {"params": encoder.parameters(), "lr": args.lr_tc, "eps": 1e-8}])
        opt = torch.optim.Adam([{"params": agent.parameters(), "lr": args.lr, "eps": 1e-5}])
    elif args.arch == "gpt":
        opt = agent.create_optimizer(0.0, args.lr, (0.9, 0.95), args.device)

    start_time = time.time()

    viz_slow = set(np.linspace(1, args.n_collects - 1, 10).astype(int))
    viz_midd = set(np.linspace(1, args.n_collects - 1, 100).astype(int)).union(viz_slow)
    viz_fast = set(np.linspace(1, args.n_collects - 1, 1000).astype(int)).union(viz_midd)

    pbar = tqdm(range(args.n_collects))
    for i_collect in pbar:
        dt_collect, dt_gae, dt_gen_batch, dt_forward, dt_loss, dt_backward = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        time0 = time.time()
        mbuffer.collect(agent, args.ctx_len)
        time1 = time.time()
        mbuffer.calc_gae(args.gamma, args.gae_lambda)
        time2 = time.time()
        # mbuffer_test.collect(agent, args.ctx_len)

        lr = get_lr(args.lr, i_collect, args.n_collects, args.lr_schedule)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        agent.train()

        for _ in range(args.n_updates):
            time3 = time.time()
            batch = mbuffer.generate_batch(args.batch_size, args.ctx_len)
            time4 = time.time()
            obs, act, done, ret, adv = batch["obs"], batch["act"], batch["done"], batch["ret"], batch["adv"]
            dist_old, val_old = batch["dist"], batch["val"]
            dist, val = agent(obs=obs, act=act, done=done)
            time5 = time.time()

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
            time6 = time.time()

            # obs_anc, obs_pos, obs_neg = sample_contrastive_batch(obs[:, :, -1, :, :], p=0.1, batch_size=args.batch_size)
            # loss_tc = calc_contrastive_loss(encoder, obs_anc, obs_pos, obs_neg)
            # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + loss_tc

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            opt.step()
            time7 = time.time()

            y_pred, y_true = val_old.flatten().cpu().numpy(), ret.flatten().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            kl_div = torch.distributions.kl_divergence(dist_old, dist).mean().item()
            if args.max_kl_div is not None and kl_div > args.max_kl_div:
                break

            dt_gen_batch += time4 - time3
            dt_forward += time5 - time4
            dt_loss += time6 - time5
            dt_backward += time7 - time6

        dt_collect += time1 - time0
        dt_gae += time2 - time1

        # ------------------- Logging ------------------- #
        data = {}
        for env_id, env in zip([args.env_id], mbuffer.envs):
            for key, val in env.key2past_rets.items():  # log returns
                rets = torch.cat(val).tolist()
                if len(rets) > 0 and i_collect in viz_fast:  # log scalar
                    data[f"charts/{env_id}_{key}"] = np.mean(rets)
                if len(rets) > 0 and i_collect in viz_midd:  # log histogram
                    data[f"charts_hist/{env_id}_{key}"] = wandb.Histogram(rets)
            if args.log_video and i_collect in viz_slow:  # log video
                vid = np.stack(env.past_obs).copy()[-450:, :4]  # t, b, c, h, w
                vid[:, :, :, -1, :] = 128
                vid[:, :, :, :, -1] = 128
                vid = rearrange(vid, "t (H W) 1 h w -> t 1 (H h) (W w)", H=2, W=2)
                print('creating vide of shape: ', vid.shape)
                data[f"media/{env_id}_vid"] = wandb.Video(vid, fps=15)

        # print(f"dt_const: {mbuffer.buffers[0].dt_const}, dt_inf: {mbuffer.buffers[0].dt_inf}, dt_env: {mbuffer.buffers[0].dt_env}, dt_learn: {dt_learn}")
        if i_collect in viz_fast:  # fast logging, ex: scalars
            data["meta/dt_const"] = mbuffer.buffers[0].dt_const
            data["meta/dt_inf"] = mbuffer.buffers[0].dt_inf
            data["meta/dt_env"] = mbuffer.buffers[0].dt_env
            data["meta/dt_collect"] = dt_collect
            data["meta/dt_gae"] = dt_gae
            data["meta/dt_gen_batch"] = dt_gen_batch
            data["meta/dt_forward"] = dt_forward
            data["meta/dt_loss"] = dt_loss
            data["meta/dt_backward"] = dt_backward

            data["details/lr"] = opt.param_groups[0]["lr"]
            data["details/loss_value"] = loss_v.item()
            data["details/loss_policy"] = loss_pg.item()
            data["details/entropy"] = loss_entropy.item()
            data["details/perplexity"] = np.e ** loss_entropy.item()
            data["details/kl_div"] = kl_div
            # data["details/clipfrac"] = np.mean(clipfracs)
            data["details/explained_variance"] = explained_var
            data["meta/SPS"] = i_collect * args.collect_size / (time.time() - start_time)
            data["meta/global_step"] = i_collect * args.collect_size
            # data["details/loss_tc"] = loss_tc.item()
        if i_collect in viz_midd:  # midd loggin, ex: histograms
            data["details_hist/entropy"] = wandb.Histogram(entropy.detach().cpu().numpy().flatten())
            data["details_hist/action"] = wandb.Histogram(act.detach().cpu().numpy().flatten())
        if i_collect in viz_slow:  # slow logging, ex: videos
            if args.save_agent is not None:  # save agent
                print("Saving agent...")
                os.makedirs(os.path.dirname(args.save_agent), exist_ok=True)
                torch.save(agent.state_dict(), args.save_agent)
        if args.track and i_collect in viz_fast:  # tracking
            wandb.log(data, step=i_collect * args.collect_size)
            # plt.close("all")

        keys_tqdm = ["charts/ret_ext", "meta/SPS"]
        pbar.set_postfix({k.split("/")[-1]: data[k] for k in keys_tqdm if k in data})


if __name__ == "__main__":
    main(parse_args())
