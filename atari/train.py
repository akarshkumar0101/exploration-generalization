# Adapted from CleanRL's ppo_atari_envpool.py
import argparse
import os
import random
import time
from distutils.util import strtobool

import agent_atari
import atari_data
import buffers
import numpy as np
import timers
import torch
import torchinfo
import utils
from einops import rearrange
from env_atari import make_env
from torch.distributions import Categorical
from tqdm.auto import tqdm

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--project", type=str, default=None, help="the wandb's project name")
parser.add_argument("--name", type=str, default=None, help="the name of this experiment")
parser.add_argument("--log-video", type=lambda x: bool(strtobool(x)), default=False)
# parser.add_argument("--viz-slow", type=lambda x: bool(strtobool(x)), default=False)
# parser.add_argument("--viz_midd", type=lambda x: bool(strtobool(x)), default=False)

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")

# Algorithm arguments
parser.add_argument("--env-ids", type=str, nargs="+", default=["Pong"], help="the id of the environment")
parser.add_argument("--obj", type=str, default="ext", help="the objective of the agent, either ext or e3b")
parser.add_argument("--ctx-len", type=int, default=4, help="agent's context length")
parser.add_argument("--total-steps", type=lambda x: int(float(x)), default=10000000, help="total timesteps of the experiments")
parser.add_argument("--n-envs", type=int, default=8, help="the number of parallel game environments")
parser.add_argument("--n-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--batch-size", type=int, default=256, help="the batch size for training")
parser.add_argument("--n-updates", type=int, default=16, help="gradient updates per collection")

parser.add_argument("--model", type=str, default="cnn", help="either cnn or gpt")
parser.add_argument("--load-agent", type=str, default=None, help="file to load the agent from")
parser.add_argument("--save-agent", type=str, default=None, help="file to periodically save the agent to")
parser.add_argument("--full-action-space", type=lambda x: bool(strtobool(x)), default=True)

parser.add_argument("--lr", type=float, default=2.5e-4, help="the learning rate of the optimizer")
parser.add_argument("--lr-warmup", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--lr-decay", type=str, default="none")

parser.add_argument("--episodic-life", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--norm-rew", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, help="Toggles advantages normalization")
parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")

parser.add_argument("--max-grad-norm", type=float, default=1.0, help="the maximum norm for the gradient clipping")
parser.add_argument("--max-kl-div", type=float, default=None, help="the target KL divergence threshold")


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
    args.n_collects = args.total_steps // args.collect_size
    return args


def main(args):
    print("Running PPO with args: ", args)
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    mbuffer = buffers.MultiBuffer()
    for env_id in args.env_ids:
        env = make_env(env_id, args.n_envs_per_id, args.obj, args.norm_rew, args.gamma, args.episodic_life, args.full_action_space, args.device, args.seed)
        mbuffer.buffers.append(buffers.Buffer(env, args.n_steps, device=args.device))

    agent = utils.create_agent(args.model, env.single_action_space.n, args.ctx_len, args.load_agent).to(args.device)
    print("Agent Summary: ")
    torchinfo.summary(
        agent,
        input_size=[(args.batch_size, args.ctx_len), (args.batch_size, args.ctx_len, 1, 84, 84), (args.batch_size, args.ctx_len), (args.batch_size, args.ctx_len)],
        dtypes=[torch.bool, torch.uint8, torch.long, torch.float],
        device=args.device,
    )
    opt = agent.create_optimizer(lr=args.lr, device=args.device)

    if args.obj == "eps":
        idm = agent_atari.IDM(env.single_action_space.n, n_dim=512, normalize=True).to(args.device)
        opt.add_param_group({"params": idm.parameters(), "lr": args.lr})
        env.configure_eps_reward(encode_fn=idm, ctx_len=16, k=4)
    if args.obj == "rnd":
        rnd_model = agent_atari.RNDModel().to(args.device)
        opt.add_param_group({"params": rnd_model.parameters(), "lr": args.lr})
        env.configure_rnd_reward(rnd_model=rnd_model)

    start_time = time.time()
    pbar = tqdm(range(args.n_collects))
    for i_collect in pbar:
        timer = timers.Timer()

        lr = utils.get_lr(args.lr, args.lr / 10.0, i_collect, args.n_collects, warmup=args.lr_warmup, decay=args.lr_decay)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        mbuffer.collect(agent, args.ctx_len, timer=timer)
        if args.ppo:
            mbuffer.calc_gae(args.gamma, args.gae_lambda)
        if args.teacher:
            mbuffer_teacher.collect(agent_teacher, args.ctx_len, timer=timer)

        agent.train()
        for _ in range(args.n_updates):
            if args.ppo:
                batch = mbuffer.generate_batch(args.batch_size, args.ctx_len)
            elif args.teacher:
                batch = mbuffer_teacher.generate_batch(args.batch_size, args.ctx_len)
            
            logits, val = agent(done=batch["done"], obs=batch["obs"], act=batch["act"], rew=batch["rew"])

            if agent.last_token_train:
                batch = {k: v[:, [-1]] for k, v in batch.items()}
                logits, val = logits[:, [-1]], val[:, [-1]]
            dist, batch_dist = Categorical(logits=logits), Categorical(logits=batch["logits"])

            with timer.add_time("calc_loss"):
                if args.ppo:
                    loss_p = utils.calc_ppo_policy_loss(dist, batch_dist, batch["act"], batch["adv"], norm_adv=args.norm_adv, clip_coef=args.clip_coef)
                    loss_v = utils.calc_ppo_value_loss(val, batch["val"], batch["ret"], clip_coef=args.clip_coef if args.clip_vloss else None)
                    loss = 1.0 * loss_p.mean() + args.vf_coef * loss_v.mean()

                    if args.obj == "eps":
                        pass
                    if args.obj == "rnd":
                        rnd_student, rnd_teacher = rnd_model(batch["obs"][:, 0], update_rms_obs=False)  # only give one frame
                        loss_rnd = utils.calc_rnd_loss(rnd_student, rnd_teacher)
                        loss = loss + 1.0 * loss_rnd.mean()
                elif args.teacher:
                    loss_klbc = torch.nn.functional.kl_div(dist.logits, batch_dist.logits, log_target=True, reduction="none").sum(dim=-1)
                    loss = 1.0 * loss_klbc.mean()
                loss_e = dist.entropy()
                loss = loss - args.ent_coef * loss_e.mean()

            with timer.add_time("opt_step"):
                opt.zero_grad()
            with timer.add_time("backward_pass"):
                loss.backward()
            with timer.add_time("opt_step"):
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()

            kl_div = torch.distributions.kl_divergence(batch_dist, dist)
            if args.max_kl_div is not None and kl_div.mean().item() > args.max_kl_div:
                break

        # ------------------- Logging ------------------- #
        data = {}
        viz_slow = i_collect % np.clip(args.n_collects // 10, 1, None) == 0
        viz_midd = i_collect % np.clip(args.n_collects // 100, 1, None) == 0 or viz_slow
        viz_fast = i_collect % np.clip(args.n_collects // 1000, 1, None) == 0 or viz_midd
        to_np = lambda x: x.detach().cpu().numpy()

        # Explained Variance
        y_pred = to_np(mbuffer.buffers[0].vals).flatten()
        y_true = to_np(mbuffer.buffers[0].rets).flatten()
        explained_var = np.nan if np.var(y_true) == 0 else 1.0 - np.var(y_true - y_pred) / np.var(y_true)

        hns = []
        for env_id, env in zip(args.env_ids, mbuffer.envs):
            hns_env = atari_data.calc_hns(env_id, env.get_past_returns()["ret_score"]).mean().item()
            if not np.isnan(hns_env):
                hns.append(hns_env)

        for env_id, env in zip(args.env_ids, mbuffer.envs):
            for key, rets in env.get_past_returns().items():  # log returns
                if len(rets) > 0 and viz_fast:  # log scalar
                    data[f"returns/{env_id}_{key}"] = rets.mean().item()
                # if len(rets) > 0 and viz_midd:  # log histogram
                #     data[f"returns_hist/{env_id}_{key}"] = wandb.Histogram(to_np(rets))
            if args.log_video and viz_slow:  # log video
                vid = np.stack(env.get_past_obs()).copy()[-450:, :4]  # t, b, c, h, w
                vid[:, :, :, -1, :] = 128
                vid[:, :, :, :, -1] = 128
                vid = rearrange(vid, "t (H W) 1 h w -> t 1 (H h) (W w)", H=2, W=2)
                print("creating video of shape: ", vid.shape)
                data[f"media/{env_id}_vid"] = wandb.Video(vid, fps=15)

        if viz_fast:  # fast logging, ex: scalars
            for key, tim in timer.key2time.items():
                data[f"time/{key}"] = tim
                # if viz_midd:
                # print(f"time/{key:30s}: {tim:.3f}")
            data["meta/SPS"] = (i_collect + 1) * args.collect_size / (time.time() - start_time)
            data["meta/global_step"] = (i_collect + 1) * args.collect_size

            if len(hns) > 0:
                data["charts/hns"] = np.mean(hns)
            data["charts/perplexity"] = np.e ** loss_e.mean().item()
            data["details/lr"] = opt.param_groups[0]["lr"]
            data["details/loss_value"] = loss_v.mean().item()
            data["details/loss_policy"] = loss_p.mean().item()
            data["details/grad_norm"] = grad_norm.item()
            data["details/entropy"] = loss_e.mean().item()
            data["details/kl_div"] = kl_div.mean().item()
            # data["details/clipfrac"] = np.mean(clipfracs)
            data["details/explained_variance"] = explained_var
        if viz_midd:  # midd loggin, ex: histograms
            # data["details_hist/entropy"] = wandb.Histogram(to_np(loss_e).flatten())
            data["details_hist/perplexity"] = wandb.Histogram(np.e ** to_np(loss_e).flatten())
            data["details_hist/action"] = wandb.Histogram(to_np(batch["act"]).flatten())
        if viz_slow:  # slow logging, ex: videos
            if args.save_agent is not None:  # save agent
                print("Saving agent...")
                os.makedirs(os.path.dirname(args.save_agent), exist_ok=True)
                torch.save(agent.state_dict(), args.save_agent)

        if args.track and viz_fast:  # tracking
            wandb.log(data, step=i_collect * args.collect_size)
        keys_tqdm = ["charts/hns", "meta/global_step", "meta/SPS"]
        pbar.set_postfix({k.split("/")[-1]: data[k] for k in keys_tqdm if k in data})


if __name__ == "__main__":
    main(parse_args())
