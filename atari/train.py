# Adapted from CleanRL's ppo_atari_envpool.py
import argparse
import os
import random
import time
from distutils.util import strtobool

import agent_atari
import atari_data
import buffers
import normalize

# import eval_diversity
import numpy as np
import timers
import torch
import torchinfo
import utils
import wandb
from einops import rearrange
from env_atari import make_concat_env
from torch.distributions import Categorical
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--entity", type=lambda x: None if x.lower() == "none" else x, default=None)
parser.add_argument("--project", type=lambda x: None if x.lower() == "none" else x, default=None)
parser.add_argument("--name", type=lambda x: None if x.lower() == "none" else x, default=None)
parser.add_argument("--log_video", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--log_hist", type=lambda x: bool(strtobool(x)), default=False)

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")

# Algorithm arguments
parser.add_argument("--env_ids", type=str, nargs="+", default=["Pong"], help="the id of the environment")
parser.add_argument("--obj", type=str, default="ext", help="the objective of the agent, either ext or e3b")
parser.add_argument("--total_steps", type=lambda x: int(float(x)), default=int(100e6), help="total timesteps of the experiments")
parser.add_argument("--n_envs", type=int, default=128, help="the number of parallel game environments")
parser.add_argument("--n_steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--batch_size", type=int, default=4096, help="the batch size for training")
parser.add_argument("--n_updates", type=int, default=16, help="gradient updates per collection")

parser.add_argument("--model", type=str, default="cnn", help="either cnn or gpt")
parser.add_argument("--ctx_len", type=int, default=4, help="agent's context length")
parser.add_argument("--load_agent_history", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--load_agent", type=lambda x: None if x.lower() == "none" else x, default=None, help="file to load the agent from")
parser.add_argument("--save_agent", type=lambda x: None if x.lower() == "none" else x, default=None, help="file to periodically save the agent to")
parser.add_argument("--full_action_space", type=lambda x: bool(strtobool(x)), default=True)

parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--lr_warmup", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--lr_decay", type=str, default="none")
parser.add_argument("--max_grad_norm", type=float, default=1.0)

parser.add_argument("--episodic_life", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--norm_rew", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--ent_coef", type=float, default=0.001)
parser.add_argument("--clip_coef", type=float, default=0.1)
parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
parser.add_argument("--vf_coef", type=float, default=0.5, help="coefficient of the value function")
parser.add_argument("--max_kl_div", type=lambda x: None if x.lower == "none" else float(x), default=None, help="the target KL divergence threshold")

parser.add_argument("--pre_obj", type=str, default="ext")
parser.add_argument("--train_klbc", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--model_teacher", type=str, default="cnn", help="either cnn or gpt")
parser.add_argument("--ctx_len_teacher", type=int, default=4, help="agent's context length")
parser.add_argument("--load_agent_teacher", type=lambda x: None if x.lower() == "none" else x, default=None, help="file to load the agent from")
parser.add_argument("--teacher_last_k", type=int, default=1)
parser.add_argument("--pre_teacher_last_k", type=int, default=1)

parser.add_argument("--n_steps_rnd_init", type=lambda x: int(float(x)), default=int(100e3))


def parse_args(*args, **kwargs):
    args, uargs = parser.parse_known_args(*args, **kwargs)
    for key in ["project", "name", "load_agent", "save_agent", "load_agent_teacher"]:
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).format(**vars(args)))
    args.n_envs_per_id = args.n_envs // len(args.env_ids)
    args.collect_size = args.n_envs * args.n_steps
    args.n_collects = args.total_steps // args.collect_size
    for k, v in dict([tuple(uarg.replace("--", "").split("=")) for uarg in uargs]).items():
        setattr(args, k, v)
    return args


def load_teacher(args, env):
    agent_teachers = []
    for env_id in args.env_ids:
        load_agent_dir = args.load_agent_teacher.format(env_id=env_id)
        files = [f"{load_agent_dir}/{file}" for file in np.sort(os.listdir(load_agent_dir))]
        files = files[-args.teacher_last_k :]
        for load_agent in files:
            print(f"Loading teacher agent from {load_agent}")
            agent = utils.create_agent(args.model_teacher, env.single_action_space.n, args.ctx_len_teacher, load_agent, device=args.device)
            agent_teachers.append(agent)
    return agent_atari.ConcatAgent(agent_teachers)


def main(args):
    print("Running PPO/KLBC with args: ", args)
    a = torch.randn(100).to(args.device)
    print(f"Testing device: {args.device}, {a.mean().item()}")
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    env = make_concat_env(args.env_ids, args.n_envs_per_id, args.obj, args.norm_rew, args.gamma, args.episodic_life, args.full_action_space, args.device, args.seed)
    buffer = buffers.Buffer(env, args.n_steps, device=args.device)
    if args.train_klbc:
        env_teacher = make_concat_env(args.env_ids, args.n_envs_per_id, args.obj, args.norm_rew, args.gamma, args.episodic_life, args.full_action_space, args.device, args.seed)
        buffer_teacher = buffers.Buffer(env_teacher, args.n_steps, device=args.device)
        agent_teacher = load_teacher(args, env_teacher)

    agent = utils.create_agent(args.model, env.single_action_space.n, args.ctx_len, args.load_agent, device=args.device)
    print("Agent Summary: ")
    torchinfo.summary(
        agent,
        input_size=[(args.batch_size, args.ctx_len), (args.batch_size, args.ctx_len, 1, 84, 84), (args.batch_size, args.ctx_len), (args.batch_size, args.ctx_len)],
        dtypes=[torch.bool, torch.uint8, torch.long, torch.float],
        device=args.device,
    )
    opt = agent.create_optimizer(lr=args.lr, device=args.device)

    if args.obj == "eps":
        idm = agent_atari.IDM(env.single_action_space.n, n_dim=64, normalize=True).to(args.device)
        opt.add_param_group({"params": idm.parameters(), "lr": args.lr})
        env.configure_eps_reward(encode_fn=idm, ctx_len=1024, k=1, p=2, obj="eps")
    if args.obj == "sps":
        env.configure_eps_reward(encode_fn=env.state_encoder, ctx_len=1024, k=1, p=0, obj="sps")
    if args.obj == "rnd":
        rnd_model = agent_atari.RNDModel().to(args.device)
        opt.add_param_group({"params": rnd_model.parameters(), "lr": args.lr})
        for envi in env.envs:
            envi.configure_rnd_reward(rnd_model=rnd_model)

    if args.n_steps_rnd_init > 0:
        print("Initializing RND model with random agent...")
        for _ in tqdm(range(args.n_steps_rnd_init // args.collect_size)):
            buffer.collect(agent_atari.RandomAgent(env.single_action_space.n), args.ctx_len)

    # rms_hist = normalize.RunningMeanStd()  # variance over entire history
    start_time = time.time()

    print("Starting Learning")
    pbar = tqdm(range(args.n_collects))
    for i_collect in pbar:
        timer = timers.Timer()
        with timer.add_time("collect"):
            if args.train_klbc:
                buffer_teacher.collect(agent_teacher, args.ctx_len_teacher, timer=timer)
            buffer.collect(agent, args.ctx_len, timer=timer)
        with timer.add_time("calc_gae"):
            buffer.calc_gae(args.gamma, args.gae_lambda)

        lr = utils.get_lr(args.lr, args.lr / 10.0, i_collect, args.n_collects, warmup=args.lr_warmup, decay=args.lr_decay)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        agent.train()
        for _ in range(args.n_updates):
            with timer.add_time("generate_batch"):
                if args.train_klbc:
                    batch = buffer_teacher.generate_batch(args.batch_size, args.ctx_len)
                else:
                    batch = buffer.generate_batch(args.batch_size, args.ctx_len)
            with timer.add_time("forward_pass"):
                logits, val = agent(done=batch["done"], obs=batch["obs"], act=batch["act"], rew=batch["rew"])
                dist, batch_dist = Categorical(logits=logits), Categorical(logits=batch["logits"])

            with timer.add_time("calc_loss"):
                if args.train_klbc:
                    loss_klbc = utils.calc_klbc_loss(dist, batch_dist)
                else:
                    loss_p = utils.calc_ppo_policy_loss(dist, batch_dist, batch["act"], batch["adv"], norm_adv=args.norm_adv, clip_coef=args.clip_coef)
                    loss_v = utils.calc_ppo_value_loss(val, batch["val"], batch["ret"], clip_coef=args.clip_coef if args.clip_vloss else None)
                loss_e = dist.entropy()
                if agent.last_token_train:
                    if args.train_klbc:
                        loss_klbc = loss_klbc[:, [-1]]
                    else:
                        loss_p, loss_v = loss_p[:, [-1]], loss_v[:, [-1]]
                    loss_e = loss_e[:, [-1]]
                if args.train_klbc:
                    loss = loss_klbc.mean()
                else:
                    loss = loss_p.mean() + args.vf_coef * loss_v.mean()
                loss = loss - args.ent_coef * loss_e.mean()

                if args.obj == "eps":
                    logits_idm = idm.predict_action(batch["obs"][:, 0], batch["obs"][:, 1])
                    entropy_idm = Categorical(logits=logits_idm).entropy()
                    loss_idm = utils.calc_idm_loss(logits_idm, batch["act"][:, 0])
                    loss = loss + 1.0 * loss_idm.mean()

                if args.obj == "rnd":
                    rnd_student, rnd_teacher = rnd_model(batch["obs"][:, 0], update_rms_obs=False)  # only give one frame
                    loss_rnd = utils.calc_rnd_loss(rnd_student, rnd_teacher)
                    loss = loss + 1.0 * loss_rnd.mean()

            with timer.add_time("opt_step"):
                opt.zero_grad()
            with timer.add_time("backward_pass"):
                loss.backward()
            with timer.add_time("opt_step"):
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                if args.obj == "eps":
                    grad_norm_idm = torch.nn.utils.clip_grad_norm_(idm.parameters(), args.max_grad_norm)
                # for pg in opt.param_groups:
                # grad_norm = torch.nn.utils.clip_grad_norm_(pg["params"], args.max_grad_norm)
                opt.step()

            kl_div = torch.distributions.kl_divergence(batch_dist, dist)
            if args.max_kl_div is not None and kl_div.mean().item() > args.max_kl_div:
                break

        # ------------------- Logging ------------------- #
        # buffer = mbuffer.buffers[0]
        # rms_traj = normalize.RunningMeanStd()  # variance over current trajectory
        # rms_coll = normalize.RunningMeanStd()  # variance over current collection
        # rms_traj.update(rearrange(buffer.obss, "n t c h w -> t n c h w"))
        # rms_coll.update(rearrange(buffer.obss, "n t c h w -> (n t) c h w"))
        # rms_hist.update(rearrange(buffer.obss, "n t c h w -> (n t) c h w"))

        data = {}
        viz_slow = i_collect % np.clip(args.n_collects // 10, 1, None) == 0
        viz_midd = i_collect % np.clip(args.n_collects // 100, 1, None) == 0 or viz_slow
        viz_fast = i_collect % np.clip(args.n_collects // 1000, 1, None) == 0 or viz_midd
        to_np = lambda x: x.detach().cpu().numpy()

        # Explained Variance
        # y_pred = to_np(buffer.buffers[0].vals).flatten()
        # y_true = to_np(buffer.buffers[0].rets).flatten()
        # explained_var = np.nan if np.var(y_true) == 0 else 1.0 - np.var(y_true - y_pred) / np.var(y_true)

        hns = [atari_data.calc_hns(env_id, np.nanmean(envi.get_past_returns()["ret_score"])) for env_id, envi in zip(args.env_ids, env.envs)]
        if args.train_klbc:
            hns_teacher = [atari_data.calc_hns(env_id, np.nanmean(envi.get_past_returns()["ret_score"])) for env_id, envi in zip(args.env_ids, env_teacher.envs)]
        for env_id, envi in zip(args.env_ids, env.envs):
            pret_data = envi.get_past_returns()
            for key, rets in pret_data.items():  # log returns
                if viz_fast:  # log scalar
                    data[f"returns_max/{env_id}_{key}"] = np.nanmax(rets)
                    data[f"returns/{env_id}_{key}"] = np.nanmean(rets)
                    data[f"returns_perstep/{env_id}_{key}"] = np.nanmean(rets / pret_data["ret_traj"])
            if args.log_video and viz_slow:  # log video
                vid = np.stack(envi.get_past_obs()).copy()[-450:, :4]  # t, b, c, h, w
                vid[:, :, :, -1, :] = 128
                vid[:, :, :, :, -1] = 128
                vid = rearrange(vid, "t (H W) 1 h w -> t 1 (H h) (W w)", H=2, W=2)
                # print("creating video of shape: ", vid.shape)
                data[f"media/{env_id}_vid"] = wandb.Video(vid, fps=15)

        if viz_fast:  # fast logging, ex: scalars
            # data["diversity/traj_pix"] = rms_traj.var.mean().item()
            # data["diversity/coll_pix"] = rms_coll.var.mean().item()
            # data["diversity/hist_pix"] = rms_hist.var.mean().item()

            for key, tim in timer.key2time.items():
                data[f"time/{key}"] = tim
                # if viz_midd:
                # print(f"time/{key:30s}: {tim:.3f}")
            data["meta/SPS"] = (i_collect + 1) * args.collect_size / (time.time() - start_time)
            data["meta/global_step"] = (i_collect + 1) * args.collect_size

            data["charts/hns"] = np.nanmean(hns)
            if args.train_klbc:
                data["charts/hns_teacher"] = np.nanmean(hns_teacher)
            data["charts/perplexity"] = np.e ** loss_e.mean().item()
            data["details/lr"] = opt.param_groups[0]["lr"]
            if args.train_klbc:
                data["details/loss_klbc"] = loss_klbc.mean().item()
            else:
                data["details/loss_value"] = loss_v.mean().item()
                data["details/loss_policy"] = loss_p.mean().item()
            if args.obj == "eps":
                data["details/loss_idm"] = loss_idm.mean().item()
                data["details/grad_norm_idm"] = grad_norm_idm.item()
                data["charts/perplexity_idm"] = np.e ** entropy_idm.mean().item()
            # data["details/rms_returns_mean"] = env.rms_returns.mean.item()
            # data["details/rms_returns_var"] = env.rms_returns.var.item()
            data["details/grad_norm"] = grad_norm.item()
            data["details/entropy"] = loss_e.mean().item()
            data["details/kl_div"] = kl_div.mean().item()
            # data["details/clipfrac"] = np.mean(clipfracs)
            # data["details/explained_variance"] = explained_var

            # for env_id, buffer in zip(args.env_ids, mbuffer.buffers):
            #     lpips_diversity = eval_diversity.calc_diversity(buffer, n_iters=1, batch_size=512, device=args.device)
            #     data[f"details/{env_id}_lpips_diversity"] = lpips_diversity.mean().item()
        if viz_midd:  # midd loggin, ex: histograms
            # data["details_hist/entropy"] = wandb.Histogram(to_np(loss_e).flatten())
            data["details_hist/perplexity"] = wandb.Histogram(np.e ** to_np(loss_e).flatten())
            data["details_hist/action"] = wandb.Histogram(to_np(batch["act"]).flatten())
        if viz_slow:  # slow logging, ex: videos
            if args.save_agent is not None:  # save agent
                save_agent = f"{args.save_agent}/agent_{i_collect:09d}.pt"
                print(f"Saving agent to {save_agent}...")
                os.makedirs(os.path.dirname(save_agent), exist_ok=True)
                torch.save(agent.state_dict(), save_agent)

        if args.track and viz_fast:  # tracking
            wandb.log(data, step=i_collect * args.collect_size)
        keys_tqdm = ["charts/hns", "meta/global_step", "meta/SPS"]
        pbar.set_postfix({k.split("/")[-1]: data[k] for k in keys_tqdm if k in data})


if __name__ == "__main__":
    main(parse_args())

"""
Move the Intrinsic reward calculation and the reward normalization away from the env wrappers
and in the main loop calculation that overrides buffer.
Why? It is much more efficient (can calculate entire buffer's intrinsic reward in one go)
And more importantly: the reward normalization will apply the same to ALL rewards in this batch.

Three phases:
- specialist
- generalist
- finetune-bc OR finetune-ppo
"""
