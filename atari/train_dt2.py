import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torchinfo
from agent_atari import CNNAgent
from buffers2 import Buffer, MultiBuffer
from decision_transformer import DecisionTransformer
from einops import rearrange
from env_atari import make_env
from tqdm.auto import tqdm
from timers import Timer

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
parser.add_argument("--lr-schedule", type=lambda x: bool(strtobool(x)), default=True, help="use lr schedule warmup and cosine decay")
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

parser.add_argument("--arch", type=str, default="cnn", help="either cnn or gpt")
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
        args.expert_agent = args.expert_agent.format(**dict(env_id="{env_id}", **vars(args)))
    if args.load_agent is not None:
        args.load_agent = args.load_agent.format(**vars(args))
    if args.save_agent is not None:
        args.save_agent = args.save_agent.format(**vars(args))

    args.n_envs_per_id = args.n_envs // len(args.env_ids)

    args.collect_size = args.n_envs * args.n_steps
    # args.total_steps = args.total_steps // args.collect_size * args.collect_size
    args.n_collects = args.total_steps // args.collect_size

    args.last_token_only = True if args.arch == "cnn" else False
    return args


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
    print("Running distillation with args: ", args)
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.arch == "cnn":
        agent = CNNAgent((args.ctx_len, 1, 84, 84), 18).to(args.device)
    elif args.arch == "gpt":
        agent = DecisionTransformer(args.ctx_len, 18, 4, 4, 4 * 64, 0.0, True).to(args.device)
    print("Agent Summary: ")
    torchinfo.summary(
        agent,
        input_size=[(args.batch_size, args.ctx_len), (args.batch_size, args.ctx_len, 1, 84, 84), (args.batch_size, args.ctx_len), (args.batch_size, args.ctx_len)],
        dtypes=[torch.bool, torch.float, torch.long, torch.float],
        device=args.device,
    )
    if args.load_agent is not None:
        agent.load_state_dict(torch.load(args.load_agent))

    if args.arch == "cnn":
        # opt = optim.Adam([{"params": agent.parameters(), "lr": args.lr, "eps": 1e-5}, {"params": encoder.parameters(), "lr": args.lr_tc, "eps": 1e-8}])
        opt = torch.optim.Adam([{"params": agent.parameters(), "lr": args.lr, "eps": 1e-5}])
    elif args.arch == "gpt":
        opt = agent.create_optimizer(0.0, args.lr, (0.9, 0.95), args.device)

    experts = []
    mbuffer, mbuffer_test = MultiBuffer(), MultiBuffer()
    for env_id in args.env_ids:
        env = make_env(env_id, n_envs=args.n_envs_per_id, obj=args.obj, e3b_encode_fn=None, gamma=args.gamma, device=args.device, seed=args.seed, buf_size=args.n_steps)
        mbuffer.buffers.append(Buffer(args.n_steps, args.n_envs_per_id, env, device=args.device))
        print("Loading expert from: ", {args.expert_agent.format(env_id=env_id)})
        # expert = CNNAgent((args.ctx_len, 1, 84, 84), 18).to(args.device)
        # expert.load_state_dict(torch.load(args.expert_agent.format(env_id=env_id)))
        expert = torch.load(args.expert_agent.format(env_id=env_id)).to(args.device)
        experts.append(expert)
    for env_id in args.env_ids_test:
        env = make_env(env_id, n_envs=args.n_envs_per_id, obj=args.obj, e3b_encode_fn=None, gamma=args.gamma, device=args.device, seed=args.seed, buf_size=args.n_steps)
        mbuffer_test.buffers.append(Buffer(args.n_steps, args.n_envs_per_id, env, device=args.device))

    timer = Timer()
    pbar = tqdm(range(args.n_collects))
    for i_collect in pbar:
        timer.key2time.clear()
        print("Collecting expert data...")
        mbuffer.collect(experts, args.ctx_len, Timer())

        print("Collecting student data...")
        mbuffer_test.collect(agent, args.ctx_len, timer)

        lr = get_lr(args.lr, i_collect, args.n_collects, args.lr_schedule)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        agent.train()
        for _ in range(args.n_updates):
            batch = mbuffer.generate_batch(args.batch_size, args.ctx_len)
            obs, dist_expert, act, done = batch["obs"], batch["dist"], batch["act"], batch["done"]
            with timer.add_time("forward_pass"):
                logits, values = agent(obs=obs, act=act, done=done)  # b t ...
            dist_student = torch.distributions.Categorical(logits=logits)
            # TODO add option for last token training only

            with timer.add_time("calc_loss"):
                kl_div = torch.nn.functional.kl_div(dist_student.logits, dist_expert.logits, log_target=True, reduction="none")
                # kl_div = torch.distributions.kl.kl_divergence(dist_student, dist_teacher)

                entropy = dist_student.entropy()
                loss_kl, loss_entropy = kl_div.mean(), entropy.mean()
                loss = loss_kl - args.ent_coef * loss_entropy

            with timer.add_time("opt_step"):
                opt.zero_grad()
            with timer.add_time("backward_pass"):
                loss.backward()
            with timer.add_time("opt_step"):
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()

        lgts = dist_student.logits
        kld_oh_student = torch.nn.functional.kl_div((lgts).log_softmax(dim=-1), (lgts * 1e6).log_softmax(dim=-1), log_target=True, reduction="none").mean().item()
        lgts = dist_expert.logits
        kld_oh_expert = torch.nn.functional.kl_div((lgts).log_softmax(dim=-1), (lgts * 1e6).log_softmax(dim=-1), log_target=True, reduction="none").mean().item()

        # ------------------- Logging ------------------- #
        viz_slow = i_collect % (args.n_collects // 10) == 0
        viz_midd = i_collect % (args.n_collects // 100) == 0 or viz_slow
        viz_fast = i_collect % (args.n_collects // 1000) == 0 or viz_midd

        data = {}
        for env_id, env in zip(args.env_ids, mbuffer.envs):
            for key, val in env.key2past_rets.items():  # log returns
                if viz_fast:  # log scalar
                    data[f"charts_expert/{env_id}_{key}"] = torch.cat(val).mean().item()
                if viz_midd:  # log histogram
                    data[f"charts_expert_hist/{env_id}_{key}"] = wandb.Histogram(torch.cat(val).tolist())
        for env_id, env in zip(args.env_ids_test, mbuffer_test.envs):
            for key, val in env.key2past_rets.items():  # log returns
                if viz_fast:  # log scalar
                    data[f"charts_student/{env_id}_{key}"] = torch.cat(val).mean().item()
                if viz_midd:  # log histogram
                    data[f"charts_student_hist/{env_id}_{key}"] = wandb.Histogram(torch.cat(val).tolist())
            if args.log_video and viz_slow:  # log video
                vid = np.stack(env.past_obs).copy()[-450:, :4]  # t, b, c, h, w
                vid[:, :, :, -1, :] = 128
                vid[:, :, :, :, -1] = 128
                vid = rearrange(vid, "t (H W) 1 h w -> t 1 (H h) (W w)", H=2, W=2)
                data[f"media/{env_id}_vid"] = wandb.Video(vid, fps=15)

        if viz_fast:
            data["charts/loss_kl"] = loss_kl.item()
            data["details/loss_kl_0"] = kl_div[:, 0].mean().item()
            data["details/loss_kl_-1"] = kl_div[:, -1].mean().item()
            data["details/entropy"] = loss_entropy.item()
            data["details/perplexity"] = np.e ** loss_entropy.item()
            data["details/lr"] = lr
            data["details/kld_oh_student"] = kld_oh_student
            data["details/kld_oh_expert"] = kld_oh_expert
        if viz_midd:
            data["details_hist/entropy"] = wandb.Histogram(entropy.detach().cpu().numpy().flatten())
            data["details_hist/action"] = wandb.Histogram(act.detach().cpu().numpy().flatten())
        if viz_slow:
            if args.save_agent is not None:  # save agent
                print("Saving agent...")
                os.makedirs(os.path.dirname(args.save_agent), exist_ok=True)
                torch.save(agent.state_dict(), args.save_agent)

        if args.track and viz_fast:
            wandb.log(data, step=i_collect * args.collect_size)

        keys_tqdm = ["loss_kl", "loss_entropy", "Breakout_student_ret_ext", "Breakout_student_dtgpt_ext"]
        pbar.set_postfix({k.split("/")[-1]: data[k] for k in keys_tqdm if k in data})


if __name__ == "__main__":
    main(parse_args())
