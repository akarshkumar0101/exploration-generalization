# Adapted from CleanRL's ppo_atari_envpool.py
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
from einops import rearrange
from env_atari import make_env
from time_contrastive import calc_contrastive_loss, sample_contrastive_batch
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

parser.add_argument("--ctx-len", type=int, default=4, help="agent's context length")

parser.add_argument("--obj", type=str, default="ext", help="the objective of the agent, either ext or e3b")
parser.add_argument("--lr-tc", type=float, default=3e-4, help="learning rate for time contrastive encoder")

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
    args.n_updates = args.total_steps // args.collect_size
    return args


class Buffer:
    def __init__(self, args, env, agent):
        self.args, self.env, self.agent = args, env, agent

        self.obss = torch.zeros((args.n_steps, args.n_envs) + env.single_observation_space.shape, dtype=torch.uint8, device=args.device)
        self.dones = torch.zeros((args.n_steps, args.n_envs), dtype=torch.bool, device=args.device)

        self.acts = torch.zeros((args.n_steps, args.n_envs) + env.single_action_space.shape, dtype=torch.long, device=args.device)
        self.dists = None
        self.logprobs = None
        self.logits = torch.zeros((args.n_steps, args.n_envs, env.single_action_space.n), device=args.device)
        self.vals = torch.zeros((args.n_steps, args.n_envs), device=args.device)
        self.rews = torch.zeros((args.n_steps, args.n_envs), device=args.device)

        self.advs = torch.zeros((args.n_steps, args.n_envs), device=args.device)
        self.rets = torch.zeros((args.n_steps, args.n_envs), device=args.device)

        _, info = env.reset()
        self.obs = info["obs"]
        self.done = torch.zeros(args.n_envs, dtype=torch.bool, device=args.device)

    def construct_obs_idx(self, i_step):
        # i_step = i_step % self.args.n_steps
        if i_step >= self.args.ctx_len - 1:
            return list(range(i_step - self.args.ctx_len + 1, i_step + 1))
        else:
            return list(range(-self.args.ctx_len + i_step + 1, 0)) + list(range(0, i_step + 1))

    @torch.no_grad()
    def collect(self):
        self.agent.eval()
        for i_step in range(args.n_steps):
            self.obss[i_step] = self.obs
            self.dones[i_step] = self.done

            self.dist, self.value = self.agent(self.construct_obs_idx(i_step))
            action = self.dist.sample()

            _, reward, _, _, info = self.env.step(action.cpu().numpy())
            self.obs, self.done = info["obs"], info["done"]

            self.vals[i_step] = self.value
            self.acts[i_step] = action
            self.logits[i_step] = self.dist.logits
            self.rews[i_step] = torch.as_tensor(reward).to(args.device)

        # self.obss[-1], self.dones[-1] = self.obs, self.done
        # _, self.value = self.agent(self.construct_obs_idx(i_step + 1))  # calculate one more value

        self.dists = torch.distributions.Categorical(logits=self.logits)
        self.logprobs = self.dists.log_prob(self.acts)

    @torch.no_grad()
    def calc_gae(self):
        args = self.args
        lastgaelam = 0
        for t in reversed(range(args.n_steps)):
            if t == args.n_steps - 1:
                nextnonterminal = ~self.done
                nextvalues = self.value
            else:
                nextnonterminal = ~self.dones[t + 1]
                nextvalues = self.vals[t + 1]
            delta = self.rews[t] + args.gamma * nextvalues * nextnonterminal - self.vals[t]
            self.advs[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        self.rets = self.advs + self.vals


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
    if args.load_agent is not None:
        agent.load_state_dict(torch.load(args.load_agent))
    torchinfo.summary(agent, input_size=(args.batch_size,) + env.single_observation_space.shape, device=args.device)
    # optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    opt = optim.Adam([{"params": agent.parameters(), "lr": args.lr, "eps": 1e-5}, {"params": encoder.parameters(), "lr": args.lr_tc, "eps": 1e-8}])

    buffer = Buffer(args, env, agent)

    viz_slow = set(np.linspace(0, args.n_updates - 1, 10).astype(int))
    viz_midd = set(np.linspace(0, args.n_updates - 1, 100).astype(int)).union(viz_slow)
    viz_fast = set(np.linspace(0, args.n_updates - 1, 1000).astype(int)).union(viz_midd)

    start_time = time.time()
    dtime_env = 0.0
    dtime_inference = 0.0
    dtime_learning = 0.0

    pbar = tqdm(range(args.n_updates))
    for i_update in pbar:
        if args.anneal_lr:  # Annealing the rate if instructed to do so.
            frac = 1.0 - (i_update) / args.n_updates
            lrnow = frac * args.lr
            opt.param_groups[0]["lr"] = lrnow

        buffer.collect()
        buffer.calc_gae()

        clipfracs = []
        for i_epoch in range(args.n_epochs):
            # i_env = torch.randint(0, args.n_envs, (args.batch_size,), device=args.device)
            # i_step = torch.randint(0, args.n_steps + 1 - args.ctx_len, (args.batch_size,), device=args.device)
            i_env = torch.randint(0, args.n_envs, (args.batch_size,), device="cpu")
            i_step = torch.randint(0, args.n_steps + 1 - args.ctx_len, (args.batch_size,), device="cpu")

            b_obs = torch.stack([buffer.obss[i : i + args.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
            b_act = torch.stack([buffer.acts[i : i + args.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
            b_logits = torch.stack([buffer.logits[i : i + args.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
            b_logprobs = torch.stack([buffer.logprobs[i : i + args.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
            b_adv = torch.stack([buffer.advs[i : i + args.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
            b_ret = torch.stack([buffer.rets[i : i + args.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
            b_val = torch.stack([buffer.vals[i : i + args.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])

            new_dist, new_val = agent(b_obs)
            new_logprob = new_dist.log_prob(b_act)
            new_entropy = new_dist.entropy()

            logratio = new_logprob - b_logprobs
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            if args.norm_adv:
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            pg_loss1 = -b_adv * ratio
            pg_loss2 = -b_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            if args.clip_vloss:
                v_loss_unclipped = (new_val - b_ret) ** 2
                v_clipped = b_val + (new_val - b_val).clamp(-args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_ret) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_val - b_ret) ** 2).mean()

            entropy_loss = new_entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            # obs_anc, obs_pos, obs_neg = sample_contrastive_batch(obs[:, :, -1, :, :], p=0.1, batch_size=args.batch_size)
            # loss_tc = calc_contrastive_loss(encoder, obs_anc, obs_pos, obs_neg)
            # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + loss_tc

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            opt.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = buffer.vals.flatten().cpu().numpy(), buffer.rets.flatten().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        data = {}
        if i_update in viz_fast:  # log ex. points with fast frequency
            data["details/lr"] = opt.param_groups[0]["lr"]
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

            # save agent
            if args.save_agent is not None:
                print("Saving agent...")
                os.makedirs(os.path.dirname(args.save_agent), exist_ok=True)
                torch.save(agent.state_dict(), f"{args.save_agent}")

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
