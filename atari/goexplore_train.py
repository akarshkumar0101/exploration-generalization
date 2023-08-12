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
parser.add_argument("--entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--project", type=str, default=None, help="the wandb's project name")
parser.add_argument("--name", type=str, default=None, help="the name of this experiment")
parser.add_argument("--log-video", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--log-hist", type=lambda x: bool(strtobool(x)), default=False)

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")

# Algorithm arguments
parser.add_argument("--env-ids", type=str, nargs="+", default=["Pong"], help="the id of the environment")
parser.add_argument("--obj", type=str, default="ext", help="the objective of the agent, either ext or e3b")
parser.add_argument("--total-steps", type=lambda x: int(float(x)), default=10000000, help="total timesteps of the experiments")
parser.add_argument("--n-envs", type=int, default=8, help="the number of parallel game environments")
parser.add_argument("--n-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--batch-size", type=int, default=256, help="the batch size for training")
parser.add_argument("--n-updates", type=int, default=16, help="gradient updates per collection")

parser.add_argument("--model", type=str, default="cnn", help="either cnn or gpt")
parser.add_argument("--ctx-len", type=int, default=4, help="agent's context length")
parser.add_argument("--load-agent-history", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--load-agent", type=str, default=None, help="file to load the agent from")
parser.add_argument("--save-agent", type=str, default=None, help="file to periodically save the agent to")
parser.add_argument("--full-action-space", type=lambda x: bool(strtobool(x)), default=True)

parser.add_argument("--lr", type=float, default=2.5e-4, help="the learning rate of the optimizer")
parser.add_argument("--lr-warmup", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--lr-decay", type=str, default="none")
parser.add_argument("--max-grad-norm", type=float, default=1.0, help="the maximum norm for the gradient clipping")

parser.add_argument("--episodic-life", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--norm-rew", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, help="Toggles advantages normalization")
parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
parser.add_argument("--max-kl-div", type=float, default=None, help="the target KL divergence threshold")

parser.add_argument("--pre-obj", type=str, default="ext")
parser.add_argument("--train-klbc", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--model-teacher", type=str, default="cnn", help="either cnn or gpt")
parser.add_argument("--ctx-len-teacher", type=int, default=4, help="agent's context length")
parser.add_argument("--load-agent-teacher", type=str, default=None, help="file to load the agent from")
parser.add_argument("--teacher-last-k", type=int, default=1)
parser.add_argument("--pre-teacher-last-k", type=int, default=1)

parser.add_argument("--n-steps-rnd-init", type=lambda x: int(float(x)), default=0)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for key in ["project", "name", "load_agent", "save_agent", "load_agent_teacher"]:
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).format(**vars(args)))
    args.n_envs_per_id = args.n_envs // len(args.env_ids)
    args.collect_size = args.n_envs * args.n_steps
    args.n_collects = args.total_steps // args.collect_size
    return args


from collections import defaultdict
from functools import partial


from buffers import Buffer, MultiBuffer
from env_atari import MyEnvpool, ToTensor

from einops import repeat

from torch.nn.functional import cross_entropy


class GEBuffer(Buffer):
    def __init__(self, env, n_steps, sample_traj_fn, device=None):
        super().__init__(env, n_steps, device=device)
        self.trajs = [None for _ in range(self.env.n_envs)]
        self.traj_lens = np.zeros(self.env.n_envs, dtype=int)
        self.i_locs = np.zeros(self.env.n_envs, dtype=int)
        self.sample_traj_fn = sample_traj_fn

    def sample_new_traj(self, ids):
        for id in ids:
            self.trajs[id] = self.sample_traj_fn()
            self.traj_lens[id] = len(self.trajs[id])
            self.i_locs[id] = 0
        obs, _ = self.env.reset_subenvs(ids)
        self.next_obs[ids] = torch.from_numpy(obs).to(self.device)

    def gecollect(self, pbar=None):
        if self.first_collect:
            self.first_collect = False
            _, info = self.env.reset()
            self.next_obs = info["obs"]
            self.sample_new_traj(np.arange(self.env.num_envs))
        for t in range(self.n_steps):
            self.obss[:, t] = self.next_obs
            action = np.array([traj[i_loc] for traj, i_loc in zip(self.trajs, self.i_locs)])
            self.i_locs += 1
            self.acts[:, t] = torch.from_numpy(action)
            _, _, _, _, info = self.env.step(action)
            self.dones[:, t] = info["done"]
            self.next_obs = info["obs"]
            self.sample_new_traj(np.where(self.i_locs >= self.traj_lens)[0])

    # data["trajs"] = np.array([np.array(cell.trajectory, dtype=np.uint8) for cell in archive.values()], dtype=object)
    # data["rets"] = np.array([cell.running_ret for cell in archive.values()])
    # data["scores"] = np.array([cell.score for cell in archive.values()])


def sample_traj(archives, sampling="uniform"):
    archive = np.random.choice(archives)
    trajs, rets, scores = archive["trajs"], archive["rets"], archive["scores"]
    ret_best = np.max(rets)
    trajs_best = trajs[rets == ret_best]
    if sampling == "uniform":
        return np.random.choice(trajs)
    elif sampling == "best_n":
        return np.random.choice(trajs_best)
    elif sampling == "best_1":
        return trajs_best[-1]


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open("atari_games_ignore.txt") as f:
        env_ids_ignore = [a.strip() for a in f.readlines()]
    with open("atari_games_104.txt") as f:
        env_ids = [a.strip() for a in f.readlines()]
    env_ids = [env_id for env_id in env_ids if env_id not in env_ids_ignore]

    env_id2archives = defaultdict(list)
    for env_id in tqdm(env_ids):
        for seed in range(0, 1):
            f = f"./data/goexplore/goexplore_{env_id}_{seed}.npy"
            if os.path.exists(f):
                # print(f"Trying {env_id}")
                archive = np.load(f"./data/goexplore/goexplore_{env_id}_{seed}.npy", allow_pickle=True).item()
                env_id2archives[env_id].append(archive)
    env_id2archives = dict(env_id2archives)
    env_id2archives = {k: env_id2archives[k] for k in list(env_id2archives.keys())[:30]}
    # env_id2archives = {k: env_id2archives[k] for k in ["BeamRider"]}

    for env_id, archives in env_id2archives.items():
        print(f"{env_id}: {len(archives)} archives")
    print(f"{len(env_id2archives)} games with archives out of {len(env_ids)} games total")

    print("Creating buffers...")
    mbuffer = MultiBuffer()
    for env_id in tqdm(env_id2archives):
        env = MyEnvpool(
            f"{env_id}-v5", num_envs=8, img_height=84, img_width=84, gray_scale=True, stack_num=1, frame_skip=4, repeat_action_probability=0.0, noop_max=1, use_fire_reset=False, full_action_space=True
        )
        env = ToTensor(env, device=args.device)

        sample_traj_fn = partial(sample_traj, archives=env_id2archives[env_id], sampling="best_1")
        gebuff = GEBuffer(env, 128, sample_traj_fn=sample_traj_fn, device=args.device)
        mbuffer.buffers.append(gebuff)
    # print("Done creating buffers")

    agent = utils.create_agent("gpt", 18, 16, None, device=args.device)
    opt = agent.create_optimizer(lr=1e-4)
    for i_iter in tqdm(range(1000)):
        print("Collecting...")
        for env_id, buf in zip(env_id2archives.keys(), mbuffer.buffers):
            buf.gecollect()
            if buf.dones.any().item():
                print(env_id)
                a = buf.dones.any(dim=-1)
            assert not buf.dones.any().item(), "No dones expected"
        # for _ in range(16):
        #     bs = len(env_id2archives) * 128
        #     batch = mbuffer.generate_batch(bs, 16)
        #     logits, values = agent(None, batch["obs"], batch["act"], None)

        #     loss_bc = cross_entropy(rearrange(logits, "b t d -> (b t) d"), rearrange(batch["act"], "b t -> (b t)"), reduction="none")
        #     loss_bc = rearrange(loss_bc, "(b t) -> b t", b=bs)
        #     loss = loss_bc.mean()

        #     opt.zero_grad()
        #     loss.backward()
        #     opt.step()
        #     print(loss.item())

    # import imageio
    # vid = rearrange(buf.obss[0], "t 1 h w -> t h w 1").detach().cpu().numpy()
    # vid = repeat(vid, "t h w 1 -> t (h 5) (w 5) 1")
    # print(vid.shape, vid.dtype, vid.min(), vid.max())
    # imageio.mimsave("temp.gif", vid, fps=30)


if __name__ == "__main__":
    main(parse_args())
