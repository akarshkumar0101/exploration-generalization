# Adapted from CleanRL's ppo_atari_envpool.py
import argparse
import glob
import os
import random
import time
from collections import defaultdict
from distutils.util import strtobool
from functools import partial

import gymnasium as gym
import numpy as np
import timers
import torch
import torchinfo
import wandb
from einops import rearrange, repeat
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy
from tqdm.auto import tqdm
from my_envs import *

from my_buffers import Buffer

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--entity", type=str, default=None)
parser.add_argument("--project", type=str, default=None)
parser.add_argument("--name", type=str, default=None)
# parser.add_argument("--log-video", type=lambda x: bool(strtobool(x)), default=False)
# parser.add_argument("--log-hist", type=lambda x: bool(strtobool(x)), default=False)

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)

# Algorithm arguments
parser.add_argument("--env_ids", type=str, nargs="+", default=["MontezumaRevenge"])
parser.add_argument("--strategy", type=str, default="best")
parser.add_argument("--n_iters", type=lambda x: int(float(x)), default=int(1e3))
parser.add_argument("--n_envs", type=int, default=4)
parser.add_argument("--n_steps", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=384)
parser.add_argument("--n_updates", type=int, default=32)

parser.add_argument("--ctx_len", type=int, default=64)
parser.add_argument("--save_agent", type=lambda x: None if x.lower() == "none" else x, default=None)

parser.add_argument("--lr", type=float, default=1e-4)
# parser.add_argument("--lr-warmup", type=lambda x: bool(strtobool(x)), default=True)
# parser.add_argument("--lr-decay", type=str, default="none")
# parser.add_argument("--max-grad-norm", type=float, default=1.0, help="the maximum norm for the gradient clipping")

parser.add_argument("--ge_data_dir", type=str, default=None)
parser.add_argument("--n_archives", type=int, default=1)


def parse_args(*args, **kwargs):
    args, uargs = parser.parse_known_args(*args, **kwargs)
    args.n_envs_per_id = args.n_envs // len(args.env_ids)
    for k, v in dict([tuple(uarg.replace("--", "").split("=")) for uarg in uargs]).items():
        setattr(args, k, v)
    return args


class GEBuffer(Buffer):
    def __init__(self, env, n_steps, sample_traj_fn, device=None):
        # TODO mange devices
        super().__init__(env, None, n_steps, device="cpu")
        self.trajs = [None for _ in range(self.env.num_envs)]
        self.traj_lens = np.zeros(self.env.num_envs, dtype=int)
        self.i_locs = np.zeros(self.env.num_envs, dtype=int)
        self.sample_traj_fn = sample_traj_fn

    def reset_with_newtraj(self, ids):
        assert len(ids) > 0
        for id in ids:
            self.trajs[id] = self.sample_traj_fn(id)
            self.traj_lens[id] = len(self.trajs[id])
            self.i_locs[id] = 0
        obs, info = self.env.reset_subenvs(ids)#, seed=[0 for _ in ids])
        # assert np.array_equal(self.env.envs[id].ale.getRAM(), self.ram_start), "env reset to seed=0"
        return obs

    def gecollect(self, pbar=None):
        if self.first_collect:
            self.first_collect = False
            self.obs = self.reset_with_newtraj(np.arange(self.env.num_envs))
        for t in range(self.n_steps):
            self.data["obs"][:, t] = self.obs
            action = np.array([traj[i_loc] for traj, i_loc in zip(self.trajs, self.i_locs)])
            self.i_locs += 1
            self.data["obs"][:, t] = action
            self.obs, _, term, trunc, _ = self.env.step(action)
            assert not any(term) and not any(trunc), "found a done in the ge buffer"

            ids_reset = np.where(self.i_locs >= self.traj_lens)[0]
            if len(ids_reset) > 0:
                self.obs[ids_reset] = self.reset_with_newtraj(ids_reset)
            if pbar is not None:
                pbar.update(1)


def make_env(args):
    envs = []
    for env_id in args.env_ids:
        envi = MyEnvpool(f"{env_id}-v5", num_envs=args.n_envs, stack_num=1, noop_max=1, use_fire_reset=False, episodic_life=True, reward_clip=False, seed=args.seed, full_action_space=True)
        # envi = ToTensor(envi, device=args.device)
        envs.append(envi)
    env = ConcatEnv(envs)
    return env


def load_env_id2archives(env_ids, ge_data_dir, n_archives):
    env_id2archives = {}
    for env_id in tqdm(env_ids):
        files = sorted(glob.glob(f"{ge_data_dir}/*{env_id}*"))
        files = files[:n_archives]
        env_id2archives[env_id] = [np.load(f, allow_pickle=True).item() for f in files]
    return env_id2archives


def get_env_id2trajs(env_id2archives, strategy="best"):
    env_id2trajs = {}
    for env_id, archives in tqdm(env_id2archives.items()):
        env_id2trajs[env_id] = []
        for archive in archives:
            trajs, rets, novelty, is_leaf = archive["traj"], archive["ret"], archive["novelty"], archive["is_leaf"]
            if strategy == "all":
                idx = np.arange(len(trajs))
            elif strategy == "best":
                idx = np.array([np.argmax(rets)])
            elif strategy == "leaf":
                idx = np.nonzero(is_leaf)[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            trajs = trajs[idx]
            env_id2trajs[env_id].extend(trajs)
    return env_id2trajs


def main(args):
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading archives")
    env_id2archives = load_env_id2archives(args.env_ids, args.ge_data_dir, args.n_archives)
    print("Creating trajs")
    env_id2trajs = get_env_id2trajs(env_id2archives, strategy=args.strategy)
    for env_id, trajs in env_id2trajs.items():
        print(f"env_id: {env_id}, #trajs: {len(trajs)}")

    print("Creating envs")
    env = make_env(args)
    print("Done creating envs!")

    def sample_traj_fn(id):
        env_id = args.env_ids[id // args.n_envs]
        trajs = env_id2trajs[env_id]
        return trajs[np.random.choice(len(trajs))]

    print("Creating buffer")
    buf = GEBuffer(env, args.n_steps, sample_traj_fn=sample_traj_fn, device=args.device)
    for _ in tqdm(range(100)):
        buf.gecollect()

    # print("Creating agent")
    # agent = utils.create_agent("gpt", 18, args.ctx_len, load_agent=None, device=args.device)
    # opt = agent.create_optimizer(lr=args.lr)

    # print("Starting learning")
    # pbar_iters = tqdm(total=args.n_iters)
    # pbar_steps = tqdm(total=args.n_steps)
    # pbar_updates = tqdm(total=args.n_updates)
    # for i_iter in range(args.n_iters):
    #     pbar_iters.update(1)

    #     pbar_steps.reset(total=args.n_steps)
    #     buf.gecollect(pbar=pbar_steps)

    #     pbar_updates.reset(total=args.n_updates)
    #     for i_update in range(args.n_updates):
    #         pbar_updates.update(1)
    #         assert args.batch_size % (len(args.env_ids) * args.n_envs) == 0
    #         batch = buf.generate_batch(args.batch_size, args.ctx_len)
    #         obs, act = batch["obs"], batch["act"]
    #         obs = obs[:, :, None, :, :]
    #         logits, values = agent(None, obs, act, None)

    #         loss_bc = torch.nn.functional.cross_entropy(rearrange(logits, "b t d -> (b t) d"), rearrange(act, "b t -> (b t)"), reduction="none")
    #         loss_bc = rearrange(loss_bc, "(b t) -> b t", b=args.batch_size)
    #         loss = loss_bc.mean()

    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #         # if i % 50 == 0:
    #         # print(np.e ** loss.item())

    #         if args.track:
    #             data = {}
    #             data["loss"] = loss.item()
    #             data["ppl"] = np.e ** loss.item()

    #             if i_iter % (args.n_iters // 100) == 0 and i_update == 0:
    #                 ppl = loss_bc.mean(dim=0).exp().detach().cpu().numpy()
    #                 pos = np.arange(len(ppl))
    #                 table = wandb.Table(data=np.stack([pos, ppl], axis=-1), columns=["ctx_pos", "ppl"])
    #                 data["ppl_vs_ctx_pos"] = wandb.plot.line(table, "ctx_pos", "ppl", title="PPL vs Context Position")
    #             wandb.log(data)

    #     if args.save_agent is not None and i_iter % (args.n_iters // 100) == 0:
    #         save_agent = args.save_agent.format(**locals())
    #         os.makedirs(os.path.dirname(save_agent), exist_ok=True)
    #         torch.save(agent.state_dict(), save_agent)


# data["trajs"] = np.array([np.array(cell.trajectory, dtype=np.uint8) for cell in archive.values()], dtype=object)
# data["rets"] = np.array([cell.running_ret for cell in archive.values()])
# data["scores"] = np.array([cell.score for cell in archive.values()])


if __name__ == "__main__":
    main(parse_args())


# TODO do goexplore runs with different seeds


# plt.plot(pos, ppl)
# plt.ylim(0, args.vocab_size)
# plt.ylabel("PPL")
# plt.xlabel("Token Position")
# data["mpl"] = plt
