# Adapted from CleanRL's ppo_atari_envpool.py
import argparse
import os
import random
import time

import hns
import numpy as np
import torch
from my_agents import *
from my_buffers import *
from my_envs import *
from torch import nn
from tqdm.auto import tqdm

import wandb

mystr = lambda x: None if x.lower() == "none" else x
mybool = lambda x: x.lower() == "true"
myint = lambda x: int(float(x))

parser = argparse.ArgumentParser()
# Wandb arguments
parser.add_argument("--track", type=mybool, default=False)
parser.add_argument("--entity", type=mystr, default=None)
parser.add_argument("--project", type=mystr, default=None)
parser.add_argument("--name", type=mystr, default=None)

# General arguments
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)

# Model arguments
parser.add_argument("--model", type=str, default="cnn_4")
parser.add_argument("--load_ckpt", type=mystr, default=None)
parser.add_argument("--save_ckpt", type=mystr, default=None)
parser.add_argument("--n_ckpts", type=int, default=1)

# Optimizer arguments
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--clip_grad_norm", type=float, default=1.0)

# Algorithm arguments
parser.add_argument("--env_ids", type=str, nargs="+", default=["Pong"])
parser.add_argument("--n_iters", type=myint, default=10000)
parser.add_argument("--n_envs", type=int, default=8)
parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_updates", type=int, default=16)

# BC arguments
parser.add_argument("--ent_coef", type=float, default=0.000)
# Go-Explore data arguments
parser.add_argument("--ge_data_dir", type=str, default=None)
parser.add_argument("--strategy", type=str, default="best")
parser.add_argument("--n_archives", type=int, default=1)
parser.add_argument("--min_traj_len", type=int, default=150)


def parse_args(*args, **kwargs):
    args, uargs = parser.parse_known_args(*args, **kwargs)
    for k, v in dict([tuple(uarg.replace("--", "").split("=")) for uarg in uargs]).items():
        setattr(args, k, v)
    return args


def calc_ce_loss(dist_student, act):
    b, t, d = dist_student.logits.shape
    logits = dist_student.logits
    logits = rearrange(logits, "b t d -> (b t) d")
    act = rearrange(act, "b t -> (b t)")
    loss = torch.nn.functional.cross_entropy(logits, act, reduction="none")
    loss = rearrange(loss, "(b t) -> b t", b=b)
    return loss


class GEBuffer(Buffer):
    def __init__(self, env, n_steps, sample_traj_fn, device=None):
        # TODO mange devices
        super().__init__(env, None, n_steps, device=device)
        self.trajs = [None for _ in range(self.env.num_envs)]
        self.traj_lens = np.zeros(self.env.num_envs, dtype=int)
        self.i_locs = np.zeros(self.env.num_envs, dtype=int)
        self.sample_traj_fn = sample_traj_fn

        # shape = tuple(self.data["obs"].shape)
        # self.data["obs"] = np.zeros(shape, dtype=np.uint8)
        # self.data["act"] = np.zeros(shape[:2], dtype=int)

    def reset_with_newtraj(self, ids):
        assert len(ids) > 0
        for id in ids:
            self.trajs[id] = self.sample_traj_fn(id)
            self.traj_lens[id] = len(self.trajs[id])
            self.i_locs[id] = 0
        obs, info = self.env.reset_subenvs(ids)  # , seed=[0 for _ in ids])
        # assert np.array_equal(self.env.envs[id].ale.getRAM(), self.ram_start), "env reset to seed=0"
        return obs

    def collect(self, pbar=None):
        if self.first_collect:
            self.first_collect = False
            self.obs, _ = self.env.reset()
            self.obs = self.reset_with_newtraj(np.arange(self.env.num_envs))
        self.n_resets = 0
        for t in range(self.n_steps):
            self.data["obs"][:, t] = self.obs
            action = np.array([traj[i_loc] for traj, i_loc in zip(self.trajs, self.i_locs)])
            action = torch.from_numpy(action).to(self.device)
            self.i_locs += 1
            self.data["act"][:, t] = action
            self.obs, _, term, trunc, _ = self.env.step(action)
            assert not any(term) and not any(trunc), "found a done in the ge buffer"

            ids_reset = np.where(self.i_locs >= self.traj_lens)[0]
            if len(ids_reset) > 0:
                self.obs[ids_reset] = self.reset_with_newtraj(ids_reset)
                self.n_resets += len(ids_reset)
            if pbar is not None:
                pbar.update(1)


def load_env_id2archives(env_ids, ge_data_dir, n_archives):
    import glob

    env_id2archives = {}
    for env_id in tqdm(env_ids):
        files = sorted(glob.glob(f"{ge_data_dir}/*{env_id}*"))
        files = files[:n_archives]
        env_id2archives[env_id] = [np.load(f, allow_pickle=True).item() for f in files]
    return env_id2archives


def get_env_id2trajs(env_id2archives, strategy="best", min_traj_len=150):
    env_id2trajs = {}
    for env_id, archives in tqdm(env_id2archives.items()):
        env_id2trajs[env_id] = []
        for archive in archives:
            trajs, rets, novelty, is_leaf = archive["traj"], archive["ret"], archive["novelty"], archive["is_leaf"]
            lens = np.array([len(traj) for traj in trajs])
            lenmask = lens > min_traj_len
            assert lenmask.sum().item() >= 1
            trajs, rets, novelty, is_leaf = trajs[lenmask], rets[lenmask], novelty[lenmask], is_leaf[lenmask]
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


def make_env(args):
    envs = []
    for env_id in args.env_ids:
        envi = MyEnvpool(f"{env_id}-v5", num_envs=args.n_envs, stack_num=1, frame_skip=4, repeat_action_probability=0.0, noop_max=1, use_fire_reset=False, full_action_space=True, seed=0)
        envi = RecordEpisodeStatistics(envi, deque_size=32)
        envs.append(envi)
    env = ConcatEnv(envs)
    env = ToTensor(env, device=args.device)
    return env


def main(args):
    print("Running GEBC with args: ", args)
    print("Starting wandb...")
    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args, save_code=True)

    print("Seeding...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Creating environment...")
    env = make_env(args)
    env_teacher = make_env(args)

    print("Creating agent...")
    agent = make_agent(args.model, 18).to(args.device)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    if args.load_ckpt is not None:
        print("Loading checkpoint...")
        ckpt = torch.load(args.load_ckpt, map_location=args.device)
        agent.load_state_dict(ckpt["agent"])

    print("Loading archives")
    env_id2archives = load_env_id2archives(args.env_ids, args.ge_data_dir, args.n_archives)
    print("Creating trajs")
    env_id2trajs = get_env_id2trajs(env_id2archives, strategy=args.strategy, min_traj_len=args.min_traj_len)
    for env_id, trajs in env_id2trajs.items():
        print(f"env_id: {env_id}, #trajs: {len(trajs)}")

    def sample_traj_fn(id):
        env_id = args.env_ids[id // args.n_envs]
        trajs = env_id2trajs[env_id]
        return trajs[np.random.choice(len(trajs))]

    print("Creating buffer...")
    buffer = Buffer(env, agent, args.n_steps, device=args.device)
    buffer_teacher = GEBuffer(env_teacher, args.n_steps, sample_traj_fn=sample_traj_fn, device=args.device)

    print("Warming up buffer...")
    for i_iter in tqdm(range(40), leave=False):
        buffer.collect()
        buffer_teacher.collect()

    start_time = time.time()
    print("Starting learning...")
    for i_iter in tqdm(range(args.n_iters)):
        buffer.collect()
        buffer_teacher.collect()

        for _ in range(args.n_updates):
            batch = buffer_teacher.generate_batch(args.batch_size, ctx_len=agent.ctx_len)

            logits, val = agent(done=batch["done"], obs=batch["obs"], act=batch["act"], rew=batch["rew"])
            dist, batch_act = torch.distributions.Categorical(logits=logits), torch.distributions.Categorical(logits=batch["act"])

            loss_bc = calc_ce_loss(dist, batch_act)
            assert loss_bc.shape == (args.batch_size, agent.ctx_len)
            loss_e = dist.entropy()

            if not agent.train_per_token:
                loss_bc, loss_e = loss_bc[:, [-1]], loss_e[:, [-1]]
            loss = 1.0 * loss_bc.mean() - args.ent_coef * loss_e.mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.clip_grad_norm)
            opt.step()

        if args.save_ckpt is not None and args.n_ckpts > 0 and (i_iter + 1) % (args.n_iters // args.n_ckpts) == 0:
            print(f"Saving Checkpoint at {i_iter}/{args.n_iters} iterations")
            file = args.save_ckpt.format(i_iter=i_iter)
            ckpt = dict(args=args, i_iter=i_iter, agent=agent.state_dict())
            os.makedirs(os.path.dirname(file), exist_ok=True)
            torch.save(ckpt, file)

        viz_slow = i_iter % np.clip(args.n_iters // 10, 1, None) == 0
        viz_midd = i_iter % np.clip(args.n_iters // 100, 1, None) == 0 or viz_slow
        viz_fast = i_iter % np.clip(args.n_iters // 1000, 1, None) == 0 or viz_midd

        data = {}
        if viz_fast:
            for envi in env.envs:
                data[f"charts/{envi.env_id}_score"] = np.mean(envi.traj_rets)
                data[f"charts/{envi.env_id}_tlen"] = np.mean(envi.traj_lens)
                data[f"charts/{envi.env_id}_score_max"] = np.max(envi.traj_rets)
                low, high = hns.atari_human_normalized_scores[envi.env_id]
                data["charts/hns"] = (np.mean(envi.traj_rets) - low) / (high - low)
            for envi in env_teacher.envs:
                data[f"charts_teacher/{envi.env_id}_score"] = np.mean(envi.traj_rets)
                data[f"charts_teacher/{envi.env_id}_tlen"] = np.mean(envi.traj_lens)
                data[f"charts_teacher/{envi.env_id}_score_max"] = np.max(envi.traj_rets)
                low, high = hns.atari_human_normalized_scores[envi.env_id]
                data["charts_teacher/hns"] = (np.mean(envi.traj_rets) - low) / (high - low)

            env_steps = (i_iter + 1) * len(args.env_ids) * args.n_envs * args.n_steps
            grad_steps = (i_iter + 1) * args.n_updates
            data["env_steps"] = env_steps
            data["grad_steps"] = grad_steps
            sps = int(env_steps / (time.time() - start_time))
            data["meta/SPS"] = sps

            data["loss"] = loss.item()
            data["ppl"] = np.e ** loss.item()
        if viz_midd:
            ppl = loss_bc.mean(dim=0).exp().detach().cpu().numpy()
            pos = np.arange(len(ppl))
            table = wandb.Table(data=np.stack([pos, ppl], axis=-1), columns=["ctx_pos", "ppl"])
            data["ppl_vs_ctx_pos"] = wandb.plot.line(table, "ctx_pos", "ppl", title="PPL vs Context Position")

        if args.track and viz_fast:
            wandb.log(data)


if __name__ == "__main__":
    main(parse_args())
