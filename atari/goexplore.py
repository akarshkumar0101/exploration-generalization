import argparse
import os
from collections import defaultdict
from distutils.util import strtobool

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import random

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--entity", type=str, default=None)
parser.add_argument("--project", type=str, default="goexplore")
parser.add_argument("--name", type=str, default=None)
# parser.add_argument("--log-video", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--log-hist", type=lambda x: bool(strtobool(x)), default=False)

# parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--env-id", type=str, default="MontezumaRevenge")
parser.add_argument("--n-iters", type=lambda x: int(float(x)), default=int(1e3))
parser.add_argument("--n-steps", type=int, default=100)
parser.add_argument("--p-repeat", type=float, default=0.0)

parser.add_argument("--save-archive", type=str, default=None)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for key in ["project", "name", "save_archive"]:
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).format(**vars(args)))
    return args


def cellfn(frame, h=8, w=11, d=8):
    cell = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cell = cv2.resize(cell, (w, h), interpolation=cv2.INTER_AREA)
    cell = cell // (256 // d)
    return cell


def hashfn(cell):
    return hash(cell.tobytes())


e1 = 0.001
e2 = 0.00001
weight_dict = dict(times_chosen=0.1, times_chosen_since_new=0, times_seen=0.3)
power_dict = dict(times_chosen=0.5, times_chosen_since_new=0.5, times_seen=0.5)


class Cell(object):
    def __init__(self):
        self.times_chosen = 0
        self.times_chosen_since_new = 0
        self.times_seen = 0

        self.ram = None
        self.running_ret = 0
        self.trajectory = []

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if key != "score" and hasattr(self, "times_seen"):
            self.score = self.cellscore()

    def cntscore(self, a):
        w, p = weight_dict[a], power_dict[a]
        v = getattr(self, a)
        return w / (v + e1) ** p + e2

    def cellscore(self):
        return self.cntscore("times_chosen") + self.cntscore("times_chosen_since_new") + self.cntscore("times_seen") + 1

    def visit(self):
        self.times_seen += 1
        return self.times_seen == 1

    def choose(self):
        self.times_chosen += 1
        self.times_chosen_since_new += 1
        return self.ram, self.running_ret, self.trajectory.copy()


def main(args):
    print(f"Running Goexplore with args: {args}")
    np.random.seed(args.seed)
    random.seed(args.seed)

    env2 = gym.make(f"ALE/{args.env_id}-v5", frameskip=1, repeat_action_probability=0.0)
    env2 = gym.wrappers.AtariPreprocessing(env2, noop_max=1, frame_skip=4, screen_size=210, grayscale_obs=False)
    env = gym.make(f"ALE/{args.env_id}-v5", frameskip=1, repeat_action_probability=0.0)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=1, frame_skip=4, screen_size=210, grayscale_obs=False)
    _, info = env.reset()
    max_lives = info["lives"]
    running_ret = 0
    action = 0
    trajectory = []

    max_traj_len = 0
    max_running_ret = 0

    archive = defaultdict(lambda: Cell())

    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args)

    for i_iter in tqdm(range(args.n_iters)):
        # ------------------------------- DATA COLLECTION ------------------------------- #
        # if i_iter>1:
        #     plt.figure(figsize=(10, 10))
        #     ak_cells = list(archive.values())
        #     for ak in range(len(ak_cells)):
        #         akcell = ak_cells[ak]
        #         env2.reset()
        #         env.restore_state(akcell.ram)
        #         ako, _, _, _, _ = env.step(0)
        #         plt.subplot(2, 6, ak+1)
        #         plt.title(f'{akcell.score: 8.5f}')
        #         plt.imshow(ako)
        #     plt.tight_layout()
        #     plt.show()

        found_new_cell = False
        for i_step in range(args.n_steps):
            if i_iter == 0 and i_step == 0:
                action = 0
            elif not np.random.random() < args.p_repeat:
                # action = env.action_space.sample()
                action = np.random.randint(0, env.action_space.n, 1).item()
            frame, reward, terminal, trunc, info = env.step(action)
            running_ret += reward
            terminal |= info["lives"] < max_lives
            terminal |= trunc
            trajectory.append(action)
            max_traj_len = max(max_traj_len, len(trajectory))
            max_running_ret = max(max_running_ret, running_ret)
            if terminal:
                break
            pixcell = cellfn(frame)
            cellhash = hashfn(pixcell)
            cell = archive[cellhash]
            first_visit = cell.visit()
            if first_visit or running_ret > cell.running_ret or running_ret == cell.running_ret and len(trajectory) < len(cell.trajectory):
                cell.cell_raw = pixcell.flatten().tolist()
                cell.ram = env.clone_state(True)
                cell.running_ret = running_ret
                cell.trajectory = trajectory.copy()
                cell.times_chosen = 0
                cell.times_chosen_since_new = 0
                found_new_cell = True
        if found_new_cell and i_iter > 0:
            restore_cell.times_chosen_since_new = 0
        scores = np.array([cell.score for cell in archive.values()])
        hashes = [cellhash for cellhash in archive.keys()]
        probs = scores / scores.sum()
        restore = np.random.choice(hashes, p=probs)
        restore_cell = archive[restore]
        ram, running_ret, trajectory = restore_cell.choose()
        env.reset()
        env.restore_state(ram)

        # ------------------------------- LOGGING ------------------------------- #
        viz_slow = i_iter % (args.n_iters // 10) == 0
        viz_midd = i_iter % (args.n_iters // 10) == 0 or viz_slow
        viz_fast = i_iter % (args.n_iters // 100) == 0 or viz_midd
        data = {}
        if viz_fast:
            data["n_cells"] = len(archive)
            # data["frames"] = 0.0
            data["max_traj_len"] = max_traj_len
            data["max_running_ret"] = max_running_ret
        # if viz_slow:
        # plt.figure(figsize=(6, 3))
        # plt.subplot(121)
        # plt.imshow(frame)
        # plt.subplot(122)
        # plt.imshow(pixcell)
        # plt.tight_layout()
        # data["cell_repr"] = plt.gcf()
        # plt.close()
        if args.log_hist and viz_slow:
            traj_lens = np.array([len(cell.trajectory) for cell in archive.values()])
            traj_rets = np.array([cell.running_ret for cell in archive.values()])
            data["traj_lens"] = wandb.Histogram(traj_lens)
            data["traj_rets"] = wandb.Histogram(traj_rets)
        if args.track and viz_fast:
            wandb.log(data, step=i_iter)
        if args.save_archive is not None and viz_slow:
            os.makedirs(os.path.dirname(args.save_archive), exist_ok=True)
            np.save(args.save_archive, dict(archive))

        if viz_slow:
            print(f"i_iter: {i_iter: 10d}, n_cells: {len(archive): 10d}, frames: 0, max_running_ret: {max_running_ret: 9.1f}")
    return archive


if __name__ == "__main__":
    main(parse_args())


"""
# gymnasium
env = gym.make(f"ALE/MontezumaRevenge-v5", frameskip=1, repeat_action_probability=0.0)
env = gym.wrappers.AtariPreprocessing(env, noop_max=1, frame_skip=4, screen_size=210, grayscale_obs=False)


# envpool
env = envpool.make_gymnasium('MontezumaRevenge-v5', img_height=210, img_width=210, gray_scale=False, stack_num=1, frame_skip=4, repeat_action_probability=0.0, noop_max=1, use_fire_reset=False)
"""
