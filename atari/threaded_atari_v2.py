from collections import defaultdict, namedtuple
from threading import Thread
from time import sleep
import numpy as np
import cv2
import gymnasium as gym
import wandb

from distutils.util import strtobool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--entity", type=str, default=None)
parser.add_argument("--project", type=str, default="goexplore")
parser.add_argument("--name", type=str, default=None)
# parser.add_argument("--log-video", type=lambda x: bool(strtobool(x)), default=False)
# parser.add_argument("--log-hist", type=lambda x: bool(strtobool(x)), default=False)

# parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument("--env-id", type=str, default="MontezumaRevenge")

parser.add_argument("--n-iters", type=int, default=1000000)

e1 = 0.001
e2 = 0.00001


def cellfn(frame):
    cell = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cell = cv2.resize(cell, (11, 8), interpolation=cv2.INTER_AREA)
    cell = cell // 32
    return cell


def hashfn(cell):
    return hash(cell.tobytes())


class Weights:
    times_chosen = 0.1
    times_chosen_since_new = 0
    times_seen = 0.3


class Powers:
    times_chosen = 0.5
    times_chosen_since_new = 0.5
    times_seen = 0.5


class Cell(object):
    def __init__(self):
        self.times_chosen = 0
        self.times_chosen_since_new = 0
        self.times_seen = 0

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if key != "score" and hasattr(self, "times_seen"):
            self.score = self.cellscore()

    def cntscore(self, a):
        w = getattr(Weights, a)
        p = getattr(Powers, a)
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
        return self.ram, self.reward, self.trajectory.copy()


def explore(id):
    global highscore, frames, iterations, best_cell, new_cell, archive

    env = gym.make(f"ALE/{args.env_id}-v5", frameskip=1, repeat_action_probability=0.0)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=1, frame_skip=4, screen_size=210, grayscale_obs=False)
    frame = env.reset()
    score = 0
    action = 0
    trajectory = []
    my_iterations = 0

    sleep(id / 10)

    while True:
        found_new_cell = False
        episode_length = 0

        for i in range(100):
            if np.random.random() > 0.95:
                action = env.action_space.sample()

            frame, reward, terminal, trunc, info = env.step(action)
            score += reward
            terminal |= info["lives"] < max_lives
            terminal |= trunc

            trajectory.append(action)
            episode_length += 4

            if score > highscore:
                highscore = score
                best_cell = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if terminal:
                frames += episode_length
                break
            else:
                cell = cellfn(frame)
                cellhash = hashfn(cell)
                cell = archive[cellhash]
                first_visit = cell.visit()
                if first_visit or score > cell.reward or score == cell.reward and len(trajectory) < len(cell.trajectory):
                    cell.ram = env.clone_state(True)
                    cell.reward = score
                    cell.trajectory = trajectory.copy()
                    cell.times_chosen = 0
                    cell.times_chosen_since_new = 0
                    new_cell = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    found_new_cell = True

        if found_new_cell and my_iterations > 0:
            restore_cell.times_chosen_since_new = 0

        scores = np.array([cell.score for cell in archive.values()])
        hashes = [cellhash for cellhash in archive.keys()]
        probs = scores / scores.sum()
        restore = np.random.choice(hashes, p=probs)
        restore_cell = archive[restore]
        ram, score, trajectory = restore_cell.choose()
        env.reset()
        env.step(1)
        env.restore_state(ram)
        my_iterations += 1
        iterations += 1


if __name__ == "__main__":
    args = parser.parse_args()

    env = gym.make(f"ALE/{args.env_id}-v5", frameskip=4, repeat_action_probability=0.0)
    _, info = env.reset()
    max_lives = info["lives"]

    archive = defaultdict(lambda: Cell())
    highscore = 0
    frames = 0
    iterations = 0

    best_cell = np.zeros((1, 1, 3))
    new_cell = np.zeros((1, 1, 3))

    if args.track:
        wandb.init(entity=args.entity, project=args.project, name=args.name, config=args)

    threads = [Thread(target=explore, args=(id,)) for id in range(args.n_threads)]

    for thread in threads:
        thread.start()

    iterations_last_log = -1

    i_main = 0
    while True:
        if i_main % 10 == 0:
            print("Iterations: %d, Cells: %d, Frames: %d, Max Reward: %d" % (iterations, len(archive), frames, highscore))
        # image = np.concatenate((best_cell, new_cell), axis = 1)
        # cv2.imshow('best cell – newest cell', image)
        # cv2.waitKey(1)
        sleep(1)

        if iterations_last_log != iterations - iterations % (args.n_iters // 1000):
            iterations_last_log = iterations - iterations % (args.n_iters // 1000)
            if args.track:
                data = {"iterations": iterations, "cells": len(archive), "frames": frames, "max reward": highscore}
                wandb.log(data, step=iterations)

        if i_main % 100 == 0:
            print("saving archive")
            np.save("archive.npy", dict(archive))
            print("done saving")

        i_main += 1

"""
# gymnasium
env = gym.make(f"ALE/MontezumaRevenge-v5", frameskip=1, repeat_action_probability=0.0)
env = gym.wrappers.AtariPreprocessing(env, noop_max=1, frame_skip=4, screen_size=210, grayscale_obs=False)


# envpool
env = envpool.make_gymnasium('MontezumaRevenge-v5', img_height=210, img_width=210, gray_scale=False, stack_num=1, frame_skip=4, repeat_action_probability=0.0, noop_max=1, use_fire_reset=False)
"""
