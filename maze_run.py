import argparse
import time
from distutils.util import strtobool
from functools import partial
from multiprocessing import Pool

import gymnasium as gym
import maze
import numpy as np
import torch
from tqdm.auto import tqdm

import env_utils
import goexplore_discrete
import maze_env


def make_single_env(maze, obs_size, frame_stack):
    env = maze_env.MazeEnv(maze, obs_size=obs_size)
    # env = env_utils.StoreObsInfo(env)
    env = gym.wrappers.FrameStack(env, frame_stack)
    env = env_utils.DeterministicReplayReset(env)
    return env

def make_env(n_envs, auto_reset=False, **kwargs):
    make_env_fn = partial(make_single_env, **kwargs)
    env = env_utils.RestorableSyncVectorEnv([make_env_fn for i in range(n_envs)], auto_reset=auto_reset)
    env = env_utils.ToTensor(env, device=None, dtype=torch.float32)
    return env

def run_goexplore(maze, obs_size, frame_stack, ge_batch_size, ge_steps, ge_len_traj, ge_beta):
    env = make_env(ge_batch_size, maze=maze, obs_size=obs_size, frame_stack=frame_stack)
    ge = goexplore_discrete.GoExplore(env)
    for i in range(ge_steps):
        nodes = ge.select_nodes(ge_batch_size, beta=ge_beta)
        ge.explore_from(nodes, ge_len_traj)
    return ge

parser = argparse.ArgumentParser()
# # general parameters
# parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
# parser.add_argument("--name", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
# parser.add_argument("--device", type=str, default=None)

# algorithm parameters
parser.add_argument("--n_procs", type=int, default=35)
parser.add_argument("--n_mazes", type=int, default=100)
parser.add_argument("--maze_size", type=int, default=50)
parser.add_argument("--obs_size", type=int, default=5)
parser.add_argument("--frame_stack", type=int, default=4)
parser.add_argument("--ge_batch_size", type=int, default=32)
parser.add_argument("--ge_steps", type=int, default=50)
parser.add_argument("--ge_len_traj", type=int, default=20)
parser.add_argument("--ge_beta", type=float, default=-2.0)

if __name__ == '__main__':
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Generating mazes...')
    mazes_train = maze_env.generate_mazes(args.maze_size, args.maze_size, args.n_mazes, maze.Maze.Create.PRIM, tqdm=tqdm)
    mazes_test = maze_env.generate_mazes(args.maze_size, args.maze_size, args.n_mazes, maze.Maze.Create.PRIM, tqdm=tqdm)
    torch.save(mazes_train, 'data/mazes_train.pt')
    torch.save(mazes_test, 'data/mazes_test.pt')
    print('Done generating mazes.')

    print('Running Go-Explore on mazes...')
    time_start = time.time()
    with Pool(args.n_procs) as pool:
        run_ge_fn = partial(run_goexplore, obs_size=args.obs_size,
                            frame_stack=args.frame_stack, ge_batch_size=args.ge_batch_size,
                            ge_steps=args.ge_steps, ge_len_traj=args.ge_len_traj, ge_beta=args.ge_beta)
        # ges = pool.map(run_ge_fn, mazes_train)
        ges = list(tqdm(pool.imap(run_ge_fn, mazes_train), total=len(mazes_train)))

    print('Done running Go-Explore on mazes.')
    print('Saving Go-Explore results...')
    torch.save(ges, 'data/ges.pt')
    print('Done saving Go-Explore results.')
    time_end = time.time()
    print('Time to run+save Go-Explore results: {:.2f} minutes'.format((time_end - time_start) / 60))