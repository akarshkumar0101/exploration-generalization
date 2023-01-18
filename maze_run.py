import time
from functools import partial
from multiprocessing import Pool

import gymnasium as gym
import numpy as np
import torch
from tqdm.auto import tqdm

import env_utils
import goexplore_discrete
import maze


class MazeEnv(gym.Env):
    def __init__(self, grids, obs_size=5):
        self.obs_size = obs_size
        self.moves = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(obs_size*2+1, obs_size*2+1), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(len(self.moves))
        self.grids = grids

        self.seed(0)

    def seed(self, seed=0):
        self.my_seed = seed

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = self.my_seed
        obs_size = self.obs_size
        height, width = self.grids[0].shape
        self.grid = np.zeros((height+obs_size*2, width+obs_size*2), dtype=np.uint8)
        self.grid[obs_size:-obs_size, obs_size:-obs_size] = self.grids[seed]

        self.yx = np.array([self.obs_size, self.obs_size])
        y, x = self.yx
        obs = self.grid[y-self.obs_size:y+self.obs_size+1, x-self.obs_size:x+self.obs_size+1].copy()
        obs[len(obs)//2, len(obs)//2] = 2 # agent
        info = {'cell': (y, x)}
        return obs, info

    def step(self, action):
        yx = self.yx + self.moves[action]
        if self.grid[yx[0], yx[1]]==1:
            self.yx = yx
        else: # hit a wall
            pass
        y, x = self.yx
        obs = self.grid[y-self.obs_size:y+self.obs_size+1, x-self.obs_size:x+self.obs_size+1].copy()
        obs[len(obs)//2, len(obs)//2] = 2 # agent
        reward = 0
        terminated, truncated = False, False
        info = {'cell': (y, x)}
        return obs, reward, terminated, truncated, info
def make_single_env(grids, frame_stack=1):
    env = MazeEnv(grids, obs_size=5)
    env = env_utils.StoreObsInfo(env)
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = env_utils.ObservationDivide(env, 2.)
    env = gym.wrappers.FrameStack(env, frame_stack)
    env = env_utils.DeterministicReplayReset(env)
    return env

def make_env(grids, n_envs, frame_stack=1, auto_reset=False):
    make_env_fn = partial(make_single_env, grids=grids, frame_stack=frame_stack)
    env = env_utils.RestorableSyncVectorEnv([make_env_fn for i in range(n_envs)], auto_reset=auto_reset)
    env = env_utils.ToTensor(env, device=None, dtype=torch.float32)
    return env

def gen_maze(seed):
    np.random.seed(seed)
    return maze.generate_maze(maze_size, maze_size)

def run_ge(seed, grids, pbar=None):
    env = make_env(grids, ge_batch_size, 4)
    for e in env.envs:
        e.seed(seed)
    ge = goexplore_discrete.GoExplore(env)
    for i in range(ge_steps):
        nodes = ge.select_nodes(ge_batch_size, beta=-2.)
        ge.explore_from(nodes, 15)
    if pbar is not None:
        pbar.update(1)
    return ge

n_procs = 35
n_seeds = 1000
maze_size = 11

ge_batch_size = 100
ge_steps = 10

if __name__ == '__main__':
    with Pool(n_procs) as p:
        time_start = time.time()
        grids = p.map(gen_maze, np.arange(n_seeds))
        grids = np.stack(grids)
        np.save('data/grids.npy', grids)
        print('Grids shape: ', grids.shape)
        time_end = time.time()
        print('Time to generate mazes: ', time_end-time_start)

    grids = np.load('data/grids.npy')
    pbar = tqdm(total=n_seeds)
    run_ge_fn = partial(run_ge, grids=grids, pbar=pbar)
    with Pool(n_procs) as p:
        time_start = time.time()
        ges = p.map(run_ge_fn, np.arange(n_seeds))
        torch.save(ges, 'data/ges.pt')
        time_end = time.time()
        print('Time to run GEs: ', time_end-time_start)