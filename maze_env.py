# -*- coding: utf-8 -*-
# Code from https://github.com/raulorteg/ai-maze-python/blob/master/maze_generator.py
"""
Created on Wed Sep  9 00:01:40 2020
@author: Raul Ortega Ochoa
"""
import gymnasium as gym
import maze
import numpy as np


def generate_mazes(n_rows=50, n_cols=50, n_mazes=1000, algorithm=maze.Maze.Create.PRIM, tqdm=None):
    m = maze.Maze()
    mazes = []
    pbar = range(n_mazes)
    if tqdm is not None:
        pbar = tqdm(pbar)
    for _ in pbar:
        m.create(n_rows, n_cols, maze.Maze.Create.PRIM)
        assert m.maze[1,1].mean()>128
        assert m.maze[-2,-2].mean()>128
        mazes.append(m.maze)
    mazes = np.stack(mazes)
    mazes = mazes[:, :, :, 0]>128
    return mazes[:, 1:-1, 1:-1]

class MazeEnv(gym.Env):
    def __init__(self, maze, obs_size=5):
        self.maze_ori = maze
        self.obs_size = obs_size

        self.possible_moves = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_size*2+1, obs_size*2+1), dtype=np.int8)
        self.action_space = gym.spaces.Discrete(len(self.possible_moves))

        h, w = self.maze_ori.shape
        self.maze = np.full((h+obs_size*2, w+obs_size*2), fill_value=-1, dtype=np.int8)
        self.maze[obs_size:-obs_size, obs_size:-obs_size] = self.maze_ori-1

        self.yx = np.array([self.obs_size, self.obs_size]) # starting position

    def reset(self, *args, **kwargs):
        self.yx = np.array([self.obs_size, self.obs_size]) # starting position
        y, x = self.yx
        obs = self.maze[y-self.obs_size:y+self.obs_size+1, x-self.obs_size:x+self.obs_size+1].copy()
        obs[len(obs)//2, len(obs)//2] = 1 # agent
        info = {'cell': (y, x)}
        return obs, info

    def step(self, action):
        yx = self.yx + self.possible_moves[action]
        if self.maze[yx[0], yx[1]]==0: # not wall
            self.yx = yx
        y, x = self.yx
        obs = self.maze[y-self.obs_size:y+self.obs_size+1, x-self.obs_size:x+self.obs_size+1].copy()
        obs[len(obs)//2, len(obs)//2] = 1 # agent
        info = {'cell': (y, x)}
        reward = 0
        terminated, truncated = False, False
        return obs, reward, terminated, truncated, info


