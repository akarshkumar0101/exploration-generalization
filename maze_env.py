# -*- coding: utf-8 -*-
# Code from https://github.com/raulorteg/ai-maze-python/blob/master/maze_generator.py
import gymnasium as gym
import maze
import numpy as np


def get_solution_path(maze):
    solution = maze.solution.sum(axis=-1)
    h, w = solution.shape
    start = (1, 1)
    end = (h-2, w-2)
    path = [start]
    while path[-1]!=end:
        y, x = path[-1]
        if (y, x-1) not in path and np.abs(solution[y, x-1]-255)<5:
            path.append((y, x-1))
        elif (y, x+1) not in path and np.abs(solution[y, x+1]-255)<5:
            path.append((y, x+1))
        elif (y-1, x) not in path and np.abs(solution[y-1, x]-255)<5:
            path.append((y-1, x))
        elif (y+1, x) not in path and np.abs(solution[y+1, x]-255)<5:
            path.append((y+1, x))
    return np.array(path)

def generate_maze_data(n_rows=50, n_cols=50, algorithm=maze.Maze.Create.PRIM):
    m = maze.Maze()
    m.create(n_rows, n_cols, algorithm)
    assert m.maze[1,1].mean()>254 # assert start exists
    assert m.maze[-2,-2].mean()>254 # assert end exists
    m.solve((0, 0), (n_rows-1, n_cols-1), maze.Maze.Solve.DEPTH) # idk why the indices are like that
    path = get_solution_path(m)
    return dict(maze=m.maze[:, :, 0]>128, path=path)

class MazeEnv(gym.Env):
    def __init__(self, maze, path, obs_size=5, reward_sparsity=10):
        self.maze_ori = maze[1:-1, 1:-1]
        h, w = self.maze_ori.shape
        self.maze = np.full((h+obs_size*2, w+obs_size*2), fill_value=-1, dtype=np.int8)
        self.maze[obs_size:-obs_size, obs_size:-obs_size] = self.maze_ori-1

        self.path = [(y+obs_size-1, x+obs_size-1) for y, x in path]

        self.obs_size = obs_size

        self.possible_moves = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_size*2+1, obs_size*2+1), dtype=np.int8)
        self.action_space = gym.spaces.Discrete(len(self.possible_moves))

        self.yx = np.array(self.path[0]) # starting position

        self.reward_sparsity = reward_sparsity
        self.reward_map = np.zeros_like(self.maze)
        for y, x in self.path[1::reward_sparsity]:
            self.reward_map[y, x] = 1

    def reset(self, *args, **kwargs):
        self.yx = np.array(self.path[0]) # starting position
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
        reward = self.reward_map[y, x] # get reward
        self.reward_map[y, x] = 0 # remove reward; already got it
        terminated, truncated = (y, x)==self.path[-1], False
        return obs, reward, terminated, truncated, info


