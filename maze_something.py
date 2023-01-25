
from functools import partial

import cv2
import matplotlib.pyplot as plt
import maze
import numpy as np
import torch
import wandb
from torch import nn
from tqdm.auto import tqdm

import env_utils
import goexplore_discrete
import maze_run
from maze_env import MazeEnv
from maze_run import make_env, make_single_env
from mzr_old import layer_init


class RandomExplorer(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.n_actions = envs.single_action_space.n
    def forward(self, x):
        return self.get_dist_and_values(x)
    def get_dist_and_values(self, x):
        logits = torch.zeros(x.shape[0], self.n_actions, device=x.device)
        values = torch.zeros(x.shape[0], 1, device=x.device)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, values[:, 0]
    def act(self, x):
        dist, _ = self.get_dist_and_values(x)
        return dist.sample()

class ImitationExplorer(nn.Module):
    def __init__(self, envs, num_frames=4):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(num_frames, 16, 3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 3 * 3, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
    def forward(self, x):
        return self.get_dist_and_values(x)

    def get_dist_and_values(self, x):
        x = self.encoder(x)
        logits, values = self.actor(x), self.critic(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, values[:, 0]

    def act(self, x):
        dist, _ = self.get_dist_and_values(x)
        return dist.sample()


def train_bc_agent(agent, x_train, y_train, batch_size=32, n_batches=10, lr=1e-3, coef_entropy=0.0, device=None, tqdm=None, wandb=None):
    agent = agent.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    pbar = range(n_batches)
    if tqdm is not None:
        pbar = tqdm(pbar)
    for i_batch in pbar:
        idxs_batch = torch.randperm(len(x_train))[:batch_size]
        x_batch, y_batch = x_train[idxs_batch].float().to(device), y_train[idxs_batch].long().to(device)
        dist, values = agent.get_dist_and_values(x_batch)

        loss_bc = loss_fn(dist.logits, y_batch).mean()
        loss_entropy = dist.entropy().mean()
        loss = loss_bc - coef_entropy * loss_entropy
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix(loss_bc=loss_bc.item(), entropy=loss_entropy.item())
        if wandb:
            wandb.log({'loss_bc': loss_bc.item(), 'entropy': loss_entropy.item()})

def create_bc_dataset(ges, n_nodes=10, n_samples_per_node=10, beta=-2.0):
    x, y = [], []
    for ge in tqdm(ges):
        env = ge.env.envs[0]

        nodes = ge.select_nodes(n_nodes, beta=beta, condition=lambda node: len(node.snapshot)>n_samples_per_node)
        for node in nodes:
            x_node, y_node = [], []
            obs, info = env.reset()
            for action in node.snapshot:
                x_node.append(obs)
                y_node.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
            idx_traj = np.random.randint(0, len(x_node), size=n_samples_per_node)
            x.extend([x_node[i] for i in idx_traj])
            y.extend([y_node[i] for i in idx_traj])
    x, y = np.stack(x), np.asarray(y)
    x, y = torch.as_tensor(x).float(), torch.as_tensor(y).long()
    return x, y

def get_state_coverage(mazes_test, agent=None):
    cells = [set() for _ in mazes_test]
    statecov = [[] for _ in mazes_test]
    for i_maze, maze in enumerate(mazes_test):
        env = maze_run.make_env(1, maze=maze, obs_size=5, frame_stack=4)
        obs, info = env.reset()
        for i_trans in range(5000):
            cells[i_maze].add(info['cell'][0])
            statecov[i_maze].append(len(cells[i_maze]))
            obs, reward, terminated, truncated, info = env.step(agent.act(obs))
    statecov = np.array(statecov)
    return statecov

def get_video(agent, maze):
    video = []
    env = maze_run.make_env(1, maze=maze, obs_size=5, frame_stack=4)
    maze_full = env.envs[0].maze
    obs, info = env.reset()
    for i_trans in range(500):
        y, x = info['cell'][0]
        left = obs[0, -1].cpu().numpy()
        left = (left+1).clip(0, 1)*255
        right = ((maze_full.copy()+1)*255).astype(np.uint8)
        left = np.stack((left,)*3, axis=-1)
        left[len(left)//2, len(left)//2] = [255, 0, 0]
        left = cv2.resize(left, right.shape, interpolation=cv2.INTER_NEAREST)
        right = np.stack((right,)*3, axis=-1)
        right[y, x] = [255, 0, 0]
        video.append(np.hstack([left, right]))
        obs, reward, terminated, truncated, info = env.step(agent.act(obs))
    video = np.stack(video)
    return video

def viz_ge_node_selection(ges):
    fig = plt.figure(figsize=(40, 5))
    for i_plt, ge in enumerate(ges[:10]):
        grid = ge.env.envs[0].maze.copy()
        cells = list(ge.cell2n_seen.keys())
        n_seen = list(ge.cell2n_seen.values())
        for i in range(len(cells)):
            y, x = cells[i]
            grid[y, x] = n_seen[i]
        plt.subplot(1, 10, i_plt+1)
        plt.imshow(grid)
        nodes = ge.select_nodes(100, beta=-2.0, condition=lambda node: len(node.snapshot)>10)
        # nodes = ge.select_nodes(100, beta=-0.5, condition=lambda node: len(node.snapshot)>10)
        cells = np.array([node.cell for node in nodes])
        plt.scatter(cells[:, 1], cells[:, 0], marker='x', c='r', s=1.)
    return fig

def viz_statecov(statecov_random, statecov_agent):
    fig = plt.figure(figsize=(10, 10))
    mean = statecov_random.mean(axis=0)
    std = statecov_random.std(axis=0)
    plt.plot(mean, label='Random Agent')
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
    mean = statecov_agent.mean(axis=0)
    std = statecov_agent.std(axis=0)
    plt.plot(mean, label='Exploration-Distilled Agent')
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
    plt.title('State Coverage vs Environment Steps')
    plt.ylabel('Number of Unique States Visited')
    plt.xlabel('Number of Environment Steps')
    plt.legend()
    return fig

import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()
# general parameters
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default=None)
# viz parameters
# parser.add_argument("--freq_viz", type=int, default=100)
# parser.add_argument("--freq_save", type=int, default=100)
# algorithm parameters
parser.add_argument("--n_batches", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--coef_entropy", type=float, default=0.0)

def main(args):
    if args.track:
        wandb.init()
    ges = torch.load('data/ges.pt')

    if args.track:
        wandb.log({'viz_ge': viz_ge_node_selection(ges)})
        plt.close('all')

    print(f'Creating BC dataset from {len(ges)} Go-Explore runs.')
    x_train, y_train = create_bc_dataset(ges, 100, 10)
    print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')

    agent_random = RandomExplorer(ges[0].env)
    agent = ImitationExplorer(ges[0].env)

    print('Training agent with BC')
    train_bc_agent(agent, x_train, y_train,
                   batch_size=args.batch_size, n_batches=args.n_batches,
                   lr=args.lr, coef_entropy=args.coef_entropy,
                   device=args.device, tqdm=tqdm, wandb=wandb if args.track else None)
    agent = agent.to('cpu')

    print('Loading test mazes')
    mazes_test = torch.load('data/mazes_test.pt')

    # maze = mazes_test[0]
    # env = maze_run.make_env(1, maze=maze, obs_size=obs_size, frame_stack=frame_stack
    # TODO: run go-explore for mazes_test and evaluate bc loss on that dataset

    print('Evaluating State Coverages')
    statecov_random = get_state_coverage(mazes_test, agent_random) 
    statecov_agent = get_state_coverage(mazes_test, agent) 
    print('Plotting State Coverages')
    if args.track:
        wandb.log({'viz_state_coverage': wandb.Image(viz_statecov(statecov_random, statecov_agent))})
        plt.close('all')

    print('Plotting Videos')
    if args.track:
        mazes = np.random.choice(len(mazes_test), min(25, len(mazes_test)), replace=False)
        mazes = [mazes_test[i] for i in mazes]
        for maze in mazes:
            video_random = get_video(agent_random, maze)
            video_agent = get_video(agent, maze)
            wandb.log({
                'video_random': wandb.Video(video_random.transpose(0, 3, 1, 2).astype(np.uint8), fps=4),
                'video_agent': wandb.Video(video_agent.transpose(0, 3, 1, 2).astype(np.uint8), fps=4)
            })


if __name__=='__main__':
    args = parser.parse_args()
    main(args)