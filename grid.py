import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt

import gymnasium as gym

import wandb

import argparse

from tqdm import tqdm


import goexplore
import returners, explorers, state2latents

parser = argparse.ArgumentParser(description='Grid')
parser.add_argument('--n_steps', type=int, default=100)
parser.add_argument('--n_exploration_steps', type=int, default=5)
parser.add_argument('--returner', type=str, default='random')
parser.add_argument('--explorer', type=str, default='random')
parser.add_argument('--latent', type=str, default='obs')

parser.add_argument('--track', type=bool, action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--exp_name', type=str, default='random')
parser.add_argument('--n_viz_steps', type=int, default=10)

class MyMiniGrid():
    def __init__(self):
        super().__init__()
        self.reset()

        self.action2vec = {
            0: torch.tensor([1, 0]),
            1: torch.tensor([0, 1]),
            2: torch.tensor([-1, 0]),
            3: torch.tensor([0, -1]),
        }
        
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.item()
        self.state = torch.clamp(self.state + self.action2vec[action], min=0, max=None)
        obs, reward, done, info = self.state, 0, False, {}
        return self.state, obs, reward, done, info
    
    def reset(self):
        self.state = torch.zeros(2)
        obs, reward, done, info = self.state, 0, False, {}
        return self.state, obs, 0, False, info
    
    def get_current_state_sim(self):
        return self.state
    
    def goto_state_sim(self, state):
        self.state = state


def main():
    args = parser.parse_args()
    print(args)

    if args.track:
        wandb.init(project='goexplore', name=args.exp_name, config=args)

    env = MyMiniGrid()

    if args.returner == 'random':
        returner = returners.RandomNodeSelector()
    elif args.returner == 'gaussian':
        returner = returners.GaussianNodeSelector()
    if args.explorer == 'random':
        explorer = explorers.RandomExplorer()
    elif args.explorer == 'policy':
        explorer = explorers.PolicyExplorer()

    state2latent = state2latents.Obs2Latent()

    ge = goexplore.GoExplore(env, returner=returner, explorer=explorer, state2latent=state2latent, n_exploration_steps=args.n_exploration_steps)
    for i_step in tqdm(range(args.n_steps)):
        ge.step()

        if args.track and i_step%(args.n_steps//args.n_viz_steps)==0:
            plt.figure(figsize=(10, 5))
            plt.subplot(121); plt.title('Nodes Colored by selection_prob')
            x, c = ge.latents.numpy(), ge.returner.score_nodes(ge.latents).numpy()
            plt.scatter(*x.T, c=c, cmap='viridis')
            for node in ge.archive.nodes:
                if node.parent is not None:
                    x = torch.stack([node.parent.latent, node.latent])
                    plt.plot(*x.numpy().T, c='r', linewidth=.2)
            plt.colorbar()

            plt.subplot(122); plt.title('Nodes Colored Productivity')
            productivities = explorers.compute_productivities(ge, alpha=0.1, child_aggregate='mean')
            productivities = torch.tensor([productivities[node] for node in ge.archive.nodes]).float()
            x, c = ge.latents.numpy(), productivities.numpy()
            plt.scatter(*x.T, c=c, cmap='viridis')
            for node in ge.archive.nodes:
                if node.parent is not None:
                    x = torch.stack([node.parent.latent, node.latent])
                    plt.plot(*x.numpy().T, c='r', linewidth=.2)
            plt.colorbar()
            plt.tight_layout()

            wandb.log({'archive': wandb.Image(plt)})
            plt.close()


if __name__=='__main__':
    main()


def get_base_env(env):
    return get_base_env(env.env) if hasattr(env, 'env') else env

# class MyMiniGrid(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env
        
#     def step(self, action):
#         obs, reward, done, _, info = self.env.step(action)
#         state_sim = self.get_current_state_sim()
#         (x, y), d = state_sim
#         obs = torch.tensor([x, y, d])
#         return state_sim, obs, reward, done, info
    
#     def reset(self):
#         obs, info = self.env.reset()
#         state_sim = self.get_current_state_sim()
#         (x, y), d = state_sim
#         obs = torch.tensor([x, y, d])
#         return state_sim, obs, 0, False, info
    
#     def get_current_state_sim(self):
#         return (self.env.agent_pos, self.env.agent_dir)
    
#     def goto_state_sim(self, state):
#         self.reset()
#         p, d = state
#         get_base_env(env).agent_pos = p
#         get_base_env(env).agent_dir = d


