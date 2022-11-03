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
parser.add_argument('--n_steps', type=int, default=100) # n policy updates
parser.add_argument('--n_trajs', type=int, default=100) # n trajectories in update
parser.add_argument('--len_traj', type=int, default=10) # len of trajectory

parser.add_argument('--n_epochs_policy', type=int, default=4)
parser.add_argument('--batch_size_policy', type=int, default=512)

parser.add_argument('--returner', type=str, default='random')
parser.add_argument('--explorer', type=str, default='policy')
parser.add_argument('--latent', type=str, default='grid')

parser.add_argument('--track', type=bool, action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--n_viz_steps', type=int, default=10)

class MyMiniGrid():
    def __init__(self, limit=100):
        super().__init__()
        self.limit = limit

        self.action2vec = {
            0: torch.tensor([1, 0]),
            1: torch.tensor([0, 1]),
            2: torch.tensor([-1, 0]),
            3: torch.tensor([0, -1]),
        }

        self.reset()
    def calc_obs(self, state=None):
        if state is None:
            state = self.state
        return state/self.limit*2
        
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.item()
        self.state = torch.clamp(self.state + self.action2vec[action], min=-self.limit, max=self.limit)
        obs, reward, done, info = self.calc_obs(), 0, False, {}
        return self.state, obs, reward, done, info
    
    def reset(self, state=None):
        if state is None:
            state = torch.zeros(2, dtype=torch.int32) - self.limit
        self.state = state
        obs, reward, done, info = self.calc_obs(), 0, False, {}
        return self.state, obs, 0, False, info
    
def main():
    args = parser.parse_args()
    print(args)

    if args.track:
        exp_name = args.exp_name if args.exp_name else f'{args.returner}_{args.explorer}_{args.latent}'
        wandb.init(project='goexplore', name=exp_name, config=args)

    env = MyMiniGrid()

    if args.returner == 'random':
        returner = returners.RandomNodeSelector()
    elif args.returner == 'gaussian':
        returner = returners.GaussianNodeSelector()
    if args.explorer == 'random':
        explorer = explorers.RandomExplorer()
    elif args.explorer == 'policy':
        explorer = explorers.PolicyExplorer()

    state2latent = state2latents.MinigridState2Latent()

    ge = goexplore.GoExplore(env, returner=returner, explorer=explorer, state2latent=state2latent)
    for i_step in tqdm(range(args.n_steps)):
        ge.step(n_trajs=args.n_trajs, len_traj=args.len_traj, n_epochs_policy=args.n_epochs_policy, batch_size_policy=args.batch_size_policy)

        data_wandb = {'n_nodes': len(ge.archive.nodes)}
        if args.track and i_step%(args.n_steps//args.n_viz_steps)==0:
            plt.figure(figsize=(10, 5))
            plt.subplot(121); plt.title('Nodes Colored by selection_prob')
            x, c = ge.latents.numpy(), ge.returner.score_nodes(ge.latents).numpy()
            plt.scatter(*x[-1000:].T, c=c[-1000:], cmap='viridis')
            for node in ge.archive.nodes[-1000:]:
                if node.parent is not None:
                    x = torch.stack([node.parent.latent, node.latent])
                    plt.plot(*x.numpy().T, c='r', linewidth=.2)
            plt.colorbar()

            plt.subplot(122); plt.title('Nodes Colored Productivity')
            productivities = explorers.compute_productivities(ge.archive.nodes, alpha=0.1, child_aggregate='mean')
            productivities = torch.tensor([productivities[node] for node in ge.archive.nodes]).float()
            x, c = ge.latents.numpy(), productivities.numpy()
            plt.scatter(*x[-1000:].T, c=c[-1000:], cmap='viridis')
            for node in ge.archive.nodes[-1000:]:
                if node.parent is not None:
                    x = torch.stack([node.parent.latent, node.latent])
                    plt.plot(*x.numpy().T, c='r', linewidth=.2)
            plt.colorbar()
            plt.tight_layout()

            if args.track:
                data_wandb.update({'viz': wandb.Image(plt)})

            plt.close()
        
        latents = ge.latents[torch.randperm(len(ge.latents))[:1000]]
        _, _, logits = ge.explorer.get_action(None, None, latents)
        probs = logits.softmax(dim=-1).mean(dim=0).detach().numpy()
        data_wandb.update({f'p({i})': p for i, p in enumerate(probs)})

        if args.track:
            wandb.log(data_wandb)


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


