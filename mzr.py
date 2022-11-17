import argparse
from distutils.util import strtobool

import cv2
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from matplotlib import cm
from torch import nn
from tqdm.auto import tqdm

import bc
import wandb
from goexplore_discrete import GoExplore, Node, calc_reward_novelty


class GoExploreAtariWrapper(gym.Wrapper):
    def __init__(self, env, dead_screen=True):
        super().__init__(env)
        self.n_lives = 6
        self.dead_screen = dead_screen
        
    def reset(self, snapshot=None):
        if snapshot is None:
            obs, reward, done, info = self.env.reset()[0], 0, False, {}
            done = self.env.unwrapped.ale.lives() < self.n_lives
            if done:
                obs = np.zeros_like(obs)
            ale_state = self.env.ale.cloneState()
        else:
            (obs, reward, done, info, ale_state) = snapshot
            self.env.ale.restoreState(ale_state)
        snapshot = (obs, reward, done, info, ale_state)
        obs, reward, done = torch.as_tensor(obs)/255., torch.as_tensor(reward), torch.as_tensor(done)
        return snapshot, obs, reward, done, info
        
    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        done = self.env.unwrapped.ale.lives() < self.n_lives
        if done and self.dead_screen:
            obs = np.zeros_like(obs)
        ale_state = self.env.ale.cloneState()
        snapshot = (obs, reward, done, info, ale_state)
        obs, reward, done = torch.as_tensor(obs)/255., torch.as_tensor(reward), torch.as_tensor(done)
        return snapshot, obs, reward, done, info

class MZRevenge():
    def __init__(self, n_envs=10, dead_screen=True, latent_h=11, latent_w=8, latent_d=20):
        super().__init__()
        self.n_envs = n_envs
        self.dead_screen = dead_screen
        self.latent_h, self.latent_w, self.latent_d = latent_h, latent_w, latent_d
        
        self.envs = [self.make_env() for _ in range(self.n_envs)]
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.reset()
        
    def make_env(self):
        env = gym.make('MontezumaRevengeDeterministic-v4', render_mode='rgb_array')
        env = gym.wrappers.ResizeObservation(env, (21*5, 16*5))
        env = gym.wrappers.GrayScaleObservation(env)
        env = GoExploreAtariWrapper(env)
        return env
        
    def to(self, *args, **kwargs):
        return self
        
    def reset(self, snapshot=None):
        if snapshot is None:
            snapshot = [None] * self.n_envs
            
        data = [env.reset(snap) for env, snap in zip(self.envs, snapshot)]
        snapshot = [d[0] for d in data]
        obs = torch.stack([d[1] for d in data])[:, None, :, :]
        reward = torch.stack([d[2] for d in data])
        done = torch.stack([d[3] for d in data])
        info = None
        return snapshot, obs, reward, done, info
    
    def step(self, action):
        data = [env.step(a) for env, a in zip(self.envs, action.tolist())]
        snapshot = [d[0] for d in data]
        obs = torch.stack([d[1] for d in data])[:, None, :, :]
        reward = torch.stack([d[2] for d in data])
        done = torch.stack([d[3] for d in data])
        info = None
        return snapshot, obs, reward, done, info
    
    def to_latent(self, snapshot, obs):
        def to_latent_single(obs):
            # obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, (self.latent_w, self.latent_h), interpolation=cv2.INTER_AREA)
            obs = (obs*self.latent_d).astype(np.uint8)
            return tuple(obs.flatten())
        return [to_latent_single(o) for o in obs.cpu().numpy()]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ImitationExplorer(nn.Module):
    def __init__(self, env, force_random=False):
        super().__init__()
        self.n_inputs = np.prod(env.observation_space.shape)
        self.n_outputs = env.action_space.n
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(1, 10, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(10, 10, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(10, 10, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(10, 10, 3, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(20, 20)),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(20, 20)),
            nn.ReLU(),
            layer_init(nn.Linear(20, 1)),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(20, 20)),
            nn.ReLU(),
            layer_init(nn.Linear(20, self.n_outputs)),
        )
        self.force_random = force_random
        
    def get_logits_values(self, x):
        x = self.encoder(x)
        logits, values = self.actor(x), self.critic(x)
        if self.force_random:
            logits, values = torch.zeros_like(logits), torch.zeros_like(values)
        return logits, values
    
    def get_action_and_value(self, x, action=None, ret_logits=False):
        logits, values = self.get_logits_values(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values

def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.track:
        run = wandb.init(
            config=args,
            name=args.name,
            save_code=True)
    
    env = MZRevenge(n_envs=args.n_envs, dead_screen=True).to(args.device)
    snapshot, obs, reward, done, info = env.reset()
    explorer = ImitationExplorer(env, force_random=(args.freq_learn is None)).to(args.device)
    opt = torch.optim.Adam(explorer.parameters(), lr=args.lr)
    ge = GoExplore(env, explorer, args.device)

    pbar = tqdm(range(args.n_steps))
    for i_step in pbar:
        data = {}
        
        nodes = ge.select_nodes(args.n_envs)
        ge.explore_from(nodes, 1, 10)

        if args.freq_learn is not None and args.freq_learn>0 and i_step%args.freq_learn==0:
            obs = torch.stack([trans[1] for node in ge for trans in node.traj])
            action = torch.stack([trans[2] for node in ge for trans in node.traj])
            mask_done = torch.stack([trans[4] for node in ge for trans in node.traj])
            reward = calc_reward_novelty(ge)
            reward = torch.stack([reward[i] for i, node in enumerate(ge) for trans in node.traj])
            reward = (reward-reward[~mask_done].mean())/(reward[~mask_done].std()+1e-9)

            bc.train_contrastive_bc(explorer, opt, obs, action, reward,
                                    args.n_updates_learn, args.batch_size, 
                                    args.coef_entropy, args.device, data, viz=False)

        n_dead = torch.stack([node.done for node in ge]).sum().item()
        n_seen_max = torch.tensor(list(ge.cell2n_seen.values())).max().item()
        data.update(dict(n_nodes=len(ge), n_cells=len(ge.cell2node), n_dead=n_dead, n_seen_max=n_seen_max))
        pbar.set_postfix(data)
        if args.track:
            # if i_step>0 and i_step%args.freq_viz==0:
                # data['coverage'] = plot_ge(ge)
                # data['exploration_strategy'] = viz_exploration_strategy(args, explorer)
            wandb.log(data)
            plt.close()
        
    if args.track:
        run.finish()
    return locals()
        

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--freq_viz", type=int, default=50)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default=None)

parser.add_argument("--n_steps", type=int, default=1000)
parser.add_argument("--n_envs", type=int, default=10)

parser.add_argument("--freq_learn", type=int, default=None)
parser.add_argument("--reward", type=str, default='log_n_seen')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n_updates_learn", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--coef_entropy", type=float, default=1e-2)


def main():
    args = parser.parse_args()
    print(args)
    run(args)

if __name__=='__main__':
    main()
