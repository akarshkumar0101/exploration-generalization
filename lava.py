import argparse
from distutils.util import strtobool

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


class LavaGrid():
    def __init__(self, size_grid=100, obs_size=9, p_lava=0.15, n_envs=10, dead_screen=True):
        super().__init__()
        assert obs_size % 2==1
        self.size_grid = size_grid
        self.map = torch.rand(self.size_grid, self.size_grid) < p_lava
        self.k = k = obs_size//2
        self.map[k:k+3, k:k+3] = False
        self.map[:k] = True; self.map[:, :k] = True
        self.map[-k:] = True; self.map[:, -k:] = True
        self.n_envs = n_envs
        self.dead_screen = dead_screen

        self.action2vec = torch.tensor([[ 1,  0],
                                        [ 0,  1],
                                        [-1,  0],
                                        [ 0, -1]])
        self.reset()
        
        self.observation_space = type('', (), {})()
        self.observation_space.sample = lambda : torch.rand((self.n_envs, 2*k+1, 2*k+1), device=self.map.device)
        self.observation_space.shape = (2*k+1, 2*k+1)
        self.action_space = type('', (), {})()
        self.action_space.sample = lambda : torch.randint(0, len(self.action2vec), size=(self.n_envs, ), dtype=torch.long, device=self.map.device)
        self.action_space.n = len(self.action2vec)
        
    def to(self, *args, **kwargs):
        self.map = self.map.to(*args, **kwargs)
        self.action2vec = self.action2vec.to(*args, **kwargs)
        return self

    def reset(self, snapshot=None):
        if snapshot is None:
            self.snapshot = torch.full((self.n_envs, 2), self.k, dtype=torch.long, device=self.map.device)
        else:
            if isinstance(snapshot, list):
                snapshot = torch.stack(snapshot)
            self.snapshot = snapshot.to(self.map.device)
        obs, done = self.calc_obs_done()
        reward = torch.zeros(self.n_envs, device=self.map.device)
        info = None # [{} for _ in range(self.n_envs)]
        return self.snapshot, obs, reward, done, info
    
    def step(self, action):
        action = self.action2vec[action]
        done = self.map[self.snapshot[:, 0], self.snapshot[:, 1]]
        self.snapshot = torch.where(done[:, None], self.snapshot, self.snapshot + action)
        self.snapshot = torch.clamp(self.snapshot, min=self.k, max=self.size_grid-self.k-1)
        
        obs, done = self.calc_obs_done()
        reward = torch.zeros(self.n_envs, device=self.map.device)
        done = self.map[self.snapshot[:, 0], self.snapshot[:, 1]]
        info = None # [{} for _ in range(self.n_envs)]
        return self.snapshot, obs, reward, done, info
    
    def calc_obs_done(self, snapshot=None):
        snapshot = self.snapshot if snapshot is None else snapshot
        obs = torch.stack([self.map[x-self.k: x+self.k+1, y-self.k: y+self.k+1] for (x, y) in snapshot]).float()
        done = self.map[snapshot[:, 0], snapshot[:, 1]]
        if self.dead_screen:
            obs[done] = 1.
        return obs, done
    
    def to_latent(self, snapshot, obs):
        done = self.map[snapshot[:, 0], snapshot[:, 1]]
        return [(tuple(a.tolist()) if not d else (-1, -1)) for a, d in zip(snapshot, done)]

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
            nn.Flatten(-2, -1),
            layer_init(nn.Linear(self.n_inputs, 20)),
            nn.Tanh(),
            layer_init(nn.Linear(20, 20)),
            nn.Tanh(),
            layer_init(nn.Linear(20, 20)),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(20, 20)),
            nn.Tanh(),
            layer_init(nn.Linear(20, 1)),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(20, 20)),
            nn.Tanh(),
            layer_init(nn.Linear(20, self.n_outputs)),
        )
        self.force_random = force_random
        
    def get_logits_values(self, x):
        x = self.encoder(x)
        logits, values = self.actor(x), self.critic(x)
        if self.force_random:
            logits, values = torch.zeros_like(logits), torch.zeros_like(values)
        return logits, values
    
    def get_action_and_value(self, x, action=None):
        logits, values = self.get_logits_values(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values
    
def plot_ge(ge):
    cells = [cell for cell in ge.cell2node.keys() if cell!=(-1, -1)]
    snapshot = torch.stack([ge.cell2node[cell].snapshot for cell in cells])
    n_seen = torch.tensor([ge.cell2n_seen[cell] for cell in cells]).float()
    
    cols = -n_seen.log()
    
    cmap = matplotlib.cm.get_cmap()
    norm = matplotlib.colors.Normalize(vmin=cols.min(), vmax=cols.max())
    cols_norm = (cols-cols.min())/(cols.max()-cols.min())
    cols = cmap(cols_norm.numpy())
    
    # img = np.ones((*ge.env.map.shape, 3))
    img = 1-ge.env.map.cpu().float().repeat(3, 1, 1).permute(1, 2, 0).numpy()
    a = ge.env.map.cpu().float()
    img[:, :, 0][a>0] = .5*np.ones_like(a)[a>0]
    img[snapshot[:, 0], snapshot[:, 1]] = cols[:, :3]
    
    plt.imshow(img)
    # plt.colorbar(cmap)
    plt.gcf().colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca())
    return plt.gcf()
    
def viz_exploration_strategy(args, explorer):
    fig, axs = plt.subplots(3, 3*2, figsize=(10, 4))
    plt.suptitle('Observation vs pi(a|obs)')
    i = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            obs = torch.zeros(args.obs_size, args.obs_size).to(args.device)
            if not (dx==0 and dy==0):
                obs[dx+args.obs_size//2, dy+args.obs_size//2] = 1.
            logits, _ = explorer.get_logits_values(obs[None])
            plt.sca(axs.flatten()[i*2+0])
            plt.imshow(obs.cpu().numpy())
            plt.sca(axs.flatten()[i*2+1])
            plt.bar(torch.arange(logits.shape[-1]), logits[0].softmax(dim=-1).tolist())
            # plt.ylim(0, 1)
            plt.tight_layout()
            i += 1
    return plt.gcf()

def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.track:
        run = wandb.init(
            config=args,
            name=args.name,
            save_code=True)
    
    env = LavaGrid(obs_size=args.obs_size, n_envs=args.n_envs, dead_screen=False).to(args.device)
    snapshot, obs, reward, done, info = env.reset()
    explorer = ImitationExplorer(env, force_random=(args.learn_method=='none')).to(args.device)
    opt = torch.optim.Adam(explorer.parameters(), lr=args.lr)
    ge = GoExplore(env, explorer, args.device)

    pbar = tqdm(range(args.n_steps))
    for i_step in pbar:
        data = {}
        
        nodes = ge.select_nodes(args.n_envs)
        ge.explore_from(nodes, 1, 10)

        if args.learn_method!='none' and i_step>0 and i_step%args.freq_learn==0:
            if args.learn_method=='bc_elite':
                losses, entropies, logits_std = bc.train_bc_elite(ge, explorer, opt, args.n_nodes_select, args.n_learn_updates,
                                                                  args.batch_size, args.coef_entropy, args.device)
            elif args.learn_method=='bc_contrast':
                losses, entropies, logits_std = bc.train_contrastive_bc(ge, explorer, opt, args.n_learn_updates, 
                                                                        args.batch_size, args.coef_entropy, args.device)
            data.update(loss_start=losses[0].item(), loss_end=losses[-1].item())

        n_dead = torch.stack([node.done for node in ge]).sum().item()
        n_seen_max = torch.tensor(list(ge.cell2n_seen.values())).max().item()
        data.update(dict(n_nodes=len(ge), n_cells=len(ge.cell2node), n_dead=n_dead, n_seen_max=n_seen_max))
        pbar.set_postfix(data)
        if args.track:
            if i_step%args.freq_viz==0:
                data['coverage'] = plot_ge(ge)
                data['exploration_strategy'] = viz_exploration_strategy(args, explorer)
            wandb.log(data)
            plt.close()
        
    if args.track:
        run.finish()
    return locals()

parser = argparse.ArgumentParser()
# general parameters
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--freq_viz", type=int, default=100)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--n_steps", type=int, default=1000)
parser.add_argument("--n_envs", type=int, default=10)

# learning parameters
parser.add_argument("--learn_method", type=str, default='none',
                    help='can be none|bc_elite|bc_contrast')
parser.add_argument("--freq_learn", type=int, default=50)
# parser.add_argument("--reward", type=str, default='log_n_seen')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n_nodes_select", type=int, default=500)
parser.add_argument("--n_learn_updates", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--coef_entropy", type=float, default=1e-2)

# lava paremeters
parser.add_argument("--obs_size", type=int, default=3)

def main():
    args = parser.parse_args()
    print(args)
    run(args)

if __name__=='__main__':
    main()


