import argparse
from distutils.util import strtobool

import torch
from torch import nn
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from gym import spaces

import wandb


from goexplore_discrete import Node, GoExplore

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
    
    def get_action_and_value(self, x, action=None, ret_logits=False):
        logits, values = self.get_logits_values(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values
    
def calc_prods_dfs(ge, reduce1=np.max, reduce2=np.max, log=True):
    node2prod = {}
    def recurse(node):
        novelty = -ge.cell2n_seen[node.latent]
        if node.children:
            for child in node.children:
                recurse(child)
            prods_children = [node2prod[child] for child in node.children]
            if reduce2 is None:
                prod = reduce1([novelty]+prods_children)
            else:
                prod = reduce1([novelty, reduce2(prods_children)])
        else:
            prod = novelty
        node2prod[node] = prod
    recurse(ge.node_root)
    prod = torch.tensor([node2prod[node] for node in ge]).float()
    return -(-prod).log() if log else prod

def calc_prods_novelty(ge, log=True):
    n_seen = torch.tensor([ge.cell2n_seen[node.latent] for node in ge]).float()
    return -n_seen.log() if log else -n_seen

def normalize(a, devs=None):
    b = a
    if devs is not None:
        # mask = ((a-a.mean())/a.std()).abs()<devs
        mask = ((a-a.median())/a.std()).abs()<devs
        b = a[mask]
    return (a-b.mean())/(a.std()+1e-9)
    
def step_policy(ge, explorer, opt, calc_prod, n_steps, batch_size=100, coef_entropy=1e-1, viz=False, device=None, data=None):
    # list of tuples (snapshot, obs, action, reward, done)
    obs = torch.stack([trans[1] for node in ge for trans in node.traj])
    action = torch.stack([trans[2] for node in ge for trans in node.traj])
    # mask_done = torch.stack([trans[4] for node in ge for trans in node.traj])
    mask_done = torch.stack([node.done for node in ge])
    prod = calc_prod(ge)
    prod_norm = (prod-prod[~mask_done].mean())/(prod[~mask_done].std()+1e-9)
    r = prod_norm[1:]
    
    losses = []
    entropies = []
    logits_list = []
    for i_batch in range(n_steps):
        idx = torch.randperm(len(obs))[:batch_size]
        b_obs, b_action, b_r = obs[idx].to(device), action[idx].to(device), r[idx].to(device)
        # if norm_batch:
            # b_prod = (b_prod-b_prod.mean())/(b_prod.std()+1e-9)
            
        logits, values = explorer.get_logits_values(b_obs)
        # logits_aug = b_r[:, None]*logits
        # logits_aug = (1./b_r[:, None])*logits
        logits_aug = torch.sign(b_r[:, None])*logits
        dist = torch.distributions.Categorical(logits=logits_aug)
        log_prob = dist.log_prob(b_action)
        entropy = dist.entropy()
        
        loss_data = (-log_prob*b_r.abs()).mean()
        loss_entropy = -entropy.mean()
        
        loss = loss_data + coef_entropy*loss_entropy

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        entropies.append(entropy.mean().item())
        logits_list.append(logits.mean().item())
        
        # pbar.set_postfix(loss=loss.item())
    losses = torch.tensor(losses)
    # print(f'Reduced loss from {losses[0].item()} to {losses[-1].item()}')
    data.update(loss_start=losses[0].item(), loss_end=losses[-1].item())

    if viz:
        plt.figure(figsize=(15, 5))
        plt.subplot(131); plt.plot(losses); plt.title('loss vs time')
        plt.subplot(132); plt.plot(entropies); plt.title('entropy vs time')
        plt.subplot(133); plt.plot(logits_list); plt.title('logits mean vs time')
        plt.show()
        
        # plt.plot(logits.softmax(dim=-1).mean(dim=-2).detach().cpu().numpy())
        # plt.hist(logits.argmax(dim=-1).detach().cpu().numpy())
        # plt.ylim(.23, .27)
        # plt.title('avg prob distribution')
        # plt.show()
        
        # plt.scatter(b_action.detach().cpu().numpy(), log_prob.detach().cpu().numpy())
        # plt.show()
        # plt.scatter(b_prod.cpu().numpy(), loss1.detach().cpu().numpy())
        # plt.show()
        
        # for i in range(4):
            # print(f'Action {i}')
            # print(log_probs[batch_actions==i].mean().item())

from matplotlib import cm
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
            plt.bar(torch.arange(4), logits[0].softmax(dim=-1).tolist())
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
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True)
    
    env = LavaGrid(obs_size=args.obs_size, n_envs=args.n_envs, dead_screen=False).to(args.device)
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
            step_policy(ge, explorer, opt, calc_prods_novelty, args.n_updates_learn, 
                        batch_size=args.batch_size, coef_entropy=args.coef_entropy, viz=False, device=args.device, data=data)

            
        n_cells = len(ge.cell2node)
        n_dead = torch.stack([node.done for node in ge]).sum().item()
        n_seen_max = torch.tensor(list(ge.cell2n_seen.values())).max().item()
        # done = torch.stack([node.done for node in ge])
        # pbar.set_postfix(cells=len(ge.cell2node), n_seen_max=n_seen.max().item())
        # cellsvtime.append(len(ge.cell2node))
        # deadvtime.append(done.sum().item())
        
        data.update(dict(n_nodes=len(ge), n_cells=len(ge.cell2node), n_dead=n_dead, n_seen_max=n_seen_max))
        pbar.set_postfix(data)
        if args.track:
            if i_step>0 and i_step%args.freq_viz==0:
                data['coverage'] = plot_ge(ge)
                data['exploration_strategy'] = viz_exploration_strategy(args, explorer)
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
parser.add_argument("--device", type=str, default='cuda:0')

parser.add_argument("--n_steps", type=int, default=1000)
parser.add_argument("--obs_size", type=int, default=3)
parser.add_argument("--n_envs", type=int, default=10)

parser.add_argument("--freq_learn", type=int, default=None)
# parser.add_argument("--learn", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--reward", type=str, default='log_n_seen')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n_updates_learn", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--coef_entropy", type=float, default=1e-2)


def main():
    # args = parser.parse_args()
    args, uargs = parser.parse_known_args()
    print(args)
    run(args)

if __name__=='__main__':
    main()


