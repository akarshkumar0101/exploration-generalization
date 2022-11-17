import argparse
from distutils.util import strtobool

import torch
from torch import nn
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import gym
from gym import spaces

import wandb

from goexplore_discrete import Node, GoExplore

class GoExploreAtariWrapper(gym.Wrapper):
    def __init__(self, env, dead_screen=True):
        super().__init__(env)
        self.n_lives = 6
        self.dead_screen = dead_screen
        
    def reset(self, snapshot=None):
        if snapshot is None:
            obs, reward, done, info = self.env.reset(), 0, False, {}
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
        obs, reward, done, info = self.env.step(action)
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
        obs = torch.stack([d[1] for d in data])
        reward = torch.stack([d[2] for d in data])
        done = torch.stack([d[3] for d in data])
        info = None
        return snapshot, obs, reward, done, info
    
    def step(self, action):
        data = [env.step(a) for env, a in zip(self.envs, action.tolist())]
        snapshot = [d[0] for d in data]
        obs = torch.stack([d[1] for d in data])
        reward = torch.stack([d[2] for d in data])
        done = torch.stack([d[3] for d in data])
        info = None
        return snapshot, obs, reward, done, info
    
    def to_latent(snapshot, obs):
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


    
def step_policy(ge, explorer, opt, calc_prod, n_steps, batch_size=100, coef_entropy=1e-1, viz=False, device=None):
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
        nodes = ge.select_nodes(args.n_envs)
        ge.explore_from(nodes, 1, 10)

        if args.freq_learn is not None and args.freq_learn>0 and i_step%args.freq_learn==0:
            step_policy(ge, explorer, opt, calc_prods_novelty, args.n_updates_learn, 
                        batch_size=args.batch_size, coef_entropy=args.coef_entropy, viz=False, device=args.device)

            
        n_cells = len(ge.cell2node)
        n_dead = torch.stack([node.done for node in ge]).sum().item()
        n_seen_max = torch.tensor(list(ge.cell2n_seen.values())).max().item()
        # done = torch.stack([node.done for node in ge])
        # pbar.set_postfix(cells=len(ge.cell2node), n_seen_max=n_seen.max().item())
        # cellsvtime.append(len(ge.cell2node))
        # deadvtime.append(done.sum().item())
        
        data = dict(n_nodes=len(ge), n_cells=len(ge.cell2node), n_dead=n_dead, n_seen_max=n_seen_max)
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


