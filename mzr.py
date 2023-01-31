import argparse
import copy
import os
import pickle
from distutils.util import strtobool
from functools import partial

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import rearrange
from matplotlib import cm
from torch import nn
from tqdm.auto import tqdm

import bc
import env_utils
import goexplore_discrete


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ImitationExplorer(nn.Module):
    def __init__(self, envs, num_frames=4):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(num_frames, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        return self.get_dist_and_values(x)

    def get_dist_and_values(self, x):
        # action = dist.sample(); log_prob = dist.log_prob(action); entropy = dist.entropy()
        x = self.encoder(x)
        logits, values = self.actor(x), self.critic(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, values[:, 0]

    def act(self, x):
        dist, _ = self.get_dist_and_values(x)
        return dist.sample()

class RandomExplorer(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.n_actions = envs.single_action_space.n

    def forward(self, x):
        return self.get_dist_and_values(x)

    def get_dist_and_values(self, x):
        # action = dist.sample(); log_prob = dist.log_prob(action); entropy = dist.entropy()
        logits = torch.zeros(len(x), self.n_actions, device=x.device)
        values = torch.zeros(len(x), 1, device=x.device)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, values[:, 0]

    def act(self, x):
        dist, _ = self.get_dist_and_values(x)
        return dist.sample()

def viz_count_distribution(ge, env1, beta=-0.5):
    cells = list(ge.cell2node.keys())
    nodes = list(ge.cell2node.values())

    n_seen = torch.as_tensor([ge.cell2n_seen[node.cell] for node in nodes])
    prob = (beta*(n_seen+1).log()).softmax(dim=0)
    n_seen, prob = n_seen.numpy(), prob.numpy()
    node2prob = {node: p for node, p in zip(nodes, prob)}

    unique_roomkeys = set([cell[2:] for cell in cells])
    fullimg = []
    for roomkey in unique_roomkeys:
        nodes_roomkey = [node for cell, node in zip(cells, nodes) if cell[2:] == roomkey]
        _, _, _, _, info = env1.restore_snapshot([nodes_roomkey[0].snapshot])
        obs_ori = info['obs_ori'][0]/255.
        h, w, c = obs_ori.shape

        heatmap_count, heatmap_prob = np.zeros((16,16)), np.zeros((16,16))
        for node in nodes_roomkey:
            y, x = node.cell[0:2]
            heatmap_count[y, x] = ge.cell2n_seen[node.cell]
            heatmap_prob[y, x] = node2prob[node]
        heatmap_count = heatmap_count/n_seen.max()
        heatmap_prob = heatmap_prob/prob.max()
        heatmap_count = cv2.resize(heatmap_count, (w, h), interpolation=cv2.INTER_NEAREST)
        heatmap_prob = cv2.resize(heatmap_prob, (w, h), interpolation=cv2.INTER_NEAREST)
        img_count = obs_ori.copy(); img_count[:, :, 0] = heatmap_count
        img_prob = obs_ori.copy(); img_prob[:, :, 0] = heatmap_prob
        fullimg.append((img_count, img_prob))
    fullimg = np.array(fullimg)
    fullimg = rearrange(fullimg, 'n two h w c -> (two h) (n w) c')
    plt.figure(figsize=(5*len(unique_roomkeys), 5*2))
    plt.imshow(fullimg)
    plt.ylabel('prob selection | count')
    plt.xlabel('unique roomkeys')
    return plt.gcf()

def viz_ge_outliers(ge, env1):
    cells = list(ge.cell2node.keys())
    nodes = [ge.cell2node[cell] for cell in cells]
    n_seen = [ge.cell2n_seen[cell] for cell in cells]

    plt.figure(figsize=(30, 10))
    nodes = [nodes[j] for j in np.argsort(n_seen)[:10]] + [nodes[j] for j in np.argsort(n_seen)[-10:]]
    for i, node in enumerate(nodes):
        _, _, _, _, info = env1.restore_snapshot([node.snapshot])
        plt.subplot(2, 10, i+1)
        plt.imshow(info['obs_ori'][0])
        plt.title(f'Visit Count: {ge.cell2n_seen[node.cell]}')
    return plt.gcf()

def viz_explorer_behavior(ge, env, agent=None, nodes_start=None, n_trajs=16, n_trajs_video=8, max_traj_len=100, tqdm=None):
    n_envs = env.num_envs
    assert n_trajs % n_envs == 0
    assert n_trajs_video % n_envs == 0

    if agent is None:
        agent = RandomExplorer(env)

    cells = np.empty((n_trajs, max_traj_len), dtype=object)
    terminateds = np.zeros((n_trajs, max_traj_len), dtype=bool)
    obs_oris = np.empty((n_trajs_video, max_traj_len), dtype=object)

    pbar = torch.arange(n_trajs).split(n_envs)
    if tqdm is not None:
        pbar = tqdm(pbar, leave=False)
    for i_batch, idxs_traj in enumerate(pbar):
        if nodes_start is None:
            obs, info = env.reset()
        else:
            snapshots = [nodes_start[i].snapshot for i in idxs_traj]
            obs, reward, terminated, truncated, info = env.restore_snapshot(snapshots)

        for i_trans in range(max_traj_len):
            cells[idxs_traj, i_trans] = info['cell'] # batch assignment
            if i_batch < n_trajs_video // n_envs:
                obs_oris[idxs_traj, i_trans] = info['obs_ori'] # batch assignment

            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            terminateds[idxs_traj, i_trans] = terminated.cpu().numpy() # batch assignment
    
    def cells2score(cells):
        return len(cells)
    def cells2score(cells):
        return [1/ge.cell2n_seen[cell] for cell in cells]

    # unique_cells_global = np.unique(cells)
    # traj_lens = (~terminateds).sum(axis=1)
    # print(f'# of Unique Cells Visited Ever: {len(unique_cells_global): 05d}')
    # print(f'Avg Traj Len: {np.mean(traj_lens):07.2f} +- {np.std(traj_lens):07.2f}')

    unique_cells_vs_traj_len = np.zeros((n_trajs, max_traj_len), dtype=int)
    for i_traj in range(n_trajs):
        for i_trans in range(max_traj_len):
            unique_cells_vs_traj_len[i_traj, i_trans] = len(np.unique(cells[i_traj, :i_trans]))

    unique_cells_vs_n_trajs = np.zeros((n_trajs), dtype=int)
    for i_traj in range(n_trajs):
        unique_cells_vs_n_trajs[i_traj] = len(np.unique(cells[:i_traj, :]))

    fig = plt.figure(figsize=(25, 5))
    plt.subplot(131)
    plt.plot((~terminateds).mean(axis=0))
    plt.title('Chances of Dying in a Single Trajectory')
    plt.ylabel('P(alive)'); plt.xlabel('Time-Step in Trajectory')

    plt.subplot(132)
    mean = unique_cells_vs_traj_len.mean(axis=0)
    max = unique_cells_vs_traj_len.max(axis=0)
    min = unique_cells_vs_traj_len.min(axis=0)
    std = unique_cells_vs_traj_len.std(axis=0)
    plt.plot(mean, label='mean')
    plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2, label='std')
    plt.plot(max, label='max')
    plt.plot(min, label='min')
    plt.title('Unique Cells Visited in a Single Trajectory')
    plt.ylabel('Unique Cells Visited'); plt.xlabel('Time-Step in Trajectory')
    plt.legend()

    plt.subplot(133)
    plt.plot(unique_cells_vs_n_trajs)
    plt.title('Unique Cells Visited in All Trajectories')
    plt.ylabel('Unique Cells Visited'); plt.xlabel('Trajectories Completed')
    video = np.array(obs_oris.tolist())
    return fig, video

def create_bc_dataset(ge, env1, n_nodes, beta=-0.5, tqdm=None):
    nodes_dataset = ge.select_nodes(n_nodes, beta=beta, condition=lambda node: node!=ge.node_root)
    x, y = [], []
    pbar = nodes_dataset
    if tqdm is not None:
        pbar = tqdm(pbar, leave=False)
    for node in pbar:
        action_history = node.snapshot['ActionHistory']['action_history']
        action_history.append(node.snapshot['action'])
        obs, _ = env1.reset()
        for action in action_history:
            x.append(obs)
            y.append(action)
            obs, _, _, _, _ = env1.step([action])
    x, y = torch.cat(x, dim=0), torch.as_tensor(y)
    return x, y

def make_single_env(frame_stack=1):
    env = gym.make('ALE/MontezumaRevenge-v5', frameskip=4, repeat_action_probability=0.0)
    env = env_utils.EasierMRActionSpace(env)
    env = env_utils.StoreObsInfo(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = env_utils.ObservationDivide(env, 255.)
    env = env_utils.MRDomainCellInfo(env)
    env = env_utils.AtariOneLife(env)
    env = gym.wrappers.FrameStack(env, frame_stack)
    env = env_utils.ActionHistory(env)
    env = env_utils.CumulativeReward(env)
    env = env_utils.ALEState(env)
    class2keys = dict(MRDomainCellInfo=None, ALEState=['ale_state'], ActionHistory=None, CumulativeReward=None,
                      AtariOneLife=None, FrameStack=['frames'])
    env = env_utils.DeterministicRestoreReset(env, class2keys)
    # env = env_utils.DeterministicReplayReset(env)
    return env

def make_env(n_envs, frame_stack=1, auto_reset=False, device=None):
    make_env_fn = partial(make_single_env, frame_stack=frame_stack)
    env = env_utils.RestorableSyncVectorEnv([make_env_fn for i in range(n_envs)], auto_reset=auto_reset)
    # env = gym.wrappers.VectorListInfo(env)
    env = env_utils.ToTensor(env, device=device, dtype=torch.float32)
    return env


parser = argparse.ArgumentParser()
# general parameters
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default=None)

# viz parameters
parser.add_argument("--freq_viz", type=int, default=10)
parser.add_argument("--freq_save", type=int, default=None)

# algorithm parameters
parser.add_argument("--n_steps", type=int, default=50)
parser.add_argument("--n_envs", type=int, default=8)
parser.add_argument("--len_traj", type=int, default=100)
parser.add_argument("--beta", type=float, default=-1.0)

parser.add_argument("--freq_learn", type=int, default=None)
parser.add_argument("--n_nodes_dataset", type=int, default=30)
parser.add_argument("--beta_dataset", type=float, default=-0.5)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--n_steps_learn", type=int, default=20)
parser.add_argument("--coef_entropy", type=float, default=1e-2)
parser.add_argument("--lr", type=float, default=1e-3)

parser.add_argument("--frame_stack", type=int, default=4)

# parser.add_argument("--explorer", type=str, default=None)
# parser.add_argument("--learn_method", type=str, default='none',
#                     help='can be none|bc_elite|bc_contrast')
# parser.add_argument("--reward", type=str, default='log_n_seen')
# parser.add_argument("--n_nodes_select", type=int, default=500)
# parser.add_argument("--n_learn_updates", type=int, default=30)

def main(args):
    print(f'Starting run with args: {args}')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.track:
        run = wandb.init(config=args, name=args.name, save_code=True)

    env = make_env(args.n_envs, args.frame_stack, device=args.device)
    env1 = make_env(1, args.frame_stack, device=args.device)
    env_viz = make_env(8, args.frame_stack, device=args.device)

    ge = goexplore_discrete.GoExplore(env)

    if args.freq_learn is not None:
        agent = ImitationExplorer(env, args.frame_stack)
    else:
        agent = RandomExplorer(env)
    agent = agent.to(args.device)

    pbar = tqdm(range(args.n_steps))
    for i_step in pbar:
        data = {}
        nodes = ge.select_nodes(env.num_envs, beta=args.beta)
        ge.explore_from(nodes, args.len_traj, agent)

        if args.freq_learn is not None and i_step>0 and i_step%args.freq_learn==0:
            x_train, y_train = create_bc_dataset(ge, env1, args.n_nodes_dataset,
                                                 beta=args.beta_dataset, tqdm=tqdm)
            bc.train_bc_agent(agent, x_train, y_train, batch_size=args.batch_size,
                              n_steps=args.n_steps_learn, lr=args.lr,
                              coef_entropy=args.coef_entropy,
                              device=args.device, tqdm=tqdm, callback_fn=None)

        # if i_step<30 or i_step%10==0:
        data['unique_cells']    = len(ge.cell2node)
        data['unique_xys']      = len(set([cell[0:2] for cell in ge.cell2node]))
        data['unique_rooms']    = len(set([cell[2:4] for cell in ge.cell2node]))
        data['unique_keys']     = len(set([cell[4: ] for cell in ge.cell2node]))
        data['unique_roomkeys'] = len(set([cell[2: ] for cell in ge.cell2node]))

        if args.track:
            if i_step%args.freq_viz==0:
                nodes_start = None
                fig, video = viz_explorer_behavior(ge, env_viz, agent, nodes_start, n_trajs=16, n_trajs_video=8, max_traj_len=50, tqdm=None)
                video = rearrange(video, 't h w c -> t c h w')
                data['explorer analysis'] = fig
                data['explorer video'] = wandb.Video(video, fps=15, format='gif')

                data['outliers'] = viz_ge_outliers(ge, env1)
                data['count dist'] = viz_count_distribution(ge, env1, beta=args.beta)
                data['histogram of n_seen'] = wandb.Histogram(list(ge.cell2n_seen.values()))
                data['histogram of len_traj'] = wandb.Histogram([node.len_traj for node in ge.cell2node.values()])
            wandb.log(data)
            plt.close('all')
            if args.freq_save is not None and i_step>0 and i_step%args.freq_save==0:
                os.makedirs(f'data/{args.name}', exist_ok=True)
                torch.save(ge, f'data/{args.name}/ge.pt')

        pbar.set_postfix({k: v for k, v in data.items() if isinstance(v, int) or isinstance(v, float)})
        if args.track:
            wandb.log(data)

    if args.track:
        run.finish()
    return locals()


if __name__=='__main__':
    args = parser.parse_args()
    main(args)





