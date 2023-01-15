import argparse
import copy
import pickle
from distutils.util import strtobool
from functools import partial

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from torch import nn
from tqdm.auto import tqdm

import bc
import env_utils
import goexplore_discrete
import wandb


def viz_count_distribution(ge, env0):
    cells = list(ge.cell2node.keys())
    nodes = list(ge.cell2node.values())
    unique_roomkeys = set([cell[2:] for cell in cells])

    plt.figure(figsize=(5*len(unique_roomkeys), 5*2))

    i_subplot = 1
    for roomkey in unique_roomkeys:
        nodes_roomkey = [node for cell, node in zip(cells, nodes) if cell[2:] == roomkey]
        _, _, _, _, info = env0.restore_snapshot(nodes_roomkey[0].snapshot)
        obs_ori = info['obs_ori']/255.
        h, w, c = obs_ori.shape

        heatmap_count = np.zeros((16,16))
        heatmap_prob = np.zeros((16,16))
        for node in nodes_roomkey:
            y, x = node.cell[0:2]
            heatmap_count[y, x] = ge.cell2n_seen[node.cell]
            heatmap_prob[y, x] = 1/np.sqrt(ge.cell2n_seen[node.cell])
        heatmap_count = heatmap_count/heatmap_count.max()
        heatmap_prob = heatmap_prob/heatmap_prob.max()
        heatmap_count = cv2.resize(heatmap_count, (w, h), interpolation=cv2.INTER_AREA)
        heatmap_prob = cv2.resize(heatmap_prob, (w, h), interpolation=cv2.INTER_AREA)

        img = obs_ori.copy()
        img[:, :, 0] = heatmap_count
        plt.subplot(2, len(unique_roomkeys), i_subplot); plt.imshow(img); plt.title('count')
        img = obs_ori.copy()
        img[:, :, 0] = heatmap_prob
        plt.subplot(2, len(unique_roomkeys), i_subplot+len(unique_roomkeys)); plt.imshow(img); plt.title('prob selection')
        i_subplot += 1
    return plt.gcf()

def viz_ge_outliers(ge, env0):
    cells = list(ge.cell2node.keys())
    nodes = [ge.cell2node[cell] for cell in cells]
    n_seen = [ge.cell2n_seen[cell] for cell in cells]

    plt.figure(figsize=(30, 10))
    for i, node in enumerate([nodes[j] for j in np.argsort(n_seen)[:10]]):
        _, _, _, _, info = env0.restore_snapshot(node.snapshot)
        plt.subplot(2, 10, i+1)
        plt.imshow(info['obs_ori'])
        plt.title(f'Visit Count: {ge.cell2n_seen[node.cell]}')
    for i, node in enumerate([nodes[j] for j in np.argsort(n_seen)[-10:]]):
        _, _, _, _, info = env0.restore_snapshot(node.snapshot)
        plt.subplot(2, 10, i+1+10)
        plt.imshow(info['obs_ori'])
        plt.title(f'Visit Count: {ge.cell2n_seen[node.cell]}')
    return plt.gcf()

def make_single_env(frame_stack=1):
    env = gym.make('ALE/MontezumaRevenge-v5', frameskip=4, repeat_action_probability=0.0)
    # env = gym.make('MontezumaRevengeDeterministic-v4')
    env = env_utils.StoreObsInfo(env)
    env = gym.wrappers.ResizeObservation(env, (21*5, 16*5))
    env = gym.wrappers.GrayScaleObservation(env)
    env = env_utils.ObservationDivide(env, 255.)
    env = env_utils.AtariOneLife(env)
    # env = env_utils.ImageCellInfo(env)
    env = env_utils.MRDomainCellInfo(env)
    # env = env_utils.ZeroReward(env)
    # env = env_utils.TerminationReward(env)
    env = env_utils.DecisionTransformerEnv(env, n_obs=frame_stack)
    env = gym.wrappers.FrameStack(env, frame_stack)
    env = env_utils.DeterministicReplayReset(env)
    return env

def make_env(n_envs, frame_stack=1, auto_reset=False):
    make_env_fn = partial(make_single_env, frame_stack=frame_stack)
    env = env_utils.RestorableSyncVectorEnv([make_env_fn for i in range(n_envs)], auto_reset=auto_reset)
    # env = gym.wrappers.VectorListInfo(env)
    env = env_utils.ToTensor(env)
    return env


def main(args):
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.track:
        run = wandb.init(
            config=args,
            name=args.name,
            save_code=True)

    env = make_env(args.n_envs, 1)
    env_single = make_single_env(10)
    # env_single = make_env(1, 1)

    ge = goexplore_discrete.GoExplore(env)

    pbar = tqdm(range(args.n_steps))
    for i_step in pbar:
        data = {}
        strategy = 'inverse_sqrt'# if i%3==0 else ('inverse_abs' if i%3==1 else 'inverse_square')

        nodes = ge.select_nodes(env.num_envs, strategy=strategy)
        ge.explore_from(nodes, args.len_traj)

        if i_step%10==0:
            data['unique_cells']    = len(ge.cell2node)
            data['unique_xys']      = len(set([cell[0:2] for cell in ge.cell2node]))
            data['unique_rooms']    = len(set([cell[2:4] for cell in ge.cell2node]))
            data['unique_keys']     = len(set([cell[4: ] for cell in ge.cell2node]))
            data['unique_roomkeys'] = len(set([cell[2: ] for cell in ge.cell2node]))
            pbar.set_postfix({k: v for k, v in data.items() if isinstance(v, int) or isinstance(v, float)})

        if args.track:
            if i_step%args.freq_viz==0:
                data['outliers'] = viz_ge_outliers(ge, env_single)
                data['count dist'] = viz_count_distribution(ge, env_single)
                data['histogram of n_seen'] = wandb.Histogram(list(ge.cell2n_seen.values()))
                data['histogram of len_traj'] = wandb.Histogram([len(node.snapshot) for node in ge.cell2node.values()])
            wandb.log(data)
            plt.close('all')
            if i_step%args.freq_save==0:
                torch.save(ge, f'results/ge.pt')

        
    if args.track:
        run.finish()
    return locals()

parser = argparse.ArgumentParser()
# general parameters
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default=None)

# viz parameters
parser.add_argument("--freq_viz", type=int, default=100)
parser.add_argument("--freq_save", type=int, default=100)

# algorithm parameters
parser.add_argument("--n_steps", type=int, default=1000)
parser.add_argument("--n_envs", type=int, default=32)
parser.add_argument("--len_traj", type=int, default=15)
# parser.add_argument("--learn_method", type=str, default='none',
#                     help='can be none|bc_elite|bc_contrast')
# parser.add_argument("--freq_learn", type=int, default=50)
# parser.add_argument("--reward", type=str, default='log_n_seen')
# parser.add_argument("--lr", type=float, default=1e-3)
# parser.add_argument("--n_nodes_select", type=int, default=500)
# parser.add_argument("--n_learn_updates", type=int, default=30)
# parser.add_argument("--batch_size", type=int, default=4096)
# parser.add_argument("--coef_entropy", type=float, default=1e-2)


if __name__=='__main__':
    args = parser.parse_args()
    main(args)





