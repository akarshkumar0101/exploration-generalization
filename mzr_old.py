import argparse
import copy
import pickle
from distutils.util import strtobool

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import repeat
from matplotlib import cm
from torch import nn
from tqdm.auto import tqdm

import bc

# from goexplore_discrete import CellNode, GoExplore, calc_reward_novelty

def viz_ge_outliers(ge, n_ex=10):
    n_ex = ge.env.n_envs
    cells = list(ge.cell2node.keys())
    n_seen = torch.tensor([ge.cell2n_seen[cell] for cell in cells])
    idx_sort = n_seen.argsort().tolist()

    fig, axs = plt.subplots(2, n_ex, figsize=(2*n_ex, 2.5*2))
    for j, idx in enumerate([idx_sort[:n_ex], idx_sort[-n_ex:][::-1]]):
        nodes = [ge.cell2node[cells[i]] for i in idx]
        n_seen_row = [n_seen[i] for i in idx]
        _, obs, _, _, _ = ge.env.reset(snapshot=[node.snapshot for node in nodes])
        for i in range(n_ex):
            axs[j, i].imshow(obs[i, 0].numpy())
            axs[j, i].set_title(f'{n_seen_row[i]}')
    axs[0, 0].set_ylabel('Lowest n_seen')
    axs[1, 0].set_ylabel('Most n_seen')
    plt.tight_layout()
    return fig

def plot_ge_outlier_coverage(ge):
    o = []
    for i in range(10):
        nodes = ge.select_nodes(ge.env.n_envs)
        _, obs, _, _, _ = ge.env.reset(snapshot=[node.snapshot for node in nodes])
        o.append(obs)
    o = torch.cat(o, dim=0)
    plt.imshow(o.max(dim=0).values.sum(dim=0).numpy())
    return plt.gcf()

# def viz_exploration_strategy(ge, ex, n_ex=10):
#     cells = list(ge.cell2n_seen.keys())
#     n_seen = torch.tensor([ge.cell2n_seen[cell] for cell in cells])
#     idx = n_seen.argsort()

#     obs_low = torch.stack([ge.cell2node[cells[i]].obs for i in idx[:n_ex]])
#     obs_high = torch.stack([ge.cell2node[cells[i]].obs for i in idx[-n_ex:]]).flip(dims=(0,))
#     n_seen_low = n_seen[idx[:n_ex]]
#     n_seen_high = n_seen[idx[-n_ex:]].flip(dims=(0,))

#     logits_low, _ = ex.get_logits_values(obs_low.to(ge.device))
#     logits_high, _ = ex.get_logits_values(obs_high.to(ge.device))
#     logits_low, logits_high = logits_low.cpu(), logits_high.cpu()

#     fig, axs = plt.subplots(2, n_ex*2, figsize=(2*n_ex*2, 2.5*2))
#     for i in range(n_ex):
#         axs[0, i*2].imshow(obs_low[i, 0].numpy())
#         axs[1, i*2].imshow(obs_high[i, 0].numpy())
#         axs[0, i*2+1].bar(torch.arange(logits_low.shape[-1]), logits_low[i].softmax(dim=-1).tolist())
#         axs[1, i*2+1].bar(torch.arange(logits_high.shape[-1]), logits_high[i].softmax(dim=-1).tolist())
#         axs[0, i*2].set_title(f'{n_seen_low[i].item()}')
#         axs[1, i*2].set_title(f'{n_seen_high[i].item()}')
#     axs[0, 0].set_ylabel('Lowest n_seen')
#     axs[1, 0].set_ylabel('Most n_seen')
#     plt.tight_layout()
#     return plt.gcf()

def viz_cells(ge, n_ex=10):
    n_ex = ge.env.n_envs
    nodes = ge.select_nodes(ge.env.n_envs)
    _, obs, _, _, _ = ge.env.reset(snapshot=[node.snapshot for node in nodes])
    cells = ge.env.get_cell(obs, ret_tuple=False)

    fig, axs = plt.subplots(2, n_ex, figsize=(2*n_ex, 2.5*2))
    for i in range(n_ex):
        axs[0, i].imshow(obs[i, 0].numpy())
        axs[1, i].imshow(cells[i])
    plt.tight_layout()
    return fig

def main(args):
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.track:
        run = wandb.init(
            config=args,
            name=args.name,
            save_code=True)
    
    env = MZRevenge(n_envs=args.n_envs, dead_screen=True).to(args.device)
    # explorer = ImitationExplorer(env, force_random=(args.learn_method=='none')).to(args.device)
    # opt = torch.optim.Adam(explorer.parameters(), lr=args.lr)
    # ge = GoExplore(env, explorer, args.device)
    ge = GoExplore(env, None, args.device)

    pbar = tqdm(range(args.n_steps))
    for i_step in pbar:
        data = {}
        ge.step(args.n_envs, args.len_traj)

        # if args.learn_method!='none' and i_step>0 and i_step%args.freq_learn==0:
        #     if args.learn_method=='bc_elite':
        #         losses, entropies, logits_std = bc.train_bc_elite(ge, explorer, opt, args.n_nodes_select, args.n_learn_updates,
        #                                                           args.batch_size, args.coef_entropy, args.device)
        #     elif args.learn_method=='bc_contrast':
        #         losses, entropies, logits_std = bc.train_contrastive_bc(ge, explorer, opt, args.n_learn_updates, 
        #                                                                 args.batch_size, args.coef_entropy, args.device)
            # data.update(loss_start=losses[0].item(), loss_end=losses[-1].item())


        cells = list(ge.cell2node.keys())
        n_seen = np.array([ge.cell2n_seen[cell] for cell in cells])
        traj_lens = [len(node.snapshot) for node in ge.cell2node.values()]
        # n_dead = torch.stack([node.done for node in ge]).sum().item()
        data.update(dict(n_cells=len(ge.cell2node), n_seen_max=n_seen.max(),
                         hist_n_seen=wandb.Histogram(n_seen), hist_traj_lens=wandb.Histogram(traj_lens)))
        if args.track:
            if i_step%args.freq_viz==0:
                data['coverage'] = plot_ge_outlier_coverage(ge)
                data['outliers'] = viz_ge_outliers(ge, n_ex=10)
                # data['exploration_strategy'] = viz_exploration_strategy(ge, explorer, n_ex=10)
                data['cells'] = viz_cells(ge, n_ex=10)
            wandb.log(data)
            plt.close('all')

            if i_step%args.freq_save==0:
                torch.save(ge, f'results/ge.pt')

        pbar.set_postfix({k: v for k, v in data.items() if isinstance(v, float)})
        
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
parser.add_argument("--n_envs", type=int, default=10)
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


