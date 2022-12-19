from collections import defaultdict

import numpy as np
import torch


class Node():
    def __init__(self, parent, traj, snapshot, cell, done):
        self.parent = parent
        self.children = []
        
        self.traj = traj

        self.snapshot = snapshot
        self.cell = cell
        self.done = done
        
    def get_actions(self):
        actions = self.snapshot
        return actions

class GoExplore():
    def __init__(self, env, explorer, device=None):
        self.env = env
        self.explorer = explorer
        self.device = device

        self.cell2node = {}
        # self.cell2n_seen = defaultdict(lambda : 0)
        self.cell2n_seen = {}
        snapshot, obs, reward, done, info = self.env.reset()
        cell = self.env.get_cell(obs)
        self.node_root = Node(None, [], snapshot[0], cell[0], done[0].item())
        self.add_node(self.node_root)
        self.cell2n_seen[cell[0]] = 0
        
    def add_node(self, node):
        if node.cell in self.cell2node:
            node_old = self.cell2node[node.cell]
            if len(node.get_actions())<len(node_old.get_actions()):
                self.cell2node[node.cell] = node
        else:
            self.cell2node[node.cell] = node
    
    def select_nodes(self, n_nodes, strategy='normal'):
        cells = list(self.cell2node.keys())
        if strategy=='random':
            p = None
        else:
            n_seen = np.array([self.cell2n_seen[cell] for cell in cells])
            p = 1./np.sqrt(n_seen+1)
            if strategy=='reverse':
                p = 1/p
            p = p/p.sum()
        return [self.cell2node[cells[i]] for i in np.random.choice(len(cells), size=n_nodes, p=p)]
    
    # def explore_from_single(self, nodes, len_traj):
    #     self.explorer = self.explorer.to(self.device)
    #     # trajs = [[] for _ in nodes] # list of tuples (snapshot, obs, action, reward, done)
    #     snapshot = [node.snapshot for node in nodes]
    #     snapshot, obs, reward, done, info = self.env.reset(snapshot)
    #     for i_trans in range(len_traj):
    #         with torch.no_grad():
    #             action, log_prob, entropy, values = self.explorer.get_action_and_value(obs.to(self.device))
    #         snapshot_next, obs_next, reward, done_next, info = self.env.step(action.cpu())
    #         # for i, traj in enumerate(trajs):
    #         #     traj.append((snapshot[i], obs[i].cpu(), action[i].cpu(), reward[i].cpu(), done[i].cpu()))
    #         # snapshot, obs, done = snapshot_next, obs_next, done_next
    #     cell = self.env.get_cell(snapshot, obs)
    #     return [Node(nodes[i], [], snapshot[i], obs[i].cpu(), cell[i], done[i].cpu()) for i in range(len(nodes))]
    
    # def explore_from(self, nodes, len_traj, n_trajs, add_nodes=True):
    #     for node in nodes:
    #         self.cell2n_seen[node.cell] += 1
            
    #     for _ in range(n_trajs):
    #         nodes = self.explore_from_single(nodes, len_traj)
    #         if add_nodes:
    #             for node in nodes:
    #                 self.add_node(node)

    def explore_from(self, nodes, len_traj):
        # self.explorer = self.explorer.to(self.device)

        cellsets = [set(node.cell) for node in nodes]
        snapshot = [node.snapshot for node in nodes]
        snapshot, obs, reward, done, info = self.env.reset(snapshot)
        for i_trans in range(len_traj):
            # with torch.no_grad():
                # action, log_prob, entropy, values = self.explorer.get_action_and_value(obs.to(self.device))
            # snapshot, obs, reward, done, info = self.env.step(action.cpu())
            snapshot, obs, reward, done, info = self.env.step(self.env.action_space.sample())
            cell = self.env.get_cell(obs)

            for i in range(len(nodes)):
                cellsets[i].add(cell[i])
                self.add_node(Node(None, [], snapshot[i], cell[i], done[i].item()))
        
        for cellset in cellsets:
            for cell in cellset:
                if cell not in self.cell2n_seen:
                    self.cell2n_seen[cell] = 0
                self.cell2n_seen[cell] += 1
    
    def step(self, n_nodes, len_traj):
        nodes = self.select_nodes(n_nodes)
        self.explore_from(nodes, len_traj)

def calc_reward_dfs(ge, reduce=lambda x, childs: x, log=True):
    """
    Calculates a reward with DFS on the GoExplore tree.
    """
    node2prod = {}
    def recurse(node):
        my_novelty = -ge.cell2n_seen[node.latent]
        if node.children:
            for child in node.children:
                recurse(child)
            prod = reduce(my_novelty, [node2prod[child] for child in node.children])
        else:
            prod = my_novelty
        node2prod[node] = prod
    recurse(ge.node_root)
    prod = torch.tensor([node2prod[node] for node in ge]).float()
    return -(-prod).log() if log else prod

def calc_reward_novelty(ge, log=True):
    """
    Calculates a reward using n_seen
    """
    n_seen = torch.tensor([ge.cell2n_seen[node.latent] for node in ge]).float()
    return -n_seen.log() if log else -n_seen