from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch


class Node():
    def __init__(self, parent, snapshot, cell, terminated):
        self.parent = parent
        self.children = []
        
        self.snapshot = snapshot
        self.cell = cell
        self.terminated = terminated

class GoExplore():
    def __init__(self, env):
        self.env = env

        self.cell2node = {}
        self.cell2n_seen = {}
        obs, info = self.env.reset()
        self.node_root = Node(None, info['snapshot'][0], info['cell'][0], False)
        self.add_node(self.node_root)
        self.cell2n_seen[self.node_root.cell] = 0
        
    def add_node(self, node):
        if node.terminated:
            return

        if node.cell in self.cell2node:
            node_old = self.cell2node[node.cell]
            if len(node.snapshot)<len(node_old.snapshot):
                self.cell2node[node.cell] = node
        else:
            self.cell2node[node.cell] = node
    
    def select_nodes(self, n_nodes, beta=-0.5):
        """
        beta =  0.0 is p ~ 1 (uniform)
        beta = -0.5 is p ~ 1/sqrt(n_seen)
        beta = -1.0 is p ~ 1/n_seen
        beta = -2.0 is p ~ 1/n_seen**2
        """
        cells = list(self.cell2node.keys())
        nodes = list(self.cell2node.values())
        n_seen = torch.as_tensor([self.cell2n_seen[cell] for cell in cells])
        p = (beta*n_seen.log()).softmax(dim=0)
        return np.random.choice(nodes, size=n_nodes, p=p.numpy())
    
    def explore_from(self, nodes, len_traj, agent_explorer=None):
        snapshots = [node.snapshot for node in nodes]
        nodes_visited = [[] for node in nodes]

        obs, reward, terminated, truncated, info = self.env.restore_snapshot(snapshots)
        for i_trans in range(len_traj):
            if agent_explorer is None:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    action = agent_explorer.act(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)

            for i in range(len(nodes)):
                node = Node(None, info['snapshot'][i], info['cell'][i], terminated[i].item())
                nodes_visited[i].append(node)

        for nodes_traj in nodes_visited:
            for node in nodes_traj:
                self.add_node(node)
            for cell in set([node.cell for node in nodes_traj]):
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