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

        self.cell2node = {} # the node
        self.cell2n_seen = {} # the number of times I've seen this cell
        self.cell2prod = {} # number of times this cell has not produced a new cell recently
        obs, info = self.env.reset()
        self.node_root = Node(None, info['snapshot'][0], info['cell'][0], False)
        self.add_node(self.node_root)
        self.cell2n_seen[self.node_root.cell] = 0
        self.cell2prod[self.node_root.cell] = 0
        
    def add_node(self, node):
        if node.terminated:
            return

        if node.cell in self.cell2node:
            node_old = self.cell2node[node.cell]
            if len(node.snapshot)<len(node_old.snapshot):
                self.cell2node[node.cell] = node
        else:
            self.cell2node[node.cell] = node
    
    def select_nodes(self, n_nodes, beta=-0.5, condition=lambda node: True):
        """
        beta =  0.0 is p ~ 1 (uniform)
        beta = -0.5 is p ~ 1/sqrt(n_seen)
        beta = -1.0 is p ~ 1/n_seen
        beta = -2.0 is p ~ 1/n_seen**2
        """
        if condition is None:
            condition = lambda node: True
        nodes = [node for node in self.cell2node.values() if condition(node)]
        x = torch.as_tensor([self.cell2n_seen[node.cell] for node in nodes])
        p = (beta*(x+1).log()).softmax(dim=0)
        return np.random.choice(nodes, size=n_nodes, p=p.numpy())
    
    def explore_from(self, nodes, len_traj, agent_explorer=None):
        snapshots = [node.snapshot for node in nodes]
        nodes_visited = [[node] for node in nodes]

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
            nodes_traj = [node for node in nodes_traj if not node.terminated]
            nodes_traj = nodes_traj[:-2] # we can't ensure the future is not dead

            for node in nodes_traj:
                self.add_node(node)

            cells_traj = [node.cell for node in nodes_traj]
            cells_traj_set = set(cells_traj)
            is_novel_traj = not cells_traj_set.issubset(set(self.cell2node.keys()))

            for cell in cells_traj_set:
                if cell not in self.cell2n_seen:
                    self.cell2n_seen[cell] = 0
                self.cell2n_seen[cell] += 1

            for cell in cells_traj_set:
                if is_novel_traj or cell not in self.cell2prod:
                    self.cell2prod[cell] = 0
                else:
                    self.cell2prod[cell] += 1

            # for cell in cells_traj_set:
            # # for cell in cells_traj:
            #     if cell not in self.cell2n_seen:
            #         self.cell2n_seen[cell] = 0
            #     self.cell2n_seen[cell] += 1
    
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