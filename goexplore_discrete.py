from collections import defaultdict

import numpy as np
import torch

class Node():
    def __init__(self, parent, traj, snapshot, obs, latent, done):
        self.parent = parent
        self.children = []
        
        self.traj = traj

        self.snapshot = snapshot
        self.obs = obs
        self.latent = latent
        self.done = done
        
    def get_full_trajectory(self):
        return ([] if self.parent is None else self.parent.get_full_trajectory()) + self.traj

class GoExplore(list):
    def __init__(self, env, explorer, device=None):
        self.env = env
        self.explorer = explorer
        self.device = device

        self.cell2node = {}
        self.cell2n_seen = defaultdict(lambda : 0)
        snapshot, obs, reward, done, info = self.env.reset()
        node_root = Node(None, [], snapshot[0].cpu(), obs[0].cpu(), self.env.to_latent(snapshot, obs)[0], done[0].cpu())
        self.node_root = node_root
        self.add_node(node_root)
        
    def add_node(self, node):
        self.cell2node[node.latent] = node
        self.append(node)
        if node.parent:
            node.parent.children.append(node)
        
    def add_node(self, node):
        self.cell2n_seen[node.latent] += 1
        self.append(node)
        if node.parent:
            node.parent.children.append(node)
            
        if node.latent in self.cell2node:
            node_old = self.cell2node[node.latent]
            if len(node.get_full_trajectory())<len(node_old.get_full_trajectory()):
                self.cell2node[node.latent] = node
        else:
            self.cell2node[node.latent] = node
    
    def select_nodes(self, n_nodes):
        cells = list(self.cell2node.keys())
        n_seen = np.array([self.cell2n_seen[cell] for cell in cells])
        p = 1./np.sqrt(n_seen+1)
        p = p/p.sum()
        # p = None
        return [self.cell2node[cells[i]] for i in np.random.choice(len(cells), size=n_nodes, p=p)]
    
    def explore_from_single(self, nodes, len_traj):
        trajs = [[] for _ in nodes] # list of tuples (snapshot, obs, action, reward, done)
        snapshot = [node.snapshot for node in nodes]
        snapshot, obs, reward, done, info = self.env.reset(snapshot)
        for i_trans in range(len_traj):
            with torch.no_grad():
                action, log_prob, entropy, values = self.explorer.get_action_and_value(obs)
                # action, log_prob = action.cpu(), log_prob.cpu()
            snapshot_next, obs_next, reward, done_next, info = self.env.step(action)
            for i, traj in enumerate(trajs):
                traj.append((snapshot[i].cpu(), obs[i].cpu(), action[i].cpu(), reward[i].cpu(), done[i].cpu()))
            snapshot, obs, done = snapshot_next, obs_next, done_next
            
        latent = self.env.to_latent(snapshot, obs)
        return [Node(nodes[i], trajs[i], snapshot[i].cpu(), obs[i].cpu(), latent[i], done[i].cpu()) for i in range(len(nodes))]
    
    def explore_from(self, nodes, len_traj, n_trajs, add_nodes=True):
        for node in nodes:
            self.cell2n_seen[node.latent] += 1
            
        for _ in range(n_trajs):
            nodes = self.explore_from_single(nodes, len_traj)
            if add_nodes:
                for node in nodes:
                    self.add_node(node)
