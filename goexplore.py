from collections import defaultdict

import numpy as np
import torch
from torch import nn

import sklearn.mixture

class RandomNodeSelector():
    def __init__(self, goexplore):
        self.goexplore = goexplore
        
    def select_node(self):
        idxs = np.arange(len(self.goexplore.archive.nodes))
        return self.goexplore.archive.nodes[np.random.choice(idxs, p=None)]
    
# class CountNodeSelector():
#     def __init__(self, goexplore):
#         self.goexplore = goexplore
        
#     def select_node(self):
#         idxs = np.arange(len(self.goexplore.archive.nodes))
#         visits = np.array([node.n_visits for node in self.goexplore.archive.nodes])
#         p = 1/np.sqrt(visits+1)
#         return self.goexplore.archive.nodes[np.random.choice(idxs, p=p/p.sum())]
    
class GaussiainNodeSelector():
    def __init__(self, goexplore, n_components=1):
        self.goexplore = goexplore
        self.gm = sklearn.mixture.GaussianMixture(n_components=n_components, random_state=0)
        self.t = 0
        
    def select_node(self):
        n = len(self.goexplore.archive.nodes)
        if n < 2:
            return self.goexplore.archive.nodes[0]

        X = np.array([node.latent for node in self.goexplore.archive.nodes])
        if self.t%10==0:
            self.gm = self.gm.fit(X)
        self.t += 1

        p = -self.gm.score_samples(X) # log probs then "invert" them
        p = torch.from_numpy(p).softmax(dim=0).numpy()
        
        return self.goexplore.archive.nodes[np.random.choice(n, p=p)]

class Node():
    def __init__(self, parent, traj, state_sim, obs, latent, depth):
        self.parent = parent
        self.children = []
        
        self.traj = traj
        self.state_sim = state_sim
        self.obs = obs
        self.latent = latent
        self.depth = depth
        self.productivity = 0
        
    def get_full_trajectory(self):
        if self.parent is None:
            return self.traj
        return self.parent.get_full_trajectory()+self.traj

class Archive():
    def __init__(self, node_root):
        # self.goexplore = goexplore
        self.node_root = node_root
        self.nodes = []
        
        self.add_node(node_root)
        
    def add_node(self, node):
        self.nodes.append(node)
        if node.parent:
            node.parent.children.append(node)
        
    # def remove_node(self, node):
    #     self.nodes.remove(node)
    #     if node.parent:
    #         node.parent.children.remove(node)
    #     self.latent2nodes[node.latent].remove(node)
        
class GoExplore():
    def __init__(self, env, NodeSelector, StateObs2Latent, Explorer, n_exploration_steps=10):
        self.env = env
        self.node_selector = NodeSelector(self)
        self.stateobs2latent = StateObs2Latent(self)
        self.explorer = Explorer(self)
        
        self.n_exploration_steps = n_exploration_steps
        
        state_sim, obs, reward, done, info = self.env.reset()
        latent = self.stateobs2latent(state_sim, obs)
        node_root = Node(None, [], state_sim, obs, latent, depth=0)
        self.archive = Archive(node_root)
        
        self.selected = []
        
    def select_node(self):
        node = self.node_selector.select_node()
        self.selected.append(node)
        return node
    
    def explore_from(self, node_start):
        traj = []
        state_sim, obs = node_start.state_sim, node_start.obs
        for i in range(self.n_exploration_steps):
            latent = self.stateobs2latent(state_sim, obs)
            action = self.explorer.get_action(state_sim, obs, latent)
            state_sim_next, obs_next, reward, done, info = self.env.step(action)
            
            traj.append((state_sim, obs, action, reward))
            # traj.append((state_sim, action, reward))
            
            state_sim, obs = state_sim_next, obs_next
            if done:
                print('DONE!!!!!!!')
                break
        
        latent = self.stateobs2latent(state_sim, obs)
        node = Node(node_start, traj, state_sim, obs, latent, node_start.depth+1)
        self.archive.add_node(node)
        return node, done
        
    def run_iteration(self):
        node_start = self.select_node()
        self.env.goto_state_sim(node_start.state_sim)
        node_end, done = self.explore_from(node_start)
        
    
