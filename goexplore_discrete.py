from collections import defaultdict

import numpy as np
import torch
from torch import nn

class RandomNodeSelector():
    def __init__(self, goexplore):
        self.goexplore = goexplore
        
    def select_node(self):
        idxs = np.arange(len(self.goexplore.archive.nodes))
        return self.goexplore.archive.nodes[np.random.choice(idxs, p=None)]
    
class CountNodeSelector():
    def __init__(self, goexplore):
        self.goexplore = goexplore
        
    def select_node(self):
        idxs = np.arange(len(self.goexplore.archive.nodes))
        visits = np.array([node.n_visits for node in self.goexplore.archive.nodes])
        p = 1/np.sqrt(visits+1)
        return self.goexplore.archive.nodes[np.random.choice(idxs, p=p/p.sum())]

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
        
        self.n_visits = 0
        
    def get_full_trajectory(self):
        if self.parent is None:
            return self.traj
        return self.parent.get_full_trajectory()+self.traj

class Archive():
    def __init__(self, node_root):
        # self.goexplore = goexplore
        self.node_root = node_root
        self.nodes = []
        
        self.latent2nodes = defaultdict(lambda : []) # maps from latent to list of nodes
        
        self.add_node(node_root)
        
        self.n_successes, self.n_fails = 0, 0
        
    def add_node(self, node):
        self.nodes.append(node)
        if node.parent:
            node.parent.children.append(node)
        self.latent2nodes[node.latent].append(node)
        
    def replace_node(self, node_old, node_new):
        self.nodes.remove(node_old)
        self.nodes.append(node_new)
        # TODO handle replacements what should happen whenyoureplacea node? (i.e. w the children and stuff)
        
        
    def remove_node(self, node):
        self.nodes.remove(node)
        if node.parent:
            node.parent.children.remove(node)
        self.latent2nodes[node.latent].remove(node)
        
    def add_node_if_better(self, node):
        if node.latent in self.latent2nodes:
            # seen before
            node_existing = self.latent2nodes[node.latent][0]
            trajlen = len(node.get_full_trajectory())
            trajlen_existing = len(node_existing.get_full_trajectory())
            if trajlen<trajlen_existing: # then only replace
                self.remove_node(node_existing)
                self.add_node(node)
                self.n_successes+=1
            else:
                self.n_fails+=1
        else:
            self.n_successes+=1
            self.add_node(node)
        
class GoExplore():
    def __init__(self, env, NodeSelector, StateObs2Latent, Explorer, n_exploration_steps=10):
        self.env = env
        self.node_selector = NodeSelector(self)
        self.stateobs2latent = StateObs2Latent(self)
        self.explorer = Explorer(self)
        
        self.n_exploration_steps = n_exploration_steps
        
        obs, _ = self.env.reset()
        state_sim, obs, reward, done, info = self.env_get_state()
        latent = self.stateobs2latent(state_sim, obs)
        node_root = Node(None, [], state_sim, obs, latent, depth=0)
        self.archive = Archive(node_root)
        
        self.selected = []
        
    def select_node(self):
        node = self.node_selector.select_node()
        node.n_visits += 1
        self.selected.append(node.state_sim[0])
        return node
    
    def explore_from(self, node_start):
        traj = []
        state_sim, obs = node_start.state_sim, node_start.obs
        for i in range(self.n_exploration_steps):
            latent = self.stateobs2latent.get_latent(state_sim, obs)
            action = self.explorer.get_action(state_sim, obs, latent)
            state_sim_next, obs_next, reward, done, info = self.env.step(action)
            
            traj.append((state_sim, obs, action, reward))
            # traj.append((state_sim, action, reward))
            
            state_sim, obs = state_sim_next, obs_next
            if done:
                print('DONE!!!!!!!')
                break
        
        latent = self.stateobs2latent.get_latent(state_sim, obs)
        node = Node(node_start, traj, state_sim, obs, latent, node_start.depth+1)
        self.archive.add_node_if_better(node)
        return node, done
        
    def run_iteration(self):
        node_start = self.select_node()
        self.env_goto_node(node_start)
        node_end, done = self.explore_from(node_start)
        
    def env_get_state(self):
        return (self.env.agent_pos, self.env.agent_dir)
    
    def env_goto_node(self, node):
        obs, _ = self.env.reset()
        pos, d = node.state_sim
        self.env.env.env.env.agent_pos = pos
        self.env.env.env.env.agent_dir = d

