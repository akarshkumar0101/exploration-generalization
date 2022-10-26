from collections import defaultdict

import numpy as np
import torch
from torch import nn

import sklearn.mixture

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
        return self.parent.get_full_trajectory() + self.traj

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
    def __init__(self, env, returner, explorer, state2latent, n_exploration_steps=10):
        self.env = env
        self.returner = returner
        self.explorer = explorer
        self.state2latent = state2latent

        self.returner.goexplore = self
        self.explorer.goexplore = self
        self.state2latent.goexplore = self
        
        self.n_exploration_steps = n_exploration_steps
        
        state_sim, obs, reward, done, info = self.env.reset()
        latent = self.state2latent(state_sim, obs)
        node_root = Node(None, [], state_sim, obs, latent, depth=0)
        self.archive = Archive(node_root)

        self.latents = torch.stack([node.latent for node in self.archive.nodes])
        
        self.selected = []

        self.i_step = 0
        
    def select_node(self):
        self.returner.fit(len(self.archive.nodes))
        logits = self.returner.score_nodes(self.latents)
        # node, logits_selection = self.returner.select_node()

        p = logits.softmax(dim=0).numpy()
        i_node = np.random.choice(len(self.archive.nodes), p=p)
        node = self.archive.nodes[i_node]

        # self.logits_selection = logits_selection
        self.selected.append(node)
        return node
    
    def explore_from(self, node_start):
        traj = []
        state_sim, obs = node_start.state_sim, node_start.obs
        for i in range(self.n_exploration_steps):
            latent = self.state2latent(state_sim, obs)
            action = self.explorer.get_action(state_sim, obs, latent)
            state_sim_next, obs_next, reward, done, info = self.env.step(action)
            
            traj.append((state_sim, obs, action, reward))
            # traj.append((state_sim, action, reward))
            
            state_sim, obs = state_sim_next, obs_next
            if done:
                print('DONE!!!!!!!')
                break
        
        latent = self.state2latent(state_sim, obs)
        node = Node(node_start, traj, state_sim, obs, latent, node_start.depth+1)
        self.archive.add_node(node)
        return node, done

    def smart_step(self):
        pass
        
    def step(self):
        node_start = self.select_node()
        self.env.goto_state_sim(node_start.state_sim)
        node_end, done = self.explore_from(node_start)

        self.latents = torch.stack([node.latent for node in self.archive.nodes])

        self.i_step += 1
        

