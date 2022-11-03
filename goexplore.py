from collections import defaultdict

import numpy as np
import torch
from torch import nn

import sklearn.mixture

class Node():
    def __init__(self, parent, traj, state_sim, obs, done, latent, depth):
        self.parent = parent
        self.children = []
        
        self.traj = traj

        self.state_sim = state_sim
        self.obs = obs
        self.done = done

        self.latent = latent
        self.depth = depth
        
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
    def __init__(self, env, returner, explorer, state2latent):
        self.env = env
        self.returner = returner
        self.explorer = explorer
        self.state2latent = state2latent

        self.returner.goexplore = self
        self.explorer.goexplore = self
        self.state2latent.goexplore = self
        
        state_sim, obs, reward, done, info = self.env.reset()
        latent = self.state2latent(state_sim, obs)
        node_root = Node(None, [], state_sim, obs, done, latent, depth=0)
        self.archive = Archive(node_root)

        self.latents = torch.stack([node.latent for node in self.archive.nodes])
        
        self.selected = []

        self.i_step = 0
        
    def select_node(self):
        # self.returner.fit(len(self.archive.nodes))
        logits = self.returner.score_nodes(self.latents)
        # node, logits_selection = self.returner.select_node()

        p = logits.softmax(dim=0).numpy()
        i_node = np.random.choice(len(self.archive.nodes), p=p)
        node = self.archive.nodes[i_node]

        # self.logits_selection = logits_selection
        self.selected.append(node)
        return node

    def explore_from(self, node_start, len_traj):
        traj = [] # list of tuples (state, obs, action, log_probs, reward)
        # state_sim, obs = node_start.state_sim, node_start.obs
        state_sim, obs, reward, done, info = self.env.goto_state_sim(node_start.state_sim)
        for i_trans in range(len_traj):
            latent = self.state2latent(state_sim, obs)
            action, log_prob, _ = self.explorer.get_action(state_sim, obs, latent)
            
            state_sim_next, obs_next, reward, done, info = self.env.step(action)
            traj.append((state_sim, obs, done, action, log_prob, reward))
            state_sim, obs = state_sim_next, obs_next
            if done:
                print('DONE!!!!!!!')
                break

        latent = self.state2latent(state_sim, obs)
        node = Node(node_start, traj, state_sim, obs, done, latent, node_start.depth+1)
        return node, done

    def collect_policy_data(self, n_trajs, len_traj):
        nodes = []
        for i_traj in range(n_trajs):
            node_start = self.select_node()
            self.env.reset(node_start.state_sim)
            node_end, done = self.explore_from(node_start, len_traj)
            nodes.append(node_end)
        return nodes

    def step(self, n_trajs, len_traj, n_epochs_policy, batch_size_policy):
        # collect new exploration data
        nodes = self.collect_policy_data(n_trajs, len_traj)
        
        # update policy using this data
        self.explorer.update_policy(nodes, n_epochs_policy, batch_size_policy)
        
        # add the end nodes to the archive
        for node in nodes:
            self.archive.add_node(node)
        self.latents = torch.stack([node.latent for node in self.archive.nodes])

        self.returner.fit(self.latents)

