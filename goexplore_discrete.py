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
        node_root = Node(None, [], snapshot[0], obs[0].cpu(), self.env.to_latent(snapshot, obs)[0], done[0].cpu())
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
                traj.append((snapshot[i], obs[i].cpu(), action[i].cpu(), reward[i].cpu(), done[i].cpu()))
            snapshot, obs, done = snapshot_next, obs_next, done_next
            
        latent = self.env.to_latent(snapshot, obs)
        return [Node(nodes[i], trajs[i], snapshot[i], obs[i].cpu(), latent[i], done[i].cpu()) for i in range(len(nodes))]
    
    def explore_from(self, nodes, len_traj, n_trajs, add_nodes=True):
        for node in nodes:
            self.cell2n_seen[node.latent] += 1
            
        for _ in range(n_trajs):
            nodes = self.explore_from_single(nodes, len_traj)
            if add_nodes:
                for node in nodes:
                    self.add_node(node)

def calc_reward_dfs(ge, reduce1=np.max, reduce2=np.max, log=True):
    """
    Calculates a reward with DFS on the GoExplore tree.
    """
    node2prod = {}
    def recurse(node):
        novelty = -ge.cell2n_seen[node.latent]
        if node.children:
            for child in node.children:
                recurse(child)
            prods_children = [node2prod[child] for child in node.children]
            if reduce2 is None:
                prod = reduce1([novelty]+prods_children)
            else:
                prod = reduce1([novelty, reduce2(prods_children)])
        else:
            prod = novelty
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