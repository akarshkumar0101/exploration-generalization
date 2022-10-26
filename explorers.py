import numpy as np
import torch
from torch import nn

import sklearn

import goexplore

class RandomExplorer():
    def __init__(self):
        self.goexplore = None
        
    def get_action(self, state_sim, obs, latent):
        # return self.env.action_space.sample()
        return np.random.choice(3)
    
class PolicyExplorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.goexplore = None
        
        self.net = nn.Sequential(
            nn.Linear(6, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 3),
        )
    def forward(self, x):
        return self.net(x)
        
    def get_action(self, state_sim, obs, latent, action=None):
        p = obs[:, [0,1]]
        d = nn.functional.one_hot(obs[:, 2], 4)
        x = torch.cat([p, d], dim=-1).float()
        logits = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action)

    def step():
        pass

    # 
    
def compute_productivities(nodes, alpha=0.01, child_aggregate='mean'):
    """
    Compute the productivity of each node in the archive.

    alpha determines how much I care about myself vs my children.
    """
    latents = torch.stack([node.latent for node in nodes])
    
    gm = sklearn.mixture.GaussianMixture(n_components=1)
    gm = gm.fit(latents.numpy())
    
    novelties = -gm.score_samples(latents.numpy())
    novelties = {node: n for node, n in zip(nodes, novelties)}
    
    productivities = {}

    if child_aggregate == 'mean':
        child_aggregate = np.mean
    elif child_aggregate == 'max':
        child_aggregate = np.max
    
    # gamma = 1.
    # alpha = .01 # how much do I care about myself vs children?
    
    def dfs_productivity(node, depth=0):
        # assert node.depth==depth
        
        productivity = novelties[node]
        if node.children:
            prods_child = []
            for child in node.children:
                # if child not in nodes:
                    # continue
                dfs_productivity(child, depth=depth+1)
                prods_child.append(productivities[child])
            # should we use mean or max here?
            # should we consider it good to have one good children or do all children have to be good?
            productivity = alpha*productivity + (1-alpha)*child_aggregate(prods_child)
        else:
            productivity = alpha*productivity + (1-alpha)*productivity # no children, so productivity is just novelty
        productivities[node] = productivity
    
    for node in nodes:
        if node not in productivities:
            dfs_productivity(node)
    
    return productivities
    

