import numpy as np
import torch
from torch import nn

import sklearn

import goexplore

class RandomExplorer():
    def __init__(self):
        self.goexplore = None
        
    def get_action(self, state_sim, obs, latent, action=None):
        # return self.env.action_space.sample()
        logits = torch.log(torch.ones(4))
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), logits

    def update_policy(self, nodes, *args, **kwargs):
        pass
    
class PolicyExplorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.goexplore = None
        
        self.net = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 4),
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.losses = []


    def forward(self, x):
        return self.net(x)
        
    def get_action(self, state_sim, obs, latent, action=None):
        # p = obs[:, [0,1]]
        # d = nn.functional.one_hot(obs[:, 2], 4)
        # x = torch.cat([p, d], dim=-1).float()
        logits = self(latent)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), logits

    def update_policy(self, nodes, n_epochs_policy, batch_size_policy):
        # print(f'updating policy with {len(nodes)} nodes')
        nodes_history = self.goexplore.archive.nodes
        n_trajs, len_traj = len(nodes), len(nodes[0].traj)
        
        prods = compute_productivities(nodes)
        prods = torch.tensor([prods[node] for node in nodes]).float()
        prods = prods[:, None].expand(-1, len_traj)
        
        # TODO normalize obs
        
#     nodes = self.goexplore.archive.nodes[1:] # ignore root because it has not trajectory
        obss = torch.stack([torch.stack([trans[1] for trans in node.traj]) for node in nodes]).float()
        actions = torch.stack([torch.stack([trans[2] for trans in node.traj]) for node in nodes]).float()
        # shape: n_trajs, len_traj, ...
        
        returns = (prods - prods.mean())/(prods.std()+1e-6)
        
        obss_f, actions_f, returns_f = obss.view(-1, obss.shape[-1]), actions.view(-1), returns.view(-1)
        
        # pe = explorers.PolicyExplorer()
        
        for i_epoch in range(n_epochs_policy):
            for i_batch, idx_b in enumerate(torch.randperm(n_trajs*len_traj).split(batch_size_policy)):
                obss_b, actions_b, returns_b = obss_f[idx_b], actions_f[idx_b], returns_f[idx_b]
                latent_b = self.goexplore.state2latent(None, obss_b)
                # print(obss_b.shape, actions_b.shape, returns_b.shape)
                
                _, logprob_b, _ = self.get_action(None, obss_b, latent_b, action=actions_b)

                loss = (-returns_b*logprob_b).mean()

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.losses.append(loss.item())
        
        # plt.plot(losses)
        # plt.show()


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

    

