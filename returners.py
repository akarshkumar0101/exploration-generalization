import numpy as np

import torch

import sklearn

class RandomNodeSelector():
    def __init__(self):
        self.goexplore = None

    def fit(self, latents):
        pass

    def score_nodes(self, X):
        n = len(X)
        logits = (torch.ones(n)/n).log()
        return logits
        
    # def select_node(self):
    #     n_nodes = len(self.goexplore.archive.nodes)
    #     logits = np.log(np.ones(n_nodes) / n_nodes)
    #     p = torch.from_numpy(logits).softmax(dim=0).numpy()
    #     return self.goexplore.archive.nodes[np.random.choice(n_nodes, p=p)], logits
    
# class CountNodeSelector():
#     def __init__(self, goexplore):
#         self.goexplore = goexplore
        
#     def select_node(self):
#         idxs = np.arange(len(self.goexplore.archive.nodes))
#         visits = np.array([node.n_visits for node in self.goexplore.archive.nodes])
#         p = 1/np.sqrt(visits+1)
#         return self.goexplore.archive.nodes[np.random.choice(idxs, p=p/p.sum())]
    
class GaussianNodeSelector():
    def __init__(self, n_components=1):
        self.goexplore = None
        self.gm = sklearn.mixture.GaussianMixture(n_components=n_components, random_state=0)
        # self.t = 0

    def fit(self, latents):
        X = latents
        if len(X) < 2:
            return
        self.gm = self.gm.fit(X.numpy())

    def score_nodes(self, latents):
        X = latents
        if len(X)<2:
            return torch.log(torch.ones(len(X)))
        logits = -self.gm.score_samples(X.numpy()) # log probs then "invert" them
        return torch.from_numpy(logits)
        
    def select_node(self):
        n = len(self.goexplore.archive.nodes)
        if n < 2:
            return self.goexplore.archive.nodes[0], np.log(np.ones(n))

        X = torch.stack([node.latent for node in self.goexplore.archive.nodes])
        if self.t%10==0:
            self.gm = self.gm.fit(X.numpy())
        self.t += 1

        logits = -self.gm.score_samples(X) # log probs then "invert" them
        p = torch.from_numpy(logits).softmax(dim=0).numpy()
        return self.goexplore.archive.nodes[np.random.choice(n, p=p)], logits
