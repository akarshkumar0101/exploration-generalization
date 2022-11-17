import numpy as np
import torch


def count_params(net):
    return np.sum([p.numel() for p in net.parameters()])

def normalize(a, devs=None):
    b = a
    if devs is not None:
        # mask = ((a-a.mean())/a.std()).abs()<devs
        mask = ((a-a.median())/a.std()).abs()<devs
        b = a[mask]
    return (a-b.mean())/(a.std()+1e-9)