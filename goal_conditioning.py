
from collections import OrderedDict
from functools import partial

import gym
import numpy as np
import torch
from einops import rearrange
from torch import nn

import ppo_simple
import utils


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
