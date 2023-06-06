import os
import re
from functools import partial

import cv2

# import gym # as gym_old
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat, einsum
from IPython.display import clear_output
from torch import nn
from tqdm.auto import tqdm

from env_atari import *
from agent_atari import *
from time_contrastive import *

import torchinfo