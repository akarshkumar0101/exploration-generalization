import argparse
import copy
import pickle
from distutils.util import strtobool

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from torch import nn
from tqdm.auto import tqdm

import bc
import wandb


class MRDomainCellInfo(gym.Wrapper):
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        self.y, self.x = self.get_agent_yx(info['obs_ori'])
        self.roomx, self.roomy = 0, 0
        self.inventory = None

        self.update_info(info['obs_ori'], info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.update_info(info['obs_ori'], info)
        return obs, reward, terminated, truncated, info

    def update_info(self, obs, info):
        y, x = self.get_agent_yx(obs)
        if (x-self.x)>5:
            self.roomx -= 1
        elif (x-self.x)<-5:
            self.roomx += 1
        if (y-self.y)>5:
            self.roomy -= 1
        elif (y-self.y)<-5:
            self.roomy += 1
        self.y, self.x = y, x
        self.inventory = self.get_cell_inventory(obs)
        info['cell'] = (self.y, self.x, self.roomy, self.roomx)+tuple(self.inventory.flatten())
        info['y'], info['x'] = self.y, self.x
        info['roomy'], info['roomx'] = self.roomy, self.roomx
        info['inventory'] = self.inventory

    def get_agent_yx(self, obs):
        h, w, c = obs.shape
        y, x = np.where((obs[:, :, 0]==228))
        if len(y)>0:
            y, x = np.mean(y), np.mean(x)
            y, x = int(y/h*16), int(x/w*16)
            return y, x
        else:
            return self.y, self.x

    def get_cell_inventory(self, obs):
        y, x, c = obs.shape
        # obs_key = obs[int(.47*y): int(.55*y), int(.05*x): int(.15*x), ...] # location of first key
        obs_inventory = obs[int(.1*y): int(.22*y), int(.3*x): int(.65*x), ...]
        cell = cv2.cvtColor(obs_inventory, cv2.COLOR_RGB2GRAY)
        cell = cv2.resize(cell, (6, 3), interpolation=cv2.INTER_AREA)
        cell = (cell/255. * 8).astype(np.uint8, casting='unsafe')
        return cell

class ImageCellInfo(gym.Wrapper):
    def __init__(self, env, latent_h=11, latent_w=8, latent_d=20):
        super().__init__(env)
        self.latent_h, self.latent_w, self.latent_d = latent_h, latent_w, latent_d

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info['cell_img'] = self.get_cell_obs(obs, self.latent_h, self.latent_w, self.latent_d, ret_tuple=False)
        info['cell'] = tuple(info['cell_img'].flatten())
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['cell_img'] = self.get_cell_obs(obs, self.latent_h, self.latent_w, self.latent_d, ret_tuple=False)
        info['cell'] = tuple(info['cell_img'].flatten())
        return obs, reward, terminated, truncated, info
    def get_cell_obs(self, obs, latent_h=11, latent_w=8, latent_d=20, ret_tuple=True):
        # obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (latent_w, latent_h), interpolation=cv2.INTER_AREA)
        obs = (obs*latent_d).astype(np.uint8, casting='unsafe')
        return tuple(obs.flatten()) if ret_tuple else obs





