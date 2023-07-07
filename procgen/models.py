import gymnasium as gym
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        fs, h, w, c = env.single_observation_space.shape
        
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(fs*c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64*4*4, 256)),
            nn.ReLU(),
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(128, env.single_action_space.n), std=0.01),
        )
        self.critic_ext = nn.Sequential(
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 32), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=0.01),
        )
        self.critic_int = nn.Sequential(
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 32), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=0.01),
        )
    
    def preprocess(self, x):
        return rearrange(x, 'b fs h w c -> b (fs c) h w').float()/128.-1. # [0, 255] -> [-1, 1]

    def get_value(self, x):
        x = self.network(self.preprocess(x))
        return self.critic_ext(x), self.critic_int(x)

    def get_action_and_value(self, x, action=None):
        x = self.preprocess(x)
        x = self.network(x)
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic_ext(x), self.critic_int(x)
    
    def get_dist_and_values(self, x):
        x = self.preprocess(x)
        x = self.network(x)
        logits = self.actor(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, self.critic_ext(x), self.critic_int(x)



class BigAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        fs, h, w, c = env.single_observation_space.shape

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(fs * c, 64, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 128, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(128 * 4 * 4, 512)),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(512, env.single_action_space.n), std=0.01),
        )
        self.critic_ext = nn.Sequential(
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=0.01),
        )
        self.critic_int = nn.Sequential(
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=0.01),
        )

    def preprocess(self, x):
        return rearrange(x, 'b fs h w c -> b (fs c) h w').float() / 128. - 1.  # [0, 255] -> [-1, 1]

    def get_value(self, x):
        x = self.network(self.preprocess(x))
        return self.critic_ext(x), self.critic_int(x)

    def get_action_and_value(self, x, action=None):
        x = self.preprocess(x)
        x = self.network(x)
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic_ext(x), self.critic_int(x)

    def get_dist_and_values(self, x):
        x = self.preprocess(x)
        x = self.network(x)
        logits = self.actor(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, self.critic_ext(x), self.critic_int(x)


class RNDModel(nn.Module):
    def __init__(self, env, obs_shape=(64, 64, 3)):
        super().__init__()
        
        h, w, c = obs_shape
        self.obs_rms = gym.wrappers.normalize.RunningMeanStd(shape=obs_shape)

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64*4*4, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64*4*4, 512)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False
        
    def update(self, x):
        # x.shape is b, fs, h, w, c
        assert isinstance(x, np.ndarray)
        self.obs_rms.update(x[:, -1])
    
    def preprocess(self, x):
        # x.shape is b, fs, h, w, c
        assert isinstance(x, torch.Tensor)
        x = x[:, -1] # get only last frame
        mean = torch.from_numpy(self.obs_rms.mean).to(x)
        var = torch.from_numpy(self.obs_rms.var).to(x)
        x = (x.float()-mean)/var.sqrt()
        x = x.clip(-5., 5.)
        x = rearrange(x, 'b h w c -> b c h w')
        return x

    def forward(self, x):
        x = self.preprocess(x)
        target_feature = self.target(x)
        predict_feature = self.predictor(x)
        return predict_feature, target_feature

# class E3B(nn.Module):
#     def __init__(self, env, obs_shape=(64, 64, 3)):
#         super().__init__()
#         h, w, c = obs_shape
#         self.encoder = nn.Sequential(
#             layer_init(nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)),
#             nn.LeakyReLU(),
#             layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
#             nn.LeakyReLU(),
#             layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
#             nn.LeakyReLU(),
#             nn.Flatten(),
#             layer_init(nn.Linear(64*4*4, 512)),
#             nn.ReLU(),
#             layer_init(nn.Linear(512, 512)),
#             nn.ReLU(),
#             layer_init(nn.Linear(512, 512)),
#         )
#         self.idm = nn.Sequential(
#             layer_init(nn.Linear(512*2, 512)),
#             nn.ReLU(),
#             layer_init(nn.Linear(512, 512)),
#             nn.ReLU(),
#             layer_init(nn.Linear(512, env.action_space.n)),
#         )
#         self.cel = nn.CrossEntropyLoss()

#     def preprocess(self, x):
#         return rearrange(x, 'b h w c -> b c h w').float()/128.-1. # [0, 255] -> [-1, 1]

#     def encode(self, obs):
#         return self.encoder(self.preprocess(obs))
    
#     def idm_forward(self, obs, obs_next, action):
#         latent = self.encode(obs)
#         latent_next = self.encode(obs_next)
#         latent_cat = torch.cat([latent, latent_next], dim=-1)
#         logits = self.idm(latent_cat)
#         return self.cel(logits, action).mean()


# TODO:
"""
- make distillation network bigger
- smooth out dataset transitions
- use dagger style imitation learning
- 
"""