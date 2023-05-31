import numpy as np
import torch
from torch import nn
from einops import rearrange


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape
        self.network = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

        self.apply(_init_weights)  # all layers
        torch.nn.init.orthogonal_(self.actor.weight, 0.01)
        torch.nn.init.orthogonal_(self.critic.weight, 1.0)

    def forward(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits, values = self.actor(hidden), self.critic(hidden)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, values[..., 0]

    def act(self, obs, act=None, done=None):
        """
        obs: (n_batch, n_steps, *obs_shape)
        act: (n_batch, n_steps-1)
        done: (n_batch, n_steps)
        """
        obs = self.get_obs(obs, done)
        return self.forward(obs)

    def get_obs(self, obs, done):
        b, t, c, h, w = obs.shape
        obs = obs.clone()
        for i_step in range(t):
            obs[done[:, i_step]][:, :i_step] = obs[done[:, i_step]][:, [i_step]]
        obs = rearrange(obs, "b t c h w -> b (t c) h w")
        return obs


class Encoder(nn.Module):
    def __init__(self, obs_shape, n_embd, normalize=True):
        super().__init__()
        c, h, w = obs_shape
        self.normalize = normalize
        self.network = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_embd),
        )

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

        self.apply(_init_weights)  # all layers

    def forward(self, x):
        latent = self.network(x / 255.0)
        if self.normalize:
            latent = nn.functional.normalize(latent, dim=-1)
        return latent


if __name__ == "__main__":
    import torchinfo

    # agent = Agent((4, 84, 84), 18)
    # torchinfo.summary(agent, input_size=(256, 4, 84, 84))

    encoder = Encoder((1, 84, 84), 128)
    torchinfo.summary(encoder, input_size=(256 * 3, 1, 84, 84))


class RandomAgent(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def forward(self, obs):
        return torch.distributions.Categorical(logits=torch.zeros(obs.shape[0], self.n_actions)), torch.zeros(obs.shape[0])

    def act(self, obs, act=None, done=None):
        """
        obs: (n_batch, n_steps, *obs_shape)
        act: (n_batch, n_steps-1)
        done: (n_batch, n_steps)
        """
        dist = torch.distributions.Categorical(logits=torch.zeros(obs.shape[0], self.n_actions, device=obs.device))
        value = torch.zeros(obs.shape[0], device=obs.device)
        return dist, value
