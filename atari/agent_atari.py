import numpy as np
import torch
from torch import nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def forward(self, x):
        return self.get_action_and_value(x)


class Encoder(nn.Module):
    def __init__(self, obs_shape, n_dim):
        super().__init__()
        c, h, w = obs_shape
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, n_dim), std=0.01),
        )

    def encode(self, x):
        latent = self.network(x / 255.0)
        latent = nn.functional.normalize(latent, dim=-1)
        return latent

    def forward(self, x):
        return self.encode(x)


if __name__ == "__main__":
    import torchinfo

    # agent = Agent((4, 84, 84), 18)
    # torchinfo.summary(agent, input_size=(256, 4, 84, 84))

    encoder = Encoder((1, 84, 84), 128)
    torchinfo.summary(encoder, input_size=(256 * 3, 1, 84, 84))
