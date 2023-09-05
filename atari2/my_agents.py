import numpy as np
import torch
from einops import rearrange

# from normalize import RunningMeanStd
from torch import nn


class RandomAgent(nn.Module):
    def __init__(self, n_acts, ctx_len=4):
        super().__init__()
        self.n_acts = n_acts
        self.train_per_token = False
        self.ctx_len = ctx_len

    def forward(self, done, obs, act, rew):
        b, t, c, h, w = obs.shape
        logits = torch.zeros((b, 1, self.n_acts), device=obs.device)
        val = torch.zeros((b, 1), device=obs.device)
        return logits, val


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class StackedCNNAgent(nn.Module):
    def __init__(self, n_acts, ctx_len=4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_acts), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.train_per_token = False
        self.ctx_len = ctx_len

    def forward(self, done, obs, act, rew):
        obs = rearrange(obs, "b t c h w -> b (t c) h w")
        hidden = self.network(obs / 255.0)
        logits, val = self.actor(hidden), self.critic(hidden)[:, 0]
        logits, val = rearrange(logits, "b a -> b 1 a"), rearrange(val, "b -> b 1")
        return logits, val


class ConcatAgent(nn.Module):
    def __init__(self, agents):
        super().__init__()
        self.agents = agents

    def forward(self, done, obs, act, rew):
        nb, t, c, h, w = obs.shape
        assert nb % len(self.agents) == 0
        done = rearrange(done, "(n b) ... -> n b ...", n=len(self.agents))
        obs = rearrange(obs, "(n b) ... -> n b ...", n=len(self.agents))
        act = rearrange(act, "(n b) ... -> n b ...", n=len(self.agents))
        rew = rearrange(rew, "(n b) ... -> n b ...", n=len(self.agents))
        logitss, valuess = zip(*[agent(donei, obsi, acti, rewi) for agent, donei, obsi, acti, rewi in zip(self.agents, done, obs, act, rew)])
        logits, values = rearrange(list(logitss), "n b ... -> (n b) ..."), rearrange(list(valuess), "n b ... -> (n b) ...")
        return logits, values


def make_agent(model, n_acts=18):
    model, ctx_len = model.split("_")
    ctx_len = int(ctx_len)
    if model == "random":
        return RandomAgent(n_acts, ctx_len)
    elif model == "cnn":
        return StackedCNNAgent(n_acts, ctx_len)
    elif model == "transformer":
        return None
    else:
        raise ValueError(f"Unknown model name {model}")
