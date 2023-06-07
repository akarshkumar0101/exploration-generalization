import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat


class NatureBackbone(nn.Module):
    pass


class CNNAgent(nn.Module):
    def __init__(self, obs_shape, n_acts):
        super().__init__()
        t, c, h, w = obs_shape
        self.network = nn.Sequential(
            nn.Conv2d(t * c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.actor = nn.Linear(512, n_acts)
        self.critic = nn.Linear(512, 1)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

        self.apply(_init_weights)  # all layers
        torch.nn.init.orthogonal_(self.actor.weight, 0.01)
        torch.nn.init.orthogonal_(self.critic.weight, 1.0)

    def forward_temp(self, obs):
        # obs.shape: b, t, c, h, w

        # logits.shape: b, n_acts
        # values.shape: b
        x = rearrange(obs, "b t c h w -> b (t c) h w")
        x = self.network(x / 255.0)
        logits, values = self.actor(x), self.critic(x)
        return logits, values[..., 0]

    def calc_masked_obs(self, obs, done):
        b, t, c, h, w = obs.shape
        obs = obs.clone()
        for i_env in range(b):
            where = torch.where(done[i_env])[0]
            if len(where) > 0:
                i_step = where[-1].item()  # i_step of last done
                obs[i_env, :i_step] = obs[i_env, [i_step]]
        return obs

    def forward(self, obs, act=None, done=None):
        # obs.shape: b, t, c, h, w
        # act.shape: b, t
        # done.shape: b, t

        # logits.shape: b, t, n_acts
        # values.shape: b, t
        b, t, c, h, w = obs.shape
        obs = self.calc_masked_obs(obs, done)
        logits, values = self.forward_temp(obs)
        logits, values = repeat(logits, "b a -> b t a", t=t), repeat(values, "b -> b t", t=t)
        return torch.distributions.Categorical(logits=logits), values


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

    def forward(self, obs):
        x = self.network(obs / 255.0)
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1)
        return x


class RandomAgent(nn.Module):
    def __init__(self, n_acts):
        super().__init__()
        self.n_acts = n_acts

    def forward(self, obs, act=None, done=None):
        # obs.shape: b, t, c, h, w
        # act.shape: b, t
        # done.shape: b, t

        # logits.shape: b, t, n_acts
        # values.shape: b, t
        b, t, c, h, w = obs.shape
        logits = torch.zeros((b, t, self.n_acts), device=obs.device)
        values = torch.zeros((b, t), device=obs.device)
        return torch.distributions.Categorical(logits=logits), values


if __name__ == "__main__":
    import torchinfo

    # agent = Agent((4, 84, 84), 18)
    # torchinfo.summary(agent, input_size=(256, 4, 84, 84))

    encoder = Encoder((1, 84, 84), 128)
    torchinfo.summary(encoder, input_size=(256 * 3, 1, 84, 84))
