import numpy as np
import torch
from einops import rearrange
from normalize import RunningMeanStd
from torch import nn


class NoopAgent(nn.Module):
    def __init__(self, n_acts, ctx_len=None, logit=4.0):
        super().__init__()
        self.n_acts = n_acts
        self.last_token_train = True
        self.logit = logit

    def forward(self, done, obs, act, rew):
        b, t, c, h, w = obs.shape
        logits = torch.zeros((b, 1, self.n_acts), device=obs.device)
        values = torch.zeros((b, 1), device=obs.device)
        logits[:, :, 0] = self.logit
        return logits, values


class RandomAgent(nn.Module):
    def __init__(self, n_acts, ctx_len=None):
        super().__init__()
        self.n_acts = n_acts
        self.last_token_train = True

    def forward(self, done, obs, act, rew):
        print(f'done: {done.shape}, obs: {obs.shape}, act: {act.shape}, rew: {rew.shape}')
        b, t, c, h, w = obs.shape
        logits = torch.zeros((b, 1, self.n_acts), device=obs.device)
        values = torch.zeros((b, 1), device=obs.device)
        return logits, values


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    def __init__(self, c_in, n_dim, normalize=False):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(c_in, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, n_dim)),
            nn.ReLU(),
        )
        self.normalize = normalize

    def forward(self, x):
        x = self.network(x / 255.0 * 2.0 - 1.0)
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1)
        return x


class IDM(nn.Module):
    def __init__(self, n_acts, n_dim=512, normalize=True):
        super().__init__()
        self.encoder = NatureCNN(1, n_dim, normalize=normalize)
        self.idm = nn.Sequential(
            layer_init(nn.Linear(n_dim * 2, n_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(n_dim, n_acts), std=0.01),
        )

    def forward(self, obs):
        return self.encoder(obs)

    def predict_action(self, obs, next_obs):
        l1, l2 = self.encoder(obs), self.encoder(next_obs)
        return self.idm(torch.cat([l1, l2], dim=-1))


class NatureCNNAgent(nn.Module):
    def __init__(self, n_acts, ctx_len, n_dim=512):
        super().__init__()
        self.encode_obs = NatureCNN(ctx_len, n_dim)
        self.actor = layer_init(nn.Linear(n_dim, n_acts), std=0.01)
        self.critic = layer_init(nn.Linear(n_dim, 1), std=1)
        self.last_token_train = True

    def forward(self, done, obs, act, rew):
        print(f'done: {done.shape}, obs: {obs.shape}, act: {act.shape}, rew: {rew.shape}')
        b, t, c, h, w = obs.shape
        x = rearrange(obs, "b t c h w -> b (t c) h w")
        hidden = self.encode_obs(x)
        logits, values = self.actor(hidden), self.critic(hidden)
        logits, values = rearrange(logits, "b a -> b 1 a"), rearrange(values, "b 1 -> b 1")
        return logits, values

    def create_optimizer(self, lr, weight_decay=0, betas=(0.9, 0.999), eps=1e-5, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

    # def init_weights(self):
    #     def _init_weights(m):
    #         if isinstance(m, (nn.Linear, nn.Conv2d)):
    #             torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
    #             torch.nn.init.zeros_(m.bias)

    #     self.apply(_init_weights)  # all layers
    #     torch.nn.init.orthogonal_(self.actor.weight, 0.01)
    #     torch.nn.init.orthogonal_(self.critic.weight, 1.0)

    # def calc_masked_obs(self, obs, done):
    #     b, t, c, h, w = obs.shape
    #     obs = obs.clone()
    #     for i_env in range(b):
    #         where = torch.where(done[i_env])[0]
    #         if len(where) > 0:
    #             i_step = where[-1].item()  # i_step of last done
    #             obs[i_env, :i_step] = obs[i_env, [i_step]]
    #     return obs


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


class RNDModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

        self.rms_obs = RunningMeanStd()

    def forward(self, obs, update_rms_obs=True):
        # obs: (b, c, h, w) = (b, 1, 84, 84)
        if update_rms_obs:
            self.rms_obs.update(obs)
        obs = self.rms_obs.normalize(obs)
        return self.predictor(obs), self.target(obs)


if __name__ == "__main__":
    agent1 = NatureCNNAgent(18, 4)
    agent2 = DecisionTransformer(18, 4)

    obs = torch.randn(16, 4, 1, 84, 84)
    act = torch.randint(0, 18, (16, 4))

    logits, values = agent1(done=None, obs=obs, act=act, rew=None)
    print(logits.shape, values.shape)
    logits, values = agent2(done=None, obs=obs, act=act, rew=None)
    print(logits.shape, values.shape)

from decision_transformer import DecisionTransformer
