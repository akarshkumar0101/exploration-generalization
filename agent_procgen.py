
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                              padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class Agent(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        h, w, c = obs_shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(256, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x.permute((0, 3, 1, 2)) / 255.0))  # "bhwc" -> "bchw"

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), logits
        return action, None, probs.entropy(), self.critic(hidden), logits

class AgentLSTM(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        h, w, c = obs_shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.lstm = nn.LSTM(256, 256)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(256, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)
        self.ignore_lstm = False
        self.no_recurrence = False

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        if self.ignore_lstm:
            return hidden, lstm_state
        if self.no_recurrence:
            done = torch.ones_like(done)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state

class IDM(nn.Module):
    def __init__(self, obs_shape, n_actions, n_features=100):
        super().__init__()
        h, w, c = obs_shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            # nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=n_features),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        
        self.idm = layer_init(nn.Linear(2*n_features, n_actions), std=0.01)
        # self.idm = nn.Sequential(
        #     nn.Linear(2*n_features, n_features),
        #     nn.ReLU(),
        #     nn.Linear(n_features, n_features),
        #     nn.ReLU(),
        #     nn.Linear(n_features, n_actions),
        # )

    def calc_features(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        return hidden
    
    def forward(self, obs, next_obs):
        l1 = self.calc_features(obs)
        l2 = self.calc_features(next_obs)
        l = torch.cat([l1, l2], dim=-1)
        logits = self.idm(l)
        return logits

class E3B(nn.Module):
    def __init__(self, num_envs, obs_shape, n_actions, n_features=100, lmbda=0.1):
        super().__init__()
        self.idm = IDM(obs_shape, n_actions, n_features)

        self.Il = nn.Parameter(torch.eye(n_features)/lmbda) # d, d
        self.Cinv = nn.Parameter(torch.zeros(num_envs, 100, 100)) # b, d, d
        self.Il.requires_grad_(False)
        self.Cinv.requires_grad_(False)
        
        self.Cinv[:] = self.Il

    @torch.no_grad() # this is required
    def calc_reward(self, obs, done=None):
        """
        obs should be of shape (n_envs, *obs_shape)
        done should be of shape (n_envs, ) signaling if this obs is for a new episode
        """
        if done is not None:
            assert done.dtype == torch.bool
            self.Cinv[done] = self.Il
        v = self.idm.calc_features(obs)[..., :, None] # b, d, 1
        u = self.Cinv @ v # b, d, 1
        b = v.mT @ u # b, 1, 1
        self.Cinv[:] = self.Cinv - u@u.mT/(1. + b) # b, d, d
        rew_eps = b[..., 0, 0].detach()
        return rew_eps
