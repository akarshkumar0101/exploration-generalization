import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
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
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
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
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), logits


class AgentLSTM(nn.Module):
    def __init__(self, obs_shape, n_actions, lstm_type="lstm"):
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
        self.lstm_type = lstm_type

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        if "ignore" in self.lstm_type:
            return hidden, lstm_state
        if "norecurrence" in self.lstm_type:
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
        if "residual" in self.lstm_type:
            new_hidden = new_hidden + hidden.reshape(new_hidden.shape)
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
    def __init__(self, obs_shape, n_actions, n_features=100, merge="cat"):
        super().__init__()
        self.n_features = n_features
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

        # self.idm = layer_init(nn.Linear(2*n_features, n_actions), std=0.01)
        self.merge = merge
        n_inputs_idm = 2 * n_features if merge == "cat" else n_features
        self.idm = nn.Sequential(
            nn.Linear(n_inputs_idm, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_actions),
        )

    def calc_features(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        return hidden

    def forward(self, obs, next_obs):
        l1 = self.calc_features(obs)
        l2 = self.calc_features(next_obs)
        l = torch.cat([l1, l2], dim=-1) if self.merge == "cat" else l2 - l1
        logits = self.idm(l)
        return logits

    def forward_smart(self, obs):  # obs has shape: (t n h w c)
        x = rearrange(obs, "t n h w c -> (t n) c h w") / 255.0
        hidden = self.network(x) # (t n) d
        hidden = rearrange(hidden, "(t n) d -> t n d", t=len(obs)) # t n d
        l1, l2 = hidden[:-1], hidden[1:] # t-1 n d
        l = torch.cat([l1, l2], dim=-1) if self.merge == "cat" else l2 - l1
        logits = self.idm(l)
        return logits # t-1 n a

