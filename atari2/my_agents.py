import numpy as np
import torch
from einops import rearrange
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


class StackedCNNAgent(nn.Module):
    def __init__(self, n_acts=18, ctx_len=4):
        super().__init__()
        self.cnn = NatureCNN(c_in=ctx_len, n_dim=512)
        self.actor = layer_init(nn.Linear(512, n_acts), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.train_per_token = False
        self.ctx_len = ctx_len

    def forward(self, done, obs, act, rew):
        obs = rearrange(obs, "b t c h w -> b (t c) h w")
        hidden = self.cnn(obs)
        logits, val = self.actor(hidden), self.critic(hidden)[:, 0]
        logits, val = rearrange(logits, "b a -> b 1 a"), rearrange(val, "b -> b 1")
        return logits, val




# Adapted from nanoGPT's model.py

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


class LayerNorm(nn.LayerNorm):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias=True):
        super().__init__(ndim, elementwise_affine=True)
        if not bias:
            self.bias = None


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        assert self.flash
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.ctx_len * 3, config.ctx_len * 3)).view(1, 1, config.ctx_len * 3, config.ctx_len * 3))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# def create_mlp(config):
#     return nn.Sequential(
#         nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
#         nn.GELU(),
#         nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
#         nn.Dropout(config.dropout),
#     )


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class Config:
    n_acts: int
    ctx_len: int
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 4 * 64
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class TransformerAgent(nn.Module):
    def __init__(self, n_acts, ctx_len, n_layers=4, n_heads=4, n_embd=4 * 64, dropout=0.0, bias=True):
        super().__init__()
        self.train_per_token = True
        self.ctx_len = ctx_len

        self.config = Config(n_acts=n_acts, ctx_len=ctx_len, n_layers=n_layers, n_heads=n_heads, n_embd=n_embd, dropout=dropout, bias=bias)

        self.encode_step = nn.Embedding(self.config.ctx_len, self.config.n_embd)
        self.encode_obs = NatureCNN(1, self.config.n_embd)
        self.encode_rtg = nn.Sequential(nn.Linear(1, self.config.n_embd), nn.Tanh())
        # self.encode_act = nn.Sequential(nn.Embedding(config.n_acts, config.n_embd), nn.Tanh())
        self.encode_act = nn.Embedding(self.config.n_acts, self.config.n_embd)

        self.drop = nn.Dropout(self.config.dropout)
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layers)])
        self.ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias)
        self.actor = nn.Linear(self.config.n_embd, self.config.n_acts)
        self.critic = nn.Linear(self.config.n_embd, 1)

        # tie the weights of the action prediction head to the action embeddings
        self.actor.weight = self.encode_act.weight

        self.init_weights()

        self.last_token_train = False

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward_temp(self, obs, rtg, act, done=None):
        # obs: (b, t, c, 84, 84)
        # rtg: (b, t)
        # act: (b, t)
        import timers

        timer = timers.Timer()

        batch_size, ctx_len, _, _, _ = obs.shape
        assert ctx_len <= self.config.ctx_len, f"Sequence too long"
        if rtg is None:
            rtg = torch.zeros(batch_size, ctx_len, dtype=torch.float32, device=obs.device)  # (batch_size, ctx_len)

        i_step = torch.arange(0, ctx_len, dtype=torch.long, device=obs.device)  # (ctx_len, )
        x_step = self.encode_step(i_step)  # (ctx_len, n_embd)

        # rtg = rearrange(rtg, "b t -> (b t) 1")
        obs = rearrange(obs, "b t c h w -> (b t) c h w")
        act = rearrange(act, "b t -> (b t)")

        # with timer.add_time("embed_rtg"):
        # x_rtg = self.encode_rtg(rtg)  # (batch_size * n_steps, n_embd)
        with timer.add_time("embed_obs"):
            x_obs = self.encode_obs(obs)  # (batch_size * n_steps, n_embd)
        with timer.add_time("embed_act"):
            x_act = self.encode_act(act)  # (batch_size * n_steps, n_embd)

        # x_rtg = x_step + rearrange(x_rtg, "(b t) d -> b t d", b=batch_size)  # (batch_size, n_steps, n_embd)
        x_obs = x_step + rearrange(x_obs, "(b t) d -> b t d", b=batch_size)  # (batch_size, n_steps, n_embd)
        x_act = x_step + rearrange(x_act, "(b t) d -> b t d", b=batch_size)  # (batch_size, n_steps, n_embd)

        # x = torch.stack([x_rtg, x_obs, x_act], dim=-2)  # (batch_size, n_steps, 3, n_embd)
        x = torch.stack([x_obs, x_act], dim=-2)  # (batch_size, n_steps, 3, n_embd)
        x = rearrange(x, "b t c d -> b (t c) d")  # (batch_size, n_steps * 3, n_embd)

        # assert torch.allclose(x[:, 0::3, :], x_rtg)  # sanity check, TODO: remove
        # assert torch.allclose(x[:, 1::3, :], x_obs)  # sanity check, TODO: remove
        # assert torch.allclose(x[:, 2::3, :], x_act)  # sanity check, TODO: remove

        x = self.drop(x)
        with timer.add_time("blocks"):
            x = self.blocks(x)
        x = self.ln_f(x)

        x = rearrange(x, "b (t c) d -> b t c d", t=ctx_len)  # (batch_size, n_steps, 3, n_embd)
        x = x[:, :, 0, :]  # (batch_size, n_steps, n_embd) - only keep the obs tokens for predicting the next action

        with timer.add_time("out_heads"):
            logits, values = self.actor(x), self.critic(x)  # (batch_size, n_steps, n_acts), (batch_size, n_steps, 1)
        # print(dict(timer.key2time))
        return logits, values[..., 0]

    def forward(self, done, obs, act, rew):
        # done.shape: b, t
        # obs.shape: b, t, c, h, w
        # act.shape: b, t (or b, t-1)
        # rew.shape: b, t

        # logits.shape: b, t, n_acts
        # values.shape: b, t
        # print the obs, act, done shape and device, and dtype
        if act.shape[1] == obs.shape[1] - 1:  # pad action
            noaction = act[:, [-1]].clone()
            act = torch.cat([act, noaction], dim=-1)
        # mask = self.create_mask(done, toks=3)
        logits, values = self.forward_temp(rtg=None, obs=obs, act=act)
        return logits, values

    def create_mask(self, done, toks=1):
        t, b = done.shape
        mask = torch.ones(b, t * toks, t * toks, dtype=torch.bool, device=done.device).tril()
        for i in range(t):
            # when done=True, me+future (i:) CANNOT attend to past (:i)
            mask[done[i], i * toks :, : i * toks] = False
        return mask

    def create_optimizer(self, lr, weight_decay=0, betas=(0.9, 0.95), device=None, **kwargs):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [{"params": decay_params, "weight_decay": weight_decay}, {"params": nodecay_params, "weight_decay": 0.0}]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer


# TODO proper masking

# TODO things to consider:
# - Which parameters experience weight decay? (ex. all weight matrices, but not biases or layernorms)
# - How to initialize each parameter? (ex. weights from N(0,0.02), biases to 0, layernorms to 1)
# - Tanh after embedding the rtg, obs, act?


def make_agent(model, n_acts=18):
    model, ctx_len = model.split("_")
    ctx_len = int(ctx_len)
    if model == "random":
        return RandomAgent(n_acts, ctx_len)
    elif model == "cnn":
        return StackedCNNAgent(n_acts, ctx_len)
    elif model == "trans":
        return TransformerAgent(n_acts, ctx_len, n_layers=4, n_heads=4, n_embd=4 * 64, dropout=0.0, bias=True)
    else:
        raise ValueError(f"Unknown model name {model}")

class ConcatAgent(nn.Module):
    def __init__(self, agents):
        super().__init__()
        self.agents = agents
        self.train_per_token = self.agents[0].train_per_token
        self.ctx_len = self.agents[0].ctx_len

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

if __name__ == "__main__":
    import torchinfo

    for model in ['cnn', 'trans']:
        for ctx_len in [4, 8]:
            print(f"{model}_{ctx_len}")
            agent = make_agent(f"{model}_{ctx_len}")
            n_params = sum(p.numel() for p in agent.parameters())
            print(f'{n_params=}')

            batch_size = 256
            torchinfo.summary(
                agent,
                input_size=[
                    (batch_size, ctx_len),
                    (batch_size, ctx_len, 1, 84, 84),
                    (batch_size, ctx_len),
                    (batch_size, ctx_len),
                ],
                dtypes=[torch.bool, torch.float, torch.long, torch.float],
            )

            done = torch.randn(batch_size, ctx_len)<0
            obs = torch.randn(batch_size, ctx_len, 1, 84, 84)
            act = torch.randint(0, 18, (batch_size, ctx_len))
            rew = torch.randn(batch_size, ctx_len)

            logits, vals = agent(done, obs, act, rew)
            assert logits.shape == (batch_size, ctx_len, 18) or logits.shape == (batch_size, 1, 18) 
            assert vals.shape == (batch_size, ctx_len) or vals.shape == (batch_size, 1)
            