import numpy as np
import torch
from agent_atari import DecisionTransformer, NatureCNNAgent, RandomAgent


def create_agent(model, n_acts, ctx_len, load_agent=None, device=None):
    if model == "cnn":
        agent = NatureCNNAgent(n_acts, ctx_len)
    elif model == "gpt":
        agent = DecisionTransformer(n_acts, ctx_len)
    elif model == "rand":
        agent = RandomAgent(n_acts)
    agent = agent.to(device)
    if load_agent is not None:
        try:
            agent.load_state_dict(torch.load(load_agent, map_location=device))
        except RuntimeError as e:
            print(f"----------------------------------------------------")
            print(f"WARNING: UNABLE TO LOAD AGENT FROM {load_agent}...")
            print(f"----------------------------------------------------")
    return agent


def get_lr(lr, lr_min, i_iter, n_iters, warmup=True, decay="none"):
    assert i_iter <= n_iters
    n_warmup = n_iters // 100
    if i_iter <= n_warmup:
        return lr * i_iter / n_warmup if warmup and n_warmup > 0 else lr
    elif i_iter <= n_iters:
        decay_ratio = (i_iter - n_warmup) / (n_iters - n_warmup)
        if decay is None or decay == "none":
            return lr
        elif decay == "linear":
            coeff = 1.0 - decay_ratio
            return lr_min + coeff * (lr - lr_min)
        elif decay == "cos":
            coeff = 0.5 * (1.0 + np.math.cos(np.pi * decay_ratio))  # coeff ranges 0..1
            assert 0 <= decay_ratio <= 1 and 0 <= coeff <= 1
            return lr_min + coeff * (lr - lr_min)
        else:
            raise ValueError(f"Unknown decay type {decay}")


def calc_ppo_policy_loss(dist, dist_old, act, adv, norm_adv=True, clip_coef=0.1):
    # can be called with dist or logits
    if isinstance(dist, torch.Tensor):
        dist = torch.distributions.Categorical(logits=dist)
    if isinstance(dist_old, torch.Tensor):
        dist_old = torch.distributions.Categorical(logits=dist_old)

    if norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = (dist.log_prob(act) - dist_old.log_prob(act)).exp()
    loss_pg1 = -adv * ratio
    loss_pg2 = -adv * ratio.clamp(1 - clip_coef, 1 + clip_coef)
    loss_pg = torch.max(loss_pg1, loss_pg2)
    return loss_pg


def calc_ppo_value_loss(val, val_old, ret, clip_coef=0.1):
    if clip_coef is not None:
        loss_v_unclipped = (val - ret) ** 2
        v_clipped = val_old + (val - val_old).clamp(-clip_coef, clip_coef)
        loss_v_clipped = (v_clipped - ret) ** 2
        loss_v_max = torch.max(loss_v_unclipped, loss_v_clipped)
        loss_v = 0.5 * loss_v_max
    else:
        loss_v = 0.5 * ((val - ret) ** 2)
    return loss_v


def calc_klbc_loss(dist_student, dist_teacher):
    return torch.nn.functional.kl_div(dist_student.logits, dist_teacher.logits, log_target=True, reduction="none")


def calc_rnd_loss(rnd_student, rnd_teacher):
    return (rnd_student - rnd_teacher.detach()).pow(2).mean(dim=-1)


def calc_idm_loss(logits, actions):
    return torch.nn.functional.cross_entropy(logits, actions, reduction="none")


def index_dict(data, fn_index):
    def index_v(v):
        if isinstance(v, torch.Tensor):
            return fn_index(v)
        elif isinstance(v, torch.distributions.Categorical):
            return torch.distributions.Categorical(logits=fn_index(v.logits))
        else:
            raise NotImplementedError

    return {k: index_v(v) for k, v in data.items()}
