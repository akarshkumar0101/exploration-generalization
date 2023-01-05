# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange, repeat
from torch.distributions.categorical import Categorical

import utils


def parse_args(envs, s=None):
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    # parser.add_argument("--num-steps", type=int, default=128,
    parser.add_argument("--num-steps", type=int, default=11,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    args, uargs = parser.parse_known_args(s)

    args.num_envs = envs.num_envs
    args.batch_size = int(envs.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def calc_gae(rewards, values, gamma=0.99, gae_lambda=0.95):
    """
    Generalized Advantage Estimation
    rewards, values should have shape (n_trajs, len_traj)
    Return value:
     - advantages, returns of shape (n_trajs, len_traj-1)
    """
    n_trajs, len_traj = rewards.shape
    t = torch.arange(len_traj).expand(n_trajs, -1)

    values_now, values_next = values[..., :-1], values[..., 1:]
    rewards_now, rewards_next = rewards[..., :-1], rewards[..., 1:]
    t_now, t_next = t[..., :-1], t[..., 1:]

    # Compute temporal difference residual (TDR)
    # shape: (n_trajs, len_traj-1)
    tdr_now = (rewards_now + gamma*values_next) - values_now
    def reverse_cumsum(x, dim=0): # cumsum from the end
        return x + x.sum(dim=dim, keepdims=True) - torch.cumsum(x, dim=dim)

    # INCORRECT
    # advantages = ((gamma*gae_lambda)**t_now) * tdr_now
    # advantages = reverse_cumsum(advantages, dim=-1)

    # CORRECT single loop
    advantages = torch.zeros_like(tdr_now)
    for i in range(0, len_traj-1):
        l = torch.arange(0, len_traj-1-i)
        advantages[:, i] = (((gamma*gae_lambda)**l) * tdr_now[:, i:]).sum(dim=-1)

    # CORRECT double loop
    # adv = torch.zeros_like(tdr_now)
    # for i in range(0, len_traj-1):
    #     for l in range(0, len_traj-1):
    #         if i+l>=len_traj-1:
    #             break
    #         adv[:, i] += (gamma*gae_lambda)**(l) * tdr_now[:, i+l]
        
    returns = advantages + values_now
    return advantages, returns

def calc_gae(reward, value, next_value, done, next_done, gamma=0.99, gae_lambda=0.95):
    """
    Generalized Advantage Estimation
    Inputs:
        - reward of shape (num_envs, num_steps)
        -  value of shape (num_envs, num_steps)
        -  next_value of shape (num_envs, )
        -   done of shape (num_envs, num_steps)
        -   next_done of shape (num_envs, )
    Returns:
        - advantage of shape (num_envs, num_steps)
        -    return of shape (num_envs, num_steps)
    """
    # bootstrap value if not done
    num_envs, num_steps = reward.shape
    with torch.no_grad():
        advantage = torch.zeros_like(reward)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - done[:, t + 1]
                nextvalues = value[:, t + 1]
            delta = reward[:, t] + gamma * nextvalues * nextnonterminal - value[:, t]
            advantage[:, t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        return_ = advantage + value
    return advantage, return_

def run_ppo_simple(agent, envs, args, callback_fn=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # if not isinstance(envs.observation_space, gym.spaces.Dict):
    #     envs = utils.DictObservationWrapper(envs)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    data = {}
    # observation space is a dict
    obs_keys = list(envs.observation_space.keys())
    for k, v in envs.observation_space.items():
        data[k] = torch.zeros((args.num_envs, args.num_steps) + v.shape).to(device)
    data[ 'action'] = torch.zeros((args.num_envs, args.num_steps)+envs.single_action_space.shape).to(device)
    data['logprob'] = torch.zeros((args.num_envs, args.num_steps)).to(device)
    data[ 'reward'] = torch.zeros((args.num_envs, args.num_steps)).to(device)
    data['entropy'] = torch.zeros((args.num_envs, args.num_steps)).to(device)
    data[  'value'] = torch.zeros((args.num_envs, args.num_steps)).to(device)
    data[   'done'] = torch.zeros((args.num_envs, args.num_steps)).to(device)

    num_updates = args.total_timesteps // args.batch_size
    for update in range(num_updates):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            lr_now = (1.0 - update/num_updates) * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now
        
        obs, info = envs.reset()
        done = torch.zeros(args.num_envs).to(device)
        for i_step in range(args.num_steps):
            for k, v in obs.items():
                data[k][:, i_step] = v
            data['done'][:, i_step] = done

            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(obs)
            obs, reward, terminated, truncated, info = envs.step(action)
            done = torch.logical_or(terminated, truncated).float()

            data[ 'action'][:, i_step] = action
            data['logprob'][:, i_step] = logprob
            data[ 'reward'][:, i_step] = reward
            data['entropy'][:, i_step] = entropy
            data[  'value'][:, i_step] = value

        value = agent.get_value(obs)
        advantage, return_ = calc_gae(data['reward'], data['value'], value, data['done'], done, args.gamma, args.gae_lambda)
        data['advantage'] = advantage
        data['return'] = return_

        # flatten the batch
        b_data = {k: rearrange(v, 'e t ... -> (e t) ...') for k, v in data.items()}

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            for mb_inds in torch.randperm(args.batch_size).split(args.minibatch_size):
                mb_data = {k: v[mb_inds] for k, v in b_data.items()}

                mb_obs_items = {k: mb_data[k] for k in obs_keys}
                mb_advantage = mb_data['advantage']
                mb_value = mb_data['value']
                mb_return = mb_data['return']
                mb_logprob = mb_data['logprob']
                mb_action = mb_data['action']

                _, mb_logprobs_new, mb_entropies_new, mb_values_new = agent.get_action_and_value(**mb_obs_items, action=mb_action.long())

                logratio = mb_logprobs_new - mb_logprob
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.norm_adv:
                    mb_advantage = (mb_advantage - mb_advantage.mean()) / (mb_advantage.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantage * ratio
                pg_loss2 = -mb_advantage * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                mb_values_new = mb_values_new.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (mb_values_new - mb_return) ** 2
                    v_clipped = mb_value + (mb_values_new - mb_value).clamp(-args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - mb_return) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((mb_values_new - mb_return) ** 2).mean()

                entropy_loss = mb_entropies_new.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_data['value'].cpu().numpy(), b_data['return'].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if callback_fn is not None:
            callback_fn(**locals())

    envs.close()
