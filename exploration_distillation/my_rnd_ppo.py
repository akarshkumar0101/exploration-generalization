# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from gym.wrappers.normalize import RunningMeanStd
from to_tensor import ToTensor
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

def calc_gae(reward, value, next_value, done, next_done, gamma=0.99, gae_lambda=0.95, ext=True):
    """
    Generalized Advantage Estimation
    Inputs:
        -     reward: (n_envs, n_steps)
        -      value: (n_envs, n_steps)
        - next_value: (n_envs, )
        -       done: (n_envs, n_steps)
        -  next_done: (n_envs, )
    Returns:
        -  advantage: (n_envs, n_steps)
        -     return: (n_envs, n_steps)
    """
    # bootstrap value if not done
    n_envs, n_steps = reward.shape
    with torch.no_grad():
        advantage = torch.zeros_like(reward)
        lastgaelam = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = 1.0 - next_done if ext else 1.0
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - done[:, t + 1] if ext else 1.0
                nextvalues = value[:, t + 1]
            delta = reward[:, t] + gamma * nextvalues * nextnonterminal - value[:, t]
            advantage[:, t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        return_ = advantage + value
    return advantage, return_

def run_rndppo(env, agent, rnd_model, 
               total_timesteps=32000000, lr=1e-4, n_steps=256, anneal_lr=True,
               gamma=0.999, gae_lambda=0.95, minibatch_size=2048, update_epochs=3,
               norm_adv=True, clip_coef=0.1, clip_vloss=True,
               ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=None,
               update_proportion=0.25, int_coef=1.0, ext_coef=2.0, int_gamma=0.99,
               num_iterations_obs_norm_init=1,
               device=None, tqdm=None, callback_fn=None):

    env = ToTensor(env, dtype=torch.float32, device=device)
    n_envs = env.num_envs
    batch_size = n_envs*n_steps

    assert isinstance(env.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    agent.to(device)
    rnd_model.to(device)
    params = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    opt = optim.Adam(params, lr=lr, eps=1e-5)

    rms_reward = RunningMeanStd()
    rms_obs = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(int_gamma)

    data = {}
    data[      'obs'] = torch.zeros((n_envs, n_steps)+env.single_observation_space.shape, device=device)
    data[   'action'] = torch.zeros((n_envs, n_steps)+env.single_action_space.shape, device=device)
    data[  'logprob'] = torch.zeros((n_envs, n_steps), device=device)
    data[  'rew_ext'] = torch.zeros((n_envs, n_steps), device=device)
    data[  'rew_int'] = torch.zeros((n_envs, n_steps), device=device)
    data[  'entropy'] = torch.zeros((n_envs, n_steps), device=device)
    data['value_ext'] = torch.zeros((n_envs, n_steps), device=device)
    data['value_int'] = torch.zeros((n_envs, n_steps), device=device)
    data[     'done'] = torch.zeros((n_envs, n_steps), device=device)

    obs, info = env.reset()
    done = torch.zeros(n_envs, device=device)

    print("Start to initialize observation normalization parameter.....")
    next_ob = []
    pbar = range(num_iterations_obs_norm_init*n_steps)
    if tqdm is not None: pbar = tqdm(pbar)
    for i_step in pbar:
        s, _, _, _, _ = env.step(env.action_space.sample())
        next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()

        if len(next_ob) % (n_steps * n_envs) == 0:
            next_ob = np.stack(next_ob)
            rms_obs.update(next_ob)
            next_ob = []
    print("End to initialize...")

    obs, info = env.reset()
    done = torch.zeros(n_envs, device=device)

    n_updates = total_timesteps // batch_size
    pbar = range(n_updates)
    if tqdm is not None: pbar = tqdm(pbar)
    for i_update in pbar:
        if anneal_lr:
            lr_now = (1.0 - i_update/n_updates) * lr
            opt.param_groups[0]["lr"] = lr_now

        for i_step in range(n_steps):
            data[      'obs'][:, i_step] = obs
            data[     'done'][:, i_step] = done

            with torch.no_grad():
                dist, value_ext, value_int = agent.get_dist_and_values(obs)
                action = dist.sample()
                logprob, entropy = dist.log_prob(action), dist.entropy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = torch.logical_or(terminated, truncated).float()

            data[   'action'][:, i_step] = action
            data[  'logprob'][:, i_step] = logprob
            data[  'entropy'][:, i_step] = entropy
            data[  'rew_ext'][:, i_step] = reward
            data['value_ext'][:, i_step] = value_ext
            data['value_int'][:, i_step] = value_int

            # FIX THIS -------------------------
            rnd_next_obs = (
                (
                    (obs[:, 3, :, :].reshape(n_envs, 1, 84, 84) - torch.from_numpy(rms_obs.mean).to(device))
                    / torch.sqrt(torch.from_numpy(rms_obs.var).to(device))
                ).clip(-5, 5)
            ).float()
            target_next_feature = rnd_model.target(rnd_next_obs)
            predict_next_feature = rnd_model.predictor(rnd_next_obs)
            data['rew_int'][:, i_step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data
            # FIX THIS -------------------------


        curiosity_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in data['rew_int'].cpu().numpy()])
        mean = np.mean(curiosity_reward_per_env)
        std = np.std(curiosity_reward_per_env)
        count = len(curiosity_reward_per_env)
        rms_reward.update_from_moments(mean, std**2, count)
        data['rew_int'] /= np.sqrt(rms_reward.var)

        with torch.no_grad():
            dist, value_ext, value_int = agent.get_dist_and_values(obs)
        data['adv_ext'], data['return_ext'] = calc_gae(data['rew_ext'], data['value_ext'], value_ext, data['done'], done,     gamma, gae_lambda, ext=True)
        data['adv_int'], data['return_int'] = calc_gae(data['rew_int'], data['value_int'], value_int, data['done'], done, int_gamma, gae_lambda, ext=False)
        # combined advantages
        data['adv'] = data['adv_ext'] * ext_coef + data['adv_int'] * int_coef

        # flatten the batch
        b_data = {k: rearrange(v, 'e t ... -> (e t) ...') for k, v in data.items()}

        # FIX THIS -------------------------
        b_obs = b_data['obs']
        rms_obs.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())
        rnd_next_obs = (
            (
                (b_obs[:, 3, :, :].reshape(-1, 1, 84, 84) - torch.from_numpy(rms_obs.mean).to(device))
                / torch.sqrt(torch.from_numpy(rms_obs.var).to(device))
            ).clip(-5, 5)
        ).float()
        # FIX THIS -------------------------

        clipfracs = []
        for i_epoch in range(update_epochs):
            for mb_inds in torch.randperm(batch_size).split(minibatch_size):
                mb_data = {k: v[mb_inds] for k, v in b_data.items()}
                mb_obs = mb_data['obs']
                mb_logprob = mb_data['logprob']
                mb_action = mb_data['action']
                mb_adv = mb_data['adv']
                mb_value_ext = mb_data['value_ext']
                mb_value_int = mb_data['value_int']
                mb_return_ext = mb_data['return_ext']
                mb_return_int = mb_data['return_int']

                ##### ------------------------------ FIX THIS
                predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                forward_loss = F.mse_loss(predict_next_state_feature, target_next_state_feature.detach(), reduction="none").mean(-1)
                mask = torch.rand(len(forward_loss), device=device)
                mask = (mask < update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )
                ##### ------------------------------ FIX THIS

                mb_dist_new, mb_value_ext_new, mb_value_int_new = agent.get_dist_and_values(mb_obs)
                mb_logprob_new = mb_dist_new.log_prob(mb_action)
                mb_entropy_new = mb_dist_new.entropy()
                logratio = mb_logprob_new - mb_logprob
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                if norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1-clip_coef, 1+clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                # Value loss
                if clip_vloss:
                    ext_v_loss_unclipped = (mb_value_ext_new - mb_return_ext) ** 2
                    ext_v_clipped = mb_value_ext + (mb_value_ext_new - mb_value_ext).clamp(-clip_coef, clip_coef)
                    ext_v_loss_clipped = (ext_v_clipped - mb_return_ext) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((mb_value_ext_new - mb_return_ext) ** 2).mean()
                int_v_loss = 0.5 * ((mb_value_int_new - mb_return_int) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss
                entropy_loss = mb_entropy_new.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef + forward_loss

                # Optimize
                opt.zero_grad()
                loss.backward()
                if max_grad_norm:
                    nn.utils.clip_grad_norm_(params, max_grad_norm)
                opt.step()

            if target_kl is not None and approx_kl>target_kl:
                break

        y_pred, y_true = b_data['value_ext'].cpu().numpy(), b_data['return_ext'].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        if tqdm is not None:
            pbar.set_postfix(avg_rew_ext=data['rew_ext'].mean().item(), avg_rew_int=data['rew_int'].mean().item(), )
        if callback_fn is not None:
            callback_fn(**locals())
