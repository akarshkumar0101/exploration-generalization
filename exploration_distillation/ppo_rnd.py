# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy
import time
from collections import deque
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers.normalize import RunningMeanStd
from tqdm.auto import tqdm


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

def init_rnd_model(rnd_model, envs, num_steps, num_iterations_obs_norm_init):
    # for step in tqdm(range(num_steps * num_iterations_obs_norm_init)):
    for step in tqdm(range(10)):
        o, _, _, _, _ = envs.step(envs.action_space.sample())
        rnd_model.update(o)

def run(agent, rnd_model, envs, tqdm=None, device=None, callback_fn=None,
        total_timesteps=1e6, learning_rate=5e-4, num_steps=256,
        anneal_lr=False, gamma=0.999, gae_lambda=0.95,
        num_minibatches=16, update_epochs=1, norm_adv=True,
        clip_coef=0.2, clip_vloss=True, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, target_kl=None,
        # RND arguments
        update_proportion=0.25, int_coef=1.0, ext_coef=2.0, int_gamma=0.99,
        num_iterations_obs_norm_init=2,
    ):
    num_envs = envs.num_envs
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    
    # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f'Using device {device}')
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = agent.to(device)
    rnd_model = rnd_model.to(device)
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr=learning_rate,
        eps=1e-5,
    )

    reward_rms = RunningMeanStd()
    discounted_reward = RewardForwardFilter(int_gamma)

    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    curiosity_rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    ext_values = torch.zeros((num_steps, num_envs)).to(device)
    int_values = torch.zeros((num_steps, num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = int(total_timesteps // batch_size)

    init_rnd_model(rnd_model, envs, num_steps, num_iterations_obs_norm_init)

    pbar = range(1, num_updates + 1)
    if tqdm is not None: pbar = tqdm(pbar, leave=False)
    for update in pbar:
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                value_ext, value_int = agent.get_value(obs[step])
                ext_values[step], int_values[step] = (
                    value_ext.flatten(),
                    value_int.flatten(),
                )
                action, logprob, _, _, _ = agent.get_action_and_value(obs[step])

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            predict_next_feature, target_next_feature = rnd_model(next_obs)
            curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data
            # for idx, d in enumerate(done):
            #     if d and info["lives"][idx] == 0:
            #         avg_returns.append(info["r"][idx])
            #         epi_ret = np.average(avg_returns)
            #         print(
            #             f"global_step={global_step}, episodic_return={info['r'][idx]}, curiosity_reward={np.mean(curiosity_rewards[step].cpu().numpy())}"
            #         )
            #         writer.add_scalar("charts/avg_episodic_return", epi_ret, global_step)
            #         writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
            #         writer.add_scalar(
            #             "charts/episode_curiosity_reward",
            #             curiosity_rewards[step][idx],
            #             global_step,
            #         )
            #         writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        curiosity_reward_per_env = np.array(
            # [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy()]
        )
        reward_rms.update(curiosity_reward_per_env.flatten())
        # mean, std, count = (
        #     np.mean(curiosity_reward_per_env),
        #     np.std(curiosity_reward_per_env),
        #     len(curiosity_reward_per_env),
        # )
        # reward_rms.update_from_moments(mean, std**2, count)

        curiosity_rewards /= np.sqrt(reward_rms.var)

        # bootstrap value if not done
        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(curiosity_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = curiosity_rewards[t] + int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + gamma * gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + int_gamma * gae_lambda * int_nextnonterminal * int_lastgaelam
                )
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)

        b_advantages = b_int_advantages * int_coef + b_ext_advantages * ext_coef

        rnd_model.update(b_obs.cpu().numpy())

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)

        # rnd_next_obs = (
        #     (
        #         (b_obs[:, 3, :, :].reshape(-1, 1, 84, 84) - torch.from_numpy(obs_rms.mean).to(device))
        #         / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
        #     ).clip(-5, 5)
        # ).float()

        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                predict_next_state_feature, target_next_state_feature = rnd_model(b_obs[mb_inds])
                forward_loss = F.mse_loss(
                    predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                ).mean(-1)

                mask = torch.rand(len(forward_loss), device=device)
                mask = (mask < update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )
                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        max_grad_norm,
                    )
                optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break
                    
        sps = int(global_step / (time.time() - start_time))
        if callback_fn is not None:
            callback_fn(**locals())
