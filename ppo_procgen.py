# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_procgenpy
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
from einops import rearrange
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from tqdm.auto import tqdm

import wandb


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='{env_id}_{num_levels:2.1e}_{obj}_{seed}')
    parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    # parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #                     help="if toggled, cuda will be enabled by default")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--project", type=str, default="egb",
                        help="the wandb's project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument('--num-levels', type=lambda x: int(float(x)), default=0)
    parser.add_argument('--start-level', type=lambda x: int(float(x)), default=None)
    parser.add_argument('--distribution-mode', type=str, default='easy')
    parser.add_argument('--kl0-coef', type=float, default=0.)

    parser.add_argument('--warmup-critic-steps', type=int, default=None)
    parser.add_argument('--load-agent', type=str, default=None)
    parser.add_argument('--save-agent', type=str, default=None)

    parser.add_argument('--obj', type=str, default='ext')

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="miner",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=lambda x: int(float(x)), default=int(25e6),
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
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
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


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
    def __init__(self, envs):
        super().__init__()
        h, w, c = envs.single_observation_space.shape
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
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x.permute((0, 3, 1, 2)) / 255.0))  # "bhwc" -> "bchw"

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    def get_output(self, x, action=None):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), probs

    def temp_out(self, x, action=None):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value


class StoreObs(gym.Wrapper):
    def __init__(self, env, n_envs=16, store_limit=1000):
        super().__init__(env)
        self.n_envs = n_envs
        self.store_limit = store_limit
        self.past_obs = []

    def reset(self):
        obs = self.env.reset()
        self.past_obs.append(obs[:self.n_envs])
        return obs

    def step(self, action):
        obs, rew, done, infos = self.env.step(action)
        self.past_obs.append(obs[:self.n_envs])
        self.past_obs = self.past_obs[-self.store_limit:]
        return obs, rew, done, infos


class VecMinerEpisodicCoverageReward(gym.Wrapper):
    def __init__(self, env, obj):
        super().__init__(env)
        self.pobs, self.mask_episodic = None, None
        self.obj = obj

    def reset(self):
        obs = self.env.reset()
        self.pobs = obs  # n_envs, h, w, c
        self.mask_episodic = (np.abs(obs - self.pobs) > 1e-3).any(axis=-1)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        # mask_change = (obs != self.pobs).any(axis=-1)
        mask_change = (obs != self.pobs)[..., 1]
        # mask_change = (np.abs(obs - self.pobs) > 1e-3).any(axis=-1)
        rew_eps = (mask_change & (~self.mask_episodic)).mean(axis=(-1, -2))
        rew_eps = np.sign(rew_eps)  # n_envs
        rew_eps[done] = 0.
        self.mask_episodic = self.mask_episodic | mask_change
        self.pobs = obs
        self.mask_episodic[done] = (np.abs(obs[done] - self.pobs[done]) > 1e-3).any(axis=-1)
        for i in range(self.num_envs):
            info[i]['rew_eps'] = rew_eps[i]
            info[i]['rew_ext'] = rew[i]
        rew = rew if self.obj == 'ext' else rew_eps
        return obs, rew, done, info


class ReturnTracker(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._ret_ext, self._ret_eps, self._traj_len = None, None, None

    def reset(self):
        obs = self.env.reset()
        self._ret_ext = np.zeros(self.num_envs, dtype=np.float32)
        self._ret_eps = np.zeros(self.num_envs, dtype=np.float32)
        self._traj_len = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step(self, action):
        obs, rew, dones, infos = self.env.step(action)
        self._ret_ext += np.array([info['rew_ext'] for info in infos])
        self._ret_eps += np.array([info['rew_eps'] for info in infos])
        self._traj_len += 1

        for i, info in enumerate(infos):
            if dones[i]:
                info['ret_ext'] = self._ret_ext[i]
                info['ret_eps'] = self._ret_eps[i]
                info['traj_len'] = self._traj_len[i]

        self._ret_ext[dones] = 0.
        self._ret_eps[dones] = 0.
        self._traj_len[dones] = 0

        return obs, rew, dones, infos


def make_env(obj, num_envs, env_id, num_levels, start_level, distribution_mode, gamma):
    envs = ProcgenEnv(num_envs=num_envs, env_name=env_id, num_levels=num_levels,
                      start_level=start_level, distribution_mode=distribution_mode)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.multi_action_space = gym.spaces.MultiDiscrete([envs.action_space.n] * num_envs)
    envs.observation_space = envs.observation_space["rgb"]
    envs.single_observation_space = envs.observation_space
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = StoreObs(envs)
    envs = VecMinerEpisodicCoverageReward(envs, obj)
    envs = ReturnTracker(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    return envs


def rollout_agent_test_env(args, agent):
    device = torch.device(args.device)
    # envs = make_env(args.num_envs, args.env_id, args.num_levels, args.start_level, args.distribution_mode, args.gamma)
    envs = make_env(args.obj, args.num_envs, args.env_id, 0, 1000000000, args.distribution_mode, args.gamma)
    next_obs = torch.Tensor(envs.reset()).to(device)
    infoss = []
    for i in range(1000):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(next_obs)
            next_obs, reward, done, infos = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)
            infoss.append(infos)
    return envs, infoss


def record_agent_data(envs, infoss, store_vid=True):
    data = {}
    rets_ext = np.array([info['ret_ext'] for infos in infoss for info in infos if 'ret_ext' in info])
    rets_eps = np.array([info['ret_eps'] for infos in infoss for info in infos if 'ret_eps' in info])
    traj_lens = np.array([info['traj_len'] for infos in infoss for info in infos if 'traj_len' in info])
    data['charts/ret_ext'] = np.mean(rets_ext)
    data['charts_hist/ret_ext'] = wandb.Histogram(rets_ext)
    data['charts/ret_eps'] = np.mean(rets_eps)
    data['charts_hist/ret_eps'] = wandb.Histogram(rets_eps)
    data['charts/traj_len'] = np.mean(traj_lens)
    data['charts_hist/traj_len'] = wandb.Histogram(traj_lens)
    if store_vid:
        vid = np.stack(envs.past_obs)  # 1000, 16, 64, 64, 3
        vid[:, :, 0, :, :] = 0  # black border on first row
        vid[:, :, :, 0, :] = 0  # black border on first col
        vid = rearrange(vid, 't (H W) h w c -> t (H h) (W w) c', H=4, W=4)
        data[f'media/vid'] = wandb.Video(rearrange(vid, 't h w c->t c h w'), fps=15)
    return data


def main(args):
    if args.start_level is None:
        args.start_level = args.seed * args.num_levels
    args.name = args.name.format(**args.__dict__)
    args.save_agent = args.save_agent.format(**args.__dict__)
    print(args)

    if args.track:
        print('Starting wandb')
        wandb.init(
            project=args.project,
            entity=args.entity,
            # sync_tensorboard=True,
            config=args,
            name=args.name,
            # monitor_gym=True,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device)
    print(f'Using device: {device}')

    print('Creating environment')
    envs = make_env(args.obj, args.num_envs, args.env_id, args.num_levels, args.start_level, args.distribution_mode,
                    args.gamma)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    print('Creating agent')
    agent, agent0 = Agent(envs), None
    if args.load_agent is not None:
        print('Loading agent')
        agent0 = Agent(envs)
        agent0.load_state_dict(torch.load(f'{args.load_agent}/agent.pt'))
        agent.load_state_dict(agent0.state_dict())
        agent0 = agent0.to(device)
    agent = agent.to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    print('Creating buffers')
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    warmup_critic_steps = 0 if args.warmup_critic_steps is None else args.warmup_critic_steps
    num_updates = num_updates + warmup_critic_steps // args.batch_size
    critic_warm = True if args.warmup_critic_steps is None else False
    if not critic_warm:  # freeze network + actor for critic warmup
        print('Freezing everything but the critic')
        for name, p in agent.named_parameters():
            if not name.startswith('critic'):
                p.requires_grad_(False)

    best_ret_ext_train = None

    print('Starting learning...')
    pbar = tqdm(range(1, num_updates + 1))
    for update in pbar:
        data = dict()

        if not critic_warm and global_step > args.warmup_critic_steps:  # unfreeze network+actor
            critic_warm = True
            print('Unfreezing critic')
            for name, p in agent.named_parameters():
                p.requires_grad_(True)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        infoss = []
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            infoss.append(infos)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                _, newlogprob, entropy, newvalue, dist = agent.get_output(b_obs[mb_inds], b_actions.long()[mb_inds])
                if args.load_agent is not None:
                    with torch.no_grad():
                        _, _, _, _, dist0 = agent0.get_output(b_obs[mb_inds], b_actions.long()[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                if args.load_agent is not None:
                    ce0 = nn.functional.cross_entropy(dist.logits, dist0.probs, reduction='none')
                    kl0 = ce0 - dist0.entropy()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + kl0.mean() * args.kl0_coef
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        data['details/lr'] = optimizer.param_groups[0]["lr"]
        data['details/value_loss'] = v_loss.item()
        data['details/policy_loss'] = pg_loss.item()
        data['details/entropy'] = entropy_loss.item()
        data['details/old_approx_kl'] = old_approx_kl.item()
        data['details/approx_kl'] = approx_kl.item()
        data['details/clipfrac'] = np.mean(clipfracs)
        data['details/explained_variance'] = explained_var
        data['meta/SPS'] = int(global_step / (time.time() - start_time))
        data['meta/global_step'] = global_step
        data['details/ce0'] = ce0.mean().item()
        data['details/kl0'] = kl0.mean().item()

        viz_slow = (update - 1) % (num_updates // 20) == 0
        data_ret = record_agent_data(envs, infoss, store_vid=viz_slow)
        data.update({f'{k}_train': v for k, v in data_ret.items()})
        if viz_slow:
            envs_test, infoss_test = rollout_agent_test_env(args, agent)
            data_ret = record_agent_data(envs_test, infoss_test, store_vid=viz_slow)
            data.update({f'{k}_test': v for k, v in data_ret.items()})

        if viz_slow and args.save_agent is not None and data['charts/ret_ext_train'] > best_ret_ext_train:
            best_ret_ext_train = data['charts/ret_ext_train']
            os.makedirs(args.save_agent, exist_ok=True)
            torch.save(agent.state_dict(), f'{args.save_agent}/agent.pt')
            torch.save(data, f'{args.save_agent}/data.pt')

        keys_tqdm = ['charts/ret_ext_train', 'charts/ret_eps_train', 'meta/SPS']
        pbar.set_postfix({k.split('/')[-1]: data[k] for k in keys_tqdm})
        if args.track:
            wandb.log(data, step=global_step)

        # pbar.set_postfix(dict(sps=data['meta/SPS']))
    envs.close()


def test_env_speed():
    env = make_env('ext', 64, 'miner', 0, 0, 'easy', 0.999)
    env.reset()
    print(env.action_space)
    start = time.time()
    steps = 1000
    for i in tqdm(range(steps)):
        env.step(np.array([env.action_space.sample() for _ in range(64)]))
    print((steps * 64) / (time.time() - start))


if __name__ == "__main__":
    # test_env_speed()

    main(parse_args())
    # use default arguments from argparse
    # envs = make_env('ext', 64, 'miner', 0, 0, 'easy', 0.999)
    # envs.reset()
    # envs.step(np.array([0]*64))

# TODO: optimize intrinsic rewards to run batched (on the agent side)
# TODO: record videos manually (not using my gym wrappers on the environment)
# Every n_updates//20 steps, create a separate test and train env and generate rollouts
# and manually stich together videos using a obs variable.
