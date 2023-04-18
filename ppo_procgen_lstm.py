# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from tqdm.auto import tqdm

import wandb
from agent_procgen import IDM, AgentLSTM
from env_procgen import make_env


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
    #     help="the name of this experiment")
    parser.add_argument("--name", type=str, default='{env_id}_{num_levels:2.1e}_{obj}_{seed}')
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    # parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        # help="if toggled, cuda will be enabled by default")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    # parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
    #     help="the wandb's project name")
    parser.add_argument("--project", type=str, default='egb')
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

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

    # My arguments
    parser.add_argument("--actions", type=str, default='all')
    parser.add_argument("--lstm-type", type=str, default='lstm')
    parser.add_argument('--num-levels', type=lambda x: int(float(x)), default=0)
    parser.add_argument('--start-level', type=lambda x: int(float(x)), default=None)
    parser.add_argument('--distribution-mode', type=str, default='easy')
    parser.add_argument('--kl0-coef', type=float, default=0.)
    parser.add_argument("--pre-env-id", type=str, default="miner")
    parser.add_argument('--pre-num-levels', type=lambda x: int(float(x)), default=0)
    parser.add_argument('--pre-obj', type=str, default=None)
    parser.add_argument("--pre-seed", type=int, default=None, help="seed of the experiment")
    parser.add_argument("--warmup-critic-steps", type=lambda x: int(float(x)), default=None)
    parser.add_argument('--load-agent', type=str, default=None)
    parser.add_argument('--save-agent', type=str, default=None)
    parser.add_argument('--idm-lr', type=float, default=5e-4)

    parser.add_argument('--obj', type=str, default='ext')


    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


@torch.no_grad()
def rollout_agent_test_env(agent, envs, n_steps=1000):
    _, info = envs.reset()
    device = info["obs"].device

    next_obs = info["obs"]
    next_done = torch.zeros(envs.num_envs).to(device)
    # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    next_lstm_state = (torch.zeros(agent.lstm.num_layers, envs.num_envs, agent.lstm.hidden_size).to(device), torch.zeros(agent.lstm.num_layers, envs.num_envs, agent.lstm.hidden_size).to(device))
    for _ in range(n_steps):
        action, _, _, _, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
        _, _, _, info = envs.step(action.cpu().numpy())
        next_obs, next_done = info["obs"], info["done"].to(torch.uint8)
    return envs


def record_agent_data(envs, store_vid=True):
    data = {}
    rets_ext = torch.cat(envs.rets_ext).cpu().numpy()
    rets_e3b = torch.cat(envs.rets_e3b).cpu().numpy()
    rets_cov = torch.cat(envs.rets_cov).cpu().numpy()
    traj_lens = torch.cat(envs.traj_lens).cpu().numpy()
    data["charts/ret_ext"] = rets_ext.mean()
    data["charts_hist/ret_ext"] = wandb.Histogram(rets_ext)
    data["charts/ret_e3b"] = rets_e3b.mean()
    data["charts_hist/ret_e3b"] = wandb.Histogram(rets_e3b)
    data["charts/ret_cov"] = rets_cov.mean()
    data["charts_hist/ret_cov"] = wandb.Histogram(rets_cov)
    data["charts/traj_len"] = traj_lens.mean()
    data["charts_hist/traj_len"] = wandb.Histogram(traj_lens)
    if store_vid:
        vid = np.stack(envs.past_obs)  # 1000, 25, 64, 64, 3
        vid[:, :, 0, :, :] = 0  # black border on first row
        vid[:, :, :, 0, :] = 0  # black border on first col
        vid = rearrange(vid, "t (H W) h w c -> t (H h) (W w) c", H=5, W=5)
        data[f"media/vid"] = wandb.Video(rearrange(vid, "t h w c->t c h w"), fps=15)
    return data


def main(args):
    if args.start_level is None:
        args.start_level = args.seed * args.num_levels
    if args.name:
        args.name = args.name.format(**args.__dict__)
    if args.save_agent:
        args.save_agent = args.save_agent.format(**args.__dict__)
    if args.load_agent:
        args.load_agent = args.load_agent.format(**args.__dict__)
    print(args)
    if args.track:
        print(f"Starting wandb with run name {args.name}...")
        wandb.init(project=args.project, config=args, name=args.name, save_code=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device)
    # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # env setup
    print("Creating env...")
    envs = make_env(args.env_id, args.obj, args.num_envs, args.start_level, args.num_levels, args.distribution_mode, args.gamma, encoder=None, device=device, cov=True, actions=args.actions)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    print("Creating agent...")
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n
    agent, agent0 = AgentLSTM(obs_shape, n_actions, lstm_type=args.lstm_type).to(device), None
    if args.load_agent is not None:
        print("Loading agent...")
        agent0 = AgentLSTM(obs_shape, n_actions, lstm_type=args.lstm_type).to(device)
        agent0.load_state_dict(torch.load(f"{args.load_agent}/agent.pt"))
        agent.load_state_dict(agent0.state_dict())
    idm = IDM(obs_shape, n_actions, n_features=64, normalize=True, merge="both").to(device)
    optimizer = optim.Adam([{"params": agent.parameters(), "lr": args.learning_rate, "eps": 1e-5}, {"params": idm.parameters(), "lr": args.idm_lr, "eps": 1e-8}])
    envs.set_encoder(idm)

    # if args.track:
    #     wandb.watch((agent, idm), log="all", log_freq=args.total_timesteps // args.batch_size // 100)

    # ALGO Logic: Storage setup
    print("Creating buffers...")
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # next_obs = torch.Tensor(envs.reset()).to(device)
    _, info = envs.reset()
    next_obs = info["obs"]
    next_done = torch.zeros(args.num_envs).to(device)
    # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    next_lstm_state = (torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device), torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device))
    num_updates = args.total_timesteps // args.batch_size
    best_ret_train = -np.inf

    warmup_critic_steps = 0 if args.warmup_critic_steps is None else args.warmup_critic_steps
    num_updates = num_updates + warmup_critic_steps // args.batch_size
    critic_warm = True if args.warmup_critic_steps is None else False
    if not critic_warm:  # freeze network + actor for critic warmup
        print('Freezing everything except the critic')
        for name, p in agent.named_parameters():
            if not name.startswith('critic'):
                p.requires_grad_(False)

    print("Starting learning...")
    pbar = tqdm(range(1, num_updates + 1))
    for update in pbar:
        if not critic_warm and global_step > args.warmup_critic_steps:  # unfreeze network+actor
            critic_warm = True
            print('Unfreezing everything')
            for name, p in agent.named_parameters():
                p.requires_grad_(True)

        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            next_obs, next_done = info["obs"], info["done"].to(torch.uint8)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
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
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
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
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # IDM loss, TODO: make this more efficient by not computing v twice
                mb_obs = b_obs[mb_inds].reshape(args.num_steps, envsperbatch, *obs_shape)  # T, N, H, W, C
                # mb_obs_flat_now = rearrange(mb_obs[:-1], 't n h w c -> (t n) h w c')
                # mb_obs_flat_nxt = rearrange(mb_obs[ 1:], 't n h w c -> (t n) h w c')
                mb_actions = b_actions.long()[mb_inds].reshape(args.num_steps, envsperbatch)  # T, N
                mb_actions_flat_now = rearrange(mb_actions[:-1], "t n -> (t n)")
                # logits_idm = e3b.idm(mb_obs_flat_now, mb_obs_flat_nxt)
                v1, v2, logits_idm = idm.forward_smart(mb_obs)  # T-1 N A
                logits_idm = logits_idm.flatten(0, 1)
                loss_idm = nn.functional.cross_entropy(logits_idm, mb_actions_flat_now, reduction="none")
                acc_idm = (logits_idm.argmax(dim=-1)==mb_actions_flat_now).float().mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + loss_idm.mean()

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

        viz_slow = (update - 1) % (num_updates // 10) == 0

        data = {}
        data["details/learning_rate"] = optimizer.param_groups[0]["lr"]
        data["details/value_loss"] = v_loss.item()
        data["details/policy_loss"] = pg_loss.item()
        data["details/entropy"] = entropy_loss.item()
        data["details_hist/entropy"] = wandb.Histogram(entropy.detach().cpu().numpy())
        data["details/old_approx_kl"] = old_approx_kl.item()
        data["details/approx_kl"] = approx_kl.item()
        data["details/clipfrac"] = np.mean(clipfracs)
        data["details/explained_variance"] = explained_var
        data["meta/SPS"] = int(global_step / (time.time() - start_time))
        data["meta/global_step"] = global_step

        data["e3b/idm_loss"] = loss_idm.mean().item()
        data["e3b/idm_accuracy"] = acc_idm.item()
        for i, am in enumerate(envs.action_meanings):
            data[f"e3b_details/loss_action_{am}"] = loss_idm[mb_actions_flat_now == i].mean().item()
        # if args.load_agent is not None:
        #     data['details/ce0'] = ce0.mean().item()
        #     data['details/kl0'] = kl0.mean().item()

        data_ret = record_agent_data(envs, store_vid=args.track and viz_slow)
        data.update({f"{k}_train": v for k, v in data_ret.items()})
        if args.track and viz_slow:
            print("Rolling out test envs...")
            envs_test = make_env(args.env_id, args.obj, args.num_envs, 1000000, 0, args.distribution_mode, args.gamma, encoder=idm, device=device, cov=True, actions=args.actions)

            envs_test = rollout_agent_test_env(agent, envs_test, n_steps=1000)
            data_ret = record_agent_data(envs_test, store_vid=args.track and viz_slow)
            data.update({f"{k}_test": v for k, v in data_ret.items()})
        
        if args.track and viz_slow:
            fig = plt.figure()
            plt.scatter(torch.cat(envs.rets_cov).tolist(), torch.cat(envs.rets_e3b).tolist())
            plt.xlabel('cov return'); plt.ylabel('e3b return')
            data['e3b/e3b_vs_cov_returns'] = wandb.Image(fig)

            fig = plt.figure()
            plt.hist(b_actions.cpu().numpy(), bins=envs.single_action_space.n*2)
            plt.xticks(ticks=np.arange(envs.single_action_space.n), labels=envs.action_meanings)
            plt.title('Action distribution')
            data['e3b/action_distribution'] = wandb.Image(fig)

        if viz_slow and args.save_agent is not None:
            print("Saving agent...")
            os.makedirs(args.save_agent, exist_ok=True)
            torch.save(agent.state_dict(), f"{args.save_agent}/agent_{global_step:012.0f}.pt")
            torch.save(idm.state_dict(), f"{args.save_agent}/idm_{global_step:012.0f}.pt")
            if data[f"charts/ret_{args.obj}_train"] > best_ret_train:
                best_ret_train = data[f"charts/ret_{args.obj}_train"]
                torch.save(agent.state_dict(), f"{args.save_agent}/agent.pt")
                torch.save(idm.state_dict(), f"{args.save_agent}/idm.pt")
                torch.save(data, f"{args.save_agent}/data.pt")

        keys_tqdm = ["charts/ret_ext_train", "charts/ret_e3b_train", "charts/ret_cov_train", "meta/SPS"]
        pbar.set_postfix({k.split("/")[-1]: data[k] for k in keys_tqdm})
        if args.track:
            wandb.log(data, step=global_step)
        plt.close('all')

    envs.close()


if __name__ == "__main__":
    main(parse_args())
