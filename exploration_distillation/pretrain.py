# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy
import argparse
import os
from distutils.util import strtobool
from functools import partial

import numpy as np
import torch
import wandb
from einops import rearrange
from tqdm.auto import tqdm

import env_utils
import models
from train import generate_video

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu', help="device to run on")
parser.add_argument("--seed", type=int, default=0, help='seed')

parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--project", type=str, default='exploration-distillation')
parser.add_argument("--name", type=str, default='distill_{env}_{pretrain_levels:05d}_{pretrain_obj}')

parser.add_argument("--load-dir", type=str, default=None)

# Experiment arguments
parser.add_argument("--env", type=str, default="miner", help="the id of the environment")
parser.add_argument("--pretrain-levels", type=int, default=1, help='level')
parser.add_argument("--pretrain-obj", type=str, default='ext', help='objective: ext or int')

parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--coef_entropy', type=float, default=0)  # 1e-2
# parser.add_argument('--ds-size', type=float, default=2e9)
parser.add_argument('--batch-size', type=int, default=4096)
parser.add_argument('--n-steps', type=int, default=50000)

parser.add_argument('--n-viz-fast', type=int, default=50000)
parser.add_argument('--n-viz-slow', type=int, default=30)


# TODO rename pretrain.py to distill.py

def get_level2files(args):
    levels = list(range(args.pretrain_levels))
    level2files = {}
    files = []
    for level in levels:
        run_dir = f'{args.load_dir}/{args.env}_{level:05d}_{args.pretrain_obj}'
        level2files[level] = sorted([f'{run_dir}/{f}' for f in os.listdir(run_dir) if f.startswith('agent')])
        files.append(sorted([f'{run_dir}/{f}' for f in os.listdir(run_dir) if f.startswith('agent')]))
    return level2files


def collect_rollout(env, agent, n_steps, device=None, pbar=None):
    x, y, y_probs = [], [], []
    agent = agent.to(device).eval()
    obs, _ = env.reset()  # or just step randomly first time
    for step in range(n_steps):
        obs = torch.from_numpy(obs)
        with torch.no_grad():
            # action, _, _, _, _ = agent.get_action_and_value(obs.to(device))
            dist, _, _ = agent.get_dist_and_values(obs.to(device))
            action = dist.sample()
        x.append(obs)
        y.append(action.cpu())
        y_probs.append(dist.probs.cpu())
        obs, reward, term, trunc, info = env.step(action.tolist())
        if pbar is not None:
            pbar.update()
    return torch.stack(x), torch.stack(y), torch.stack(y_probs)


def collect_batch(env_name, level2files, n_agents, n_envs, n_steps, device=None):
    env = env_utils.make_env(1, env_name=f'procgen-{env_name}-v0', level_id=0)
    agent = models.Agent(env)

    lf = [(level, file) for level, files in level2files.items() for file in files]
    lf = [lf[i] for i in np.random.choice(len(lf), size=n_agents)]

    envs, x, y, y_probs = [], [], [], []
    pbar = tqdm(total=n_agents*n_steps)
    for level, file in lf:
        agent.load_state_dict(torch.load(file))
        env = env_utils.make_env(n_envs, env_name=f'procgen-{env_name}-v0', level_id=level)
        xi, yi, yi_probs = collect_rollout(env, agent, n_steps, device=device, pbar=pbar)
        envs.append(env)
        x.append(xi)
        y.append(yi)
        y_probs.append(yi_probs)
    return envs, torch.stack(x), torch.stack(y), torch.stack(y_probs)


def eval_performance(agent, env_name, levels, n_steps=2000, device=None):
    env = env_utils.make_env(len(levels), env_name=f'procgen-{env_name}-v0', level_id=levels)
    agent = agent.to(device)
    obs, _ = env.reset()  # or just step randomly first time
    for step in tqdm(range(n_steps)):
        obs = torch.from_numpy(obs)
        with torch.no_grad():
            action, _, _, _, _ = agent.get_action_and_value(obs.to(device))
        obs, reward, term, trunc, info = env.step(action.tolist())
    rets_ext = [e.past_returns_ext for e in env.envs]
    rets_eps = [e.past_returns_eps for e in env.envs]
    return env, rets_ext, rets_eps


def callback(args, main_kwargs, **kwargs):
    data = dict()

    if kwargs['i_step'] % max(1, (kwargs['n_steps'] // args.n_viz_fast)) == 0:
        data['distill/loss_bc'] = kwargs['loss_bc'].item()
        data['distill/loss_entropy'] = kwargs['loss_entropy'].item()
        dist = torch.distributions.Categorical(probs=kwargs['y_probs_batch'])
        data['distill/expert_entropy'] = dist.entropy().mean().item()

        envs_experts = main_kwargs['batch_data']['envs_expert']
        envs = [e for svenv in envs_experts for e in svenv.envs]
        rets_ext = [r for e in envs for r in e.past_returns_ext]
        rets_eps = [r for e in envs for r in e.past_returns_eps]
        data['charts_hist/expert_rets_ext_hist'] = wandb.Histogram(rets_ext)
        data['charts_hist/expert_rets_eps_hist'] = wandb.Histogram(rets_eps)
        data['charts/expert_rets_ext'] = np.mean(rets_ext)
        data['charts/expert_rets_eps'] = np.mean(rets_eps)

    if kwargs['i_step'] % max(1, (kwargs['n_steps'] // args.n_viz_slow)) == 0:
        levels_seen = (np.arange(128) % args.pretrain_levels).tolist()
        levels_zeroshot = (np.arange(128) + 10000).tolist()
        levels_all = levels_seen + levels_zeroshot
        env, rets_ext, rets_eps = eval_performance(main_kwargs['agent'], args.env, levels_all, n_steps=1000,
                                                   device=args.device)
        envs_seen, envs_zeroshot = env.envs[:len(levels_seen)], env.envs[len(levels_seen):]
        rets_ext_seen, rets_ext_zeroshot = rets_ext[:len(levels_seen)], rets_ext[len(levels_seen):]
        rets_eps_seen, rets_eps_zeroshot = rets_eps[:len(levels_seen)], rets_eps[len(levels_seen):]
        # flatten list of list to a list
        rets_ext_seen = [r for e in rets_ext_seen for r in e]
        rets_ext_zeroshot = [r for e in rets_ext_zeroshot for r in e]
        rets_eps_seen = [r for e in rets_eps_seen for r in e]
        rets_eps_zeroshot = [r for e in rets_eps_zeroshot for r in e]

        data['charts_hist/rets_ext_seen'] = wandb.Histogram(rets_ext_seen)
        data['charts_hist/rets_eps_seen'] = wandb.Histogram(rets_eps_seen)
        data['charts_hist/rets_ext_zeroshot'] = wandb.Histogram(rets_ext_zeroshot)
        data['charts_hist/rets_eps_zeroshot'] = wandb.Histogram(rets_eps_zeroshot)
        data['charts/rets_ext_seen'] = np.mean(rets_ext_seen)
        data['charts/rets_eps_seen'] = np.mean(rets_eps_seen)
        data['charts/rets_ext_zeroshot'] = np.mean(rets_ext_zeroshot)
        data['charts/rets_eps_zeroshot'] = np.mean(rets_eps_zeroshot)

        video = generate_video(envs_seen, shape=(5, 5))
        data['media/video_seen'] = wandb.Video(rearrange(video, 't h w c->t c h w'), fps=15)
        video = generate_video(envs_zeroshot, shape=(5, 5))
        data['media/video_zeroshot'] = wandb.Video(rearrange(video, 't h w c->t c h w'), fps=15)

    if 'pbar' in kwargs:
        kwargs['pbar'].set_postfix({key: val for key, val in data.items() if isinstance(val, (int, float))})
    if args.track:
        wandb.log(data)


def train_bc_agent(agent, get_next_batch, batch_size=32, n_steps=10, lr=1e-3, coef_entropy=0.0,
                   device=None, tqdm=None, callback_fn=None):
    """
    Behavior Cloning
    """
    agent = agent.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    pbar = range(n_steps)
    if tqdm is not None:
        pbar = tqdm(pbar)
    for i_step in pbar:
        x_batch, y_batch, y_probs_batch = get_next_batch(**locals())
        x_batch, y_batch, y_probs_batch = x_batch.float().to(device), y_batch.long().to(
            device), y_probs_batch.float().to(device)
        dist, _, _ = agent.get_dist_and_values(x_batch)

        # loss_bc = loss_fn(dist.logits, y_batch).mean() # hard action loss
        loss_bc = loss_fn(dist.logits, y_probs_batch).mean()  # soft action loss
        loss_entropy = dist.entropy().mean()
        loss = loss_bc - coef_entropy * loss_entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        if tqdm is not None:
            pbar.set_postfix(loss_bc=loss_bc.item(), entropy=loss_entropy.item())
        if callback_fn is not None:
            callback_fn(**locals())


def main(args):
    assert args.pretrain_obj in {'ext', 'int', 'eps', 'epd'}

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(args)
    # env-level-type-pretrain-levels-type  |  type in {ext, int}
    name = args.name.format(**args.__dict__)
    run_dir = f'data/{name}'
    print(name)
    print(run_dir)

    if args.track:
        wandb.init(
            # entity=args.wandb_entity,
            project=args.project,
            name=name,
            config=vars(args),
            save_code=True,
            # sync_tensorboard=True,
            # monitor_gym=True,
        )

    # levels, files = get_levels_files(args.env, args.pretrain_levels, args.pretrain_obj)
    # levels, files, _, n_steps = get_updated_levels_files(levels, files, size=args.ds_size)
    level2files = get_level2files(args)
    level2files = {k: [v[-1]] for k, v in level2files.items()}  # only use last file
    # n_steps = int(args.ds_size/(4*64*64*3)/len(level2files))

    env = env_utils.make_env(1, env_name=f'procgen-{args.env}-v0', level_id=0)
    agent = models.Agent(env)
    n_params = np.sum([p.numel() for p in agent.parameters()])
    print(f'Agent # parameters: {n_params:012d}')

    batch_data = {}

    def get_next_batch(i_step, **kwargs):
        if i_step % 10 == 0:
            envs_expert, x_train, y_train, y_probs_train = collect_batch(args.env, level2files, n_agents=10, n_envs=4,
                                                                         n_steps=1000, device=args.device)
            print(f'Dataset size: {x_train.numel() * x_train.element_size() / 1e9} GB')
            print(f'Dataset: {x_train.shape}, {y_train.shape}, {y_probs_train.shape}, {x_train.dtype}')
            x_train = rearrange(x_train, '... fs h w c -> (...) fs h w c')
            y_train = rearrange(y_train, '... -> (...)')
            y_probs_train = rearrange(y_probs_train, '... l -> (...) l')
            print(f'Dataset: {x_train.shape}, {y_train.shape}, {y_probs_train.shape}, {x_train.dtype}')

            batch_data['envs_expert'] = envs_expert
            batch_data['x_train'] = x_train
            batch_data['y_train'] = y_train
            batch_data['y_probs_train'] = y_probs_train
        else:
            x_train, y_train, y_probs_train = batch_data['x_train'], batch_data['y_train'], batch_data['y_probs_train']

        idxs_batch = torch.randperm(len(x_train))[:args.batch_size]
        x_batch, y_batch, y_probs_batch = x_train[idxs_batch], y_train[idxs_batch], y_probs_train[idxs_batch]
        return x_batch, y_batch, y_probs_batch

    cb = partial(callback, args=args, main_kwargs=locals())
    train_bc_agent(agent, get_next_batch, batch_size=args.batch_size,
                   n_steps=args.n_steps, lr=args.lr, coef_entropy=args.coef_entropy,
                   device=args.device, tqdm=tqdm, callback_fn=cb)

    if args.track:
        os.makedirs(run_dir, exist_ok=True)
        torch.save(agent.state_dict(), f"{run_dir}/agent.pt")

    return locals()


if __name__ == "__main__":
    main(parser.parse_args())
