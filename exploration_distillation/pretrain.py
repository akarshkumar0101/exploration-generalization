# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy
import argparse
import os
from distutils.util import strtobool
from functools import partial

import matplotlib.pyplot as plt
import models
import numpy as np
import torch
import wandb
from einops import rearrange
from tqdm.auto import tqdm

import bc
import env_utils

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu', help="device to run on")
parser.add_argument("--seed", type=int, default=0, help='seed')

parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--project", type=str, default='exploration-distillation')
parser.add_argument("--name", type=str, default='distill_{args.env}_{args.pretrain_levels:05d}_{args.pretrain_obj}')

# Experiment arguments
parser.add_argument("--env", type=str, default="miner", help="the id of the environment")
parser.add_argument("--pretrain-levels", type=int, default=1, help='level')
parser.add_argument("--pretrain-obj", type=str, default='ext', help='objective: ext or int')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--coef_entropy', type=float, default=0) # 1e-2
parser.add_argument('--ds-size', type=float, default=10e9)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--n-steps', type=int, default=5000)

# TODO rename pretrain.py to distill.py

def get_levels_files(env_name, pretrain_levels, pretrain_obj):
    levels = list(range(pretrain_levels))
    files = []
    for level in levels:
        run_name = f"{env_name}_{level:05d}_{pretrain_obj}"
        run_dir = f'data/{run_name}'
        files.append(sorted([f'{run_dir}/{f}' for f in os.listdir(run_dir) if f.startswith('agent')]))

    levels, files = np.array(levels).astype(np.int32), np.array(files)
    levels, files = np.broadcast_arrays(levels[:, None], files)
    return levels, files

def get_updated_levels_files(levels, files, size=2e8):
    # number of agents possible with 10 GB data and 1000 timesteps
    n = int(size/(4*64*64*3)/1000)
    files_flat = np.array(files)[:, ::-1][:, :].T.flatten() # consider all agent checkpoints
    files_flat = files_flat[:n]
    levels_flat = np.array(levels)[:, ::-1][:, :].T.flatten()
    levels_flat = levels_flat[:n]
    n_steps = int(1000*n/len(files_flat))
    print(f'{n=}, {n_steps=}, {len(files_flat)=}')
    return levels_flat, files_flat, n, n_steps

def collect_rollout(env, agent, n_steps, device=None):
    x, y = [], []
    agent = agent.to(device)
    obs, _ = env.reset() # or just step randomly first time
    for step in range(n_steps):
        obs = torch.from_numpy(obs)
        with torch.no_grad():
            action, _, _, _, _ = agent.get_action_and_value(obs.to(device))
        x.append(obs)
        y.append(action.cpu())
        obs, reward, term, trunc, info = env.step(action.tolist())
    return torch.stack(x), torch.stack(y)

def collect_dataset(levels, files, env_name, n_steps, device=None):
    env = env_utils.make_env(1, env_name=f'procgen-{env_name}-v0', level_id=0)
    agent = models.Agent(env)

    x, y = [], []
    for level, file in zip(tqdm(levels.tolist()), files):
        agent.load_state_dict(torch.load(file))
        env = env_utils.make_env(1, env_name=f'procgen-{env_name}-v0', level_id=level)
        xi, yi = collect_rollout(env, agent, n_steps, device=device)
        x.append(xi)
        y.append(yi)
    return torch.stack(x), torch.stack(y)

def callback(args, main_kwargs, **kwargs):
    data = dict()
    
    data['distill/loss_bc'] = kwargs['loss_bc'].item()
    data['distill/loss_entropy'] = kwargs['loss_entropy'].item()
    
    if 'pbar' in kwargs:
        kwargs['pbar'].set_postfix({key: val for key, val in data.items() if isinstance(val, (int, float))})
    if args.track:
        wandb.log(data)
        
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
        
    levels, files = get_levels_files(args.env, args.pretrain_levels, args.pretrain_obj)
    levels, files, _, n_steps = get_updated_levels_files(levels, files, size=args.ds_size)

    x_train, y_train = collect_dataset(levels, files, args.env, n_steps, device=args.device)

    print(f'Dataset size: {x_train.shape}, {y_train.shape}, {x_train.dtype}')
    x_train = rearrange(x_train, '... fs h w c -> (...) fs h w c')
    y_train = rearrange(y_train, '... -> (...)')
    print(f'Dataset size: {x_train.shape}, {y_train.shape}, {x_train.dtype}')
    print(f'{x_train.numel()*x_train.element_size()/1e9} GB')
    
    env = env_utils.make_env(1, env_name=f'procgen-{args.env}-v0', level_id=0)
    agent = models.Agent(env)
    n_params = np.sum([p.numel() for p in agent.parameters()])
    print(f'Agent # parameters: {n_params:012d}')
    
    cb = partial(callback, args=args, main_kwargs=locals())
    bc.train_bc_agent(agent, x_train, y_train, batch_size=args.batch_size,
                      n_steps=args.n_steps, lr=args.lr, coef_entropy=args.coef_entropy,
                      device=args.device, tqdm=tqdm, callback_fn=cb)
    
    if args.track:
        os.makedirs(run_dir, exist_ok=True)
        torch.save(agent.state_dict(), f"{run_dir}/agent.pt")

    return locals()

if __name__=="__main__":
    args = parser.parse_args()
    main(args)
