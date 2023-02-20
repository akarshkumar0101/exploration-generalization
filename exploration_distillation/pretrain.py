# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy
import argparse
from distutils.util import strtobool
from functools import partial

import matplotlib.pyplot as plt
import models
import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

import bc
import env_utils

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu', help="device to run on")
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--seed", type=int, default=0, help='seed')

# Experiment arguments
parser.add_argument("--env", type=str, default="miner", help="the id of the environment")
parser.add_argument("--pretrain-levels", type=int, default=0, help='level')
parser.add_argument("--pretrain-obj", type=str, default='ext', help='objective: ext or int')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--coef_entropy', type=float, default=1e-2)
parser.add_argument('--n_dataset', type=int, default=1e6)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_steps', type=int, default=1000)

def callback(**kwargs):
    pass

def main():
    args = parser.parse_args()
    assert args.pretrain_obj in {'ext', 'int'}
    args.ext_coef, args.int_coef = (1.0, 0.0) if args.train_obj=='ext' else (0.0, 1.0)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # env-level-type-pretrain-levels-type  |  type in {ext, int}
    run_name = f"{args.env}_{args.level:05d}_{args.train_obj}"
    if args.pretrain_levels>0:
        run_name += f"_pretrain_{args.pretrain_levels:05d}_{args.pretrain_obj}"
    run_dir = f'data/{run_name}'
    print(run_name)
    print(run_dir)

    if args.track:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project_name,
            name=run_name,
            config=vars(args),
            save_code=True,
            # sync_tensorboard=True,
            # monitor_gym=True,
        )
        
    env = env_utils.make_env(args.num_envs, env_name=f'procgen-{args.env}-v0', level_id=args.level)
    agent = models.Agent(env)
    rnd = models.RNDModel(env, (64, 64, 3))
    n_params = np.sum([p.numel() for p in agent.parameters()])
    print(f'Agent # parameters: {n_params:012d}')
    n_params = np.sum([p.numel() for p in rnd.parameters()])
    print(f'RND   # parameters: {n_params:012d}')

    x_train, y_train = None, None

    cb = partial(callback, args=args, main_kwargs=locals())
    bc.train_bc_agent(agent, x_train, y_train, batch_size=args.batch_size,
                      n_steps=args.n_steps, lr=args.lr, coef_entropy=args.coef_entropy,
                      device=args.device, tqdm=tqdm, callback_fn=cb)


if __name__=="__main__":
    main()

