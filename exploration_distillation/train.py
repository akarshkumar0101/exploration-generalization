# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy
import argparse
from distutils.util import strtobool
from functools import partial

import matplotlib.pyplot as plt
import models
import numpy as np
import ppo_rnd
import torch
import wandb
from einops import rearrange, repeat
from tqdm.auto import tqdm

import env_utils

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu', help="device to run on")
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--seed", type=int, default=0, help='seed')

# Experiment arguments
parser.add_argument("--env", type=str, default="miner", help="the id of the environment")
parser.add_argument("--level", type=int, default=0, help='level')
parser.add_argument("--train-obj", type=str, default='ext', help='objective: ext or int')
parser.add_argument("--pretrain-levels", type=int, default=0, help='level')
parser.add_argument("--pretrain-obj", type=str, default='ext', help='objective: ext or int')

# Algorithm arguments
parser.add_argument("--total-timesteps", type=int, default=4e6,
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
parser.add_argument("--num-minibatches", type=int, default=16,
    help="the number of mini-batches")
parser.add_argument("--update-epochs", type=int, default=1,
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

# RND arguments
parser.add_argument("--update-proportion", type=float, default=0.25,
    help="proportion of exp used for predictor update")
parser.add_argument("--int-coef", type=float, default=1.0,
    help="coefficient of extrinsic reward")
parser.add_argument("--ext-coef", type=float, default=2.0,
    help="coefficient of intrinsic reward")
parser.add_argument("--int-gamma", type=float, default=0.99,
    help="Intrinsic reward discount rate")
parser.add_argument("--num-iterations-obs-norm-init", type=int, default=2,
    help="number of iterations to initialize the observations normalization parameters")

def generate_video(env, shape=(2, 2)):
    """
    Combine X videos and generate a single video of shape (t, h, w, c)
    with appropriate border, looping, padding.
    """
    n_vids = np.prod(shape)
    assert len(env.envs) >= n_vids
    ptos = [e.past_traj_obs for e in env.envs[:n_vids]]
    maxlen = max([len(i) for i in ptos])
    def process_video(vid):
        vid[:, 0, :, :] = 0 # paint black border
        vid[:, :, 0, :] = 0 # paint black border
        # loop video
        vid = repeat(vid, 't h w c -> (loop t) h w c', loop=maxlen//len(vid))
        # end padding: repeat last frame
        padding = repeat(vid[-1], 'h w c -> t h w c', t=maxlen-len(vid))
        vid = np.concatenate([vid, padding], axis=0)
        return vid
    vids = np.stack([process_video(vid) for vid in ptos])
    vids = rearrange(vids, '(b1 b2) t h w c -> t (b1 h) (b2 w) c', b1=shape[0], b2=shape[1])
    return vids

def callback(args, main_kwargs, **kwargs):
    env = main_kwargs['env']

    data = dict()
    data['charts/entropy'] = kwargs['entropy_loss'].item()
    data['details/value_loss'] = kwargs['v_loss'].item()
    data['details/policy_loss'] = kwargs['pg_loss'].item()
    data['details/learning_rate'] = kwargs['optimizer'].param_groups[0]["lr"]
    data['details/old_approx_kl'] = kwargs['old_approx_kl'].item()
    data['details/approx_kl'] = kwargs['approx_kl'].item()
    data['details/clipfrac'] = np.mean(kwargs['clipfracs'])
    # data['losses/explained_variance'] = kwargs['explained_var']
    data['details/SPS'] = kwargs['sps']

    first_obs = env.envs[0].first_obs.copy()
    calc_traj_cov = lambda o:(o.std(axis=0).mean(axis=-1)>0).sum()/first_obs.mean(axis=-1).size
    
    if np.all([e.past_traj_obs is not None for e in env.envs]):
        traj_cov = np.array([calc_traj_cov(e.past_traj_obs) for e in env.envs])
        full_cov = calc_traj_cov(np.concatenate([e.past_traj_obs for e in env.envs]))
        data['coverage/{1:03d}_trajs'] = traj_cov.mean()
        data[f'coverage/{env.num_envs:03d}_trajs'] = full_cov
        traj_returns = np.array([e.past_returns[-1] for e in env.envs])
        traj_lens = np.array([len(e.past_traj_obs) for e in env.envs])
        data['charts/returns_hist'] = wandb.Histogram(traj_returns)
        data['charts/traj_lens_hist'] = wandb.Histogram(traj_lens)
        data['charts/returns'] = traj_returns.mean()
        data['charts/traj_lens'] = traj_lens.mean()

    data['per_step/ext_rewards'] = kwargs['rewards'].mean().item()
    data['per_step/int_rewards'] = kwargs['curiosity_rewards'].mean().item()
    data['per_step/ext_rewards_hist'] = wandb.Histogram(kwargs['rewards'].flatten().tolist())
    data['per_step/int_rewards_hist'] = wandb.Histogram(kwargs['curiosity_rewards'].flatten().tolist())

    if args.track and kwargs['update']%10==0:
        plt.figure(figsize=(10, 7))
        plt.subplot(221)
        plt.imshow(env.envs[0].first_obs)
        plt.subplot(222)
        o = kwargs['b_obs'][:, -1].cpu().numpy().std(axis=0).mean(axis=-1)
        plt.imshow(o)
        plt.tight_layout()
        data['coverage/heatmap'] = wandb.Image(plt.gcf())
        plt.close('all')
        
    vids_exist = np.all([e.past_traj_obs is not None for e in env.envs])
    if args.track and vids_exist and kwargs['update']%10==0:
        video = generate_video(env, shape=(2, 2))
        data['charts/video'] = wandb.Video(rearrange(video, 't h w c->t c h w'), fps=15)
    
    # freq_save = int(kwargs['num_updates']//10)
    if args.track and kwargs['update']%40==0:
        torch.save(kwargs['agent'].state_dict(), f"{main_kwargs['run_dir']}/agent_{kwargs['update']:05d}.pt")
        torch.save(kwargs['rnd_model'].state_dict(), f"{main_kwargs['run_dir']}/rnd_{kwargs['update']:05d}.pt")
        
    if 'pbar' in kwargs:
        kwargs['pbar'].set_postfix({key: val for key, val in data.items() if isinstance(val, (int, float))})
    if args.track:
        wandb.log(data)

def main():
    args = parser.parse_args()
    assert args.train_obj in {'ext', 'int'}
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
            # entity=args.wandb_entity,
            project='exploration-distillation',
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

    cb = partial(callback, args=args, main_kwargs=locals())
    ppo_rnd.run(agent, rnd, env, tqdm=tqdm, device=args.device, callback_fn=cb,
                total_timesteps=args.total_timesteps, learning_rate=args.learning_rate, num_steps=args.num_steps,
                anneal_lr=args.anneal_lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
                num_minibatches=args.num_minibatches, update_epochs=args.update_epochs, norm_adv=args.norm_adv,
                clip_coef=args.clip_coef, clip_vloss=args.clip_vloss, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
                # RND arguments
                update_proportion=args.update_proportion, int_coef=args.int_coef, ext_coef=args.ext_coef,
                int_gamma=args.int_gamma, num_iterations_obs_norm_init=args.num_iterations_obs_norm_init,
                )

if __name__=="__main__":
    main()

