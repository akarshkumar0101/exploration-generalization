import argparse

import numpy as np
import torch
import wandb
from procgen import ProcgenEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
# from baselines import logger
# from baselines.common.models import build_impala_cnn
# from baselines.common.mpi_util import setup_mpi_gpus
from stable_baselines3.common.vec_env import VecEnvWrapper, VecExtractDictObs, VecMonitor, VecNormalize


# this file was worth it because now we're using
# much faster env
# ppo from stable baselines

class StoreObs(VecEnvWrapper):
    def __init__(self, venv, n_envs=25, store_limit=1000):
        super().__init__(venv)
        self.n_envs = n_envs
        self.store_limit = store_limit
        self.past_obs = []

    def reset(self):
        obs = self.venv.reset()
        self.past_obs.append(obs[:self.n_envs])
        return obs

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        self.past_obs.append(obs[:self.n_envs])
        self.past_obs = self.past_obs[-self.store_limit:]
        return obs, rew, done, infos


class VecMinerEpisodicCoverageReward(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.pobs, self.mask_episodic = None, None

    def reset(self):
        obs = self.venv.reset()
        self.pobs = obs  # n_envs, h, w, c
        self.mask_episodic = (np.abs(obs - self.pobs) > 1e-3).any(axis=-1)
        return obs

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        mask_change = (np.abs(obs - self.pobs) > 1e-3).any(axis=-1)
        rew_eps = (mask_change & (~self.mask_episodic)).mean(axis=(-1, -2))
        rew_eps = np.sign(rew_eps)  # n_envs
        rew_eps[done] = 0.
        self.mask_episodic = self.mask_episodic | mask_change
        self.pobs = obs
        self.mask_episodic[done] = (np.abs(obs[done] - self.pobs[done]) > 1e-3).any(axis=-1)
        for i in range(self.num_envs):
            info[i]['rew_eps'] = rew_eps[i]
            info[i]['rew_ext'] = rew[i]
        return obs, rew, done, info


class ReturnTracker(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self._ret_ext = None
        self._ret_eps = None
        self.ret_ext = None
        self.ret_eps = None

        self.nsteps = 0

    def reset(self):
        obs = self.venv.reset()
        self._ret_ext = np.zeros(self.num_envs, dtype=np.float32)
        self._ret_eps = np.zeros(self.num_envs, dtype=np.float32)
        self.ret_ext = np.zeros(self.num_envs, dtype=np.float32)
        self.ret_eps = np.zeros(self.num_envs, dtype=np.float32)
        return obs

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        self._ret_ext += np.array([info['rew_ext'] for info in infos])
        self._ret_eps += np.array([info['rew_eps'] for info in infos])

        self.ret_ext[done] = self._ret_ext[done]
        self.ret_eps[done] = self._ret_eps[done]
        self._ret_ext[done] = 0.
        self._ret_eps[done] = 0.
        self.nsteps += 1

        return obs, rew, done, infos


def make_env(n_envs=64, env_name='miner', num_levels=1, start_level=0, distribution_mode='easy'):
    print(env_name)
    venv = ProcgenEnv(num_envs=n_envs, env_name=env_name,
                      num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    # venv.action_space = gym.spaces.MultiDiscrete([venv.action_space.n for _ in range(venv.num_envs)])
    venv = VecExtractDictObs(venv, 'rgb')
    venv = VecMonitor(venv=venv, filename=None)
    venv = VecMinerEpisodicCoverageReward(venv)
    venv = ReturnTracker(venv)
    # venv = VecNormalize(venv=venv, ob=False)
    venv = VecNormalize(venv=venv, norm_obs=False)
    venv = StoreObs(venv, n_envs=25, store_limit=1000)
    return venv


class MyCallback(BaseCallback):
    def __init__(self, args, venv):
        super().__init__()
        self.args = args
        self.venv = venv

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        i_update = self.n_calls // self.args.n_steps
        viz_slow = i_update % max(1, (self.args.n_updates // 20)) == 0
        viz_fast = i_update % max(1, (self.args.n_updates // 1000)) == 0

        data = {}
        if self.args.track and viz_fast:
            # data['train_charts'] =
            pass
        if self.args.track and viz_slow:
            pass
        print()
        print(self.venv.venv.venv.ret_ext.mean())
        print(self.venv.venv.venv.ret_eps.mean())
        print(self.venv.venv.venv.nsteps, self.venv.venv.venv.nsteps * self.venv.num_envs)
        print(self.n_calls)
        print()

        # if 'pbar' in kwargs:
        #     kwargs['pbar'].set_postfix({key: val for key, val in data.items() if isinstance(val, (int, float))})
        if self.args.track:
            wandb.log(data)


def main(args):
    args.n_updates = int(args.timesteps / (args.n_steps * args.n_envs))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.track:
        wandb.init(project=args.project, name=args.name, config=args, save_code=True)

    print('Creating env')
    venv = make_env(args.n_envs, args.env_name, args.n_levels, args.level_start, args.distribution_mode)
    # venv = make_env(args.n_envs, args.env_name, args.n_levels, args.level_start, args.distribution_mode)

    print('Creating PPO')
    ppo = PPO('CnnPolicy', venv, args.lr, args.n_steps, args.batch_size, args.n_epochs, args.gamma, args.gae_lambda,
              args.clip_range, clip_range_vf=None,
              normalize_advantage=True, ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5, use_sde=False,
              sde_sample_freq=-1, target_kl=None, tensorboard_log=None, policy_kwargs=None, verbose=0, seed=None,
              device='cpu', _init_setup_model=True)

    print('Starting learning')
    callback = MyCallback(args, venv)
    ppo.learn(total_timesteps=args.timesteps, callback=callback, log_interval=None, tb_log_name=None,
              reset_num_timesteps=True,
              progress_bar=True)
    return locals()


parser = argparse.ArgumentParser(description='Process procgen training arguments.')

parser.add_argument("--device", type=str, default='cpu', help="device to run on")
parser.add_argument("--seed", type=int, default=0, help='seed')
parser.add_argument("--track", default=False, action='store_const', const=True)
parser.add_argument("--project", type=str, default='exploration-distillation')
parser.add_argument("--name", type=str, default='{env}_{level:05d}_{train_obj}_{pretrain_levels}_{pretrain_obj}')

parser.add_argument("--obj", type=str, default='ext', choices=['ext', 'eps'])

parser.add_argument('--env_name', type=str, default='miner')
parser.add_argument('--n_envs', type=int, default=64)
parser.add_argument('--distribution_mode', type=str, default='easy',
                    choices=["easy", "hard", "exploration", "memory", "extreme"])
parser.add_argument('--n_levels', type=int, default=0)
parser.add_argument('--level_start', type=int, default=0)
parser.add_argument('--timesteps', type=int, default=50_000_000)

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.999)
parser.add_argument('--gae_lambda', type=float, default=0.95)
parser.add_argument('--ent_coef', type=float, default=0.01)

parser.add_argument('--n_steps', type=int, default=256)
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=2048)  # n_minibatches = 256*64/2048 = 8
parser.add_argument('--clip_range', type=float, default=0.2)

if __name__ == '__main__':
    main(parser.parse_args())
