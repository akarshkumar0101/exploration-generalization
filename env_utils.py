from functools import partial

import gym as gym_old
import gymnasium as gym
import numpy as np

import procgen


class MyProcgenEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        os = env.observation_space
        self.observation_space = gym.spaces.Box(low=os.low, high=os.high, shape=os.shape, dtype=os.dtype)
        self.action_space = gym.spaces.Discrete(env.action_space.n)
        self.last_obs = None

    def __getattr__(self, name: str):
        if name == 'render_mode':
            return 'rgb_array'
        return super().__getattr__(name)

    def reset(self, *args, **kwargs):
        obs = self.env.reset()
        self.last_obs = obs
        info = {}
        return obs, info

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.env.step(*args, **kwargs)
        self.last_obs = obs
        info['rew_ext'] = reward
        return obs, reward, done, False, info

    def render(self):
        return self.last_obs


class MinerActionRestriction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        udlr = {'UP', 'DOWN', 'LEFT', 'RIGHT'}
        self.possible_actions = []
        for i, action in enumerate(env.unwrapped.env.env.combos):
            if len(action) == 0 or (len(action) == 1 and action[0] in udlr):
                self.possible_actions.append(i)
        self.action_space = gym.spaces.Discrete(len(self.possible_actions))

    def step(self, action):
        return self.env.step(self.possible_actions[action])


# class RescaleObservation(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         os = env.observation_space
#         self.observation_space = gym.spaces.Box(low=os.low, high=os.high/255., shape=os.shape, dtype=np.float32)
#     def observation(self, obs):
# return (obs/255.).astype(np.float32)

class EpisodeStats(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.first_obs = None
        self.past_traj_obs = None
        self.running_traj_obs = []

        self.past_returns = []
        self.running_return = 0

        self.past_returns_ext = []
        self.running_return_ext = 0
        self.past_returns_eps = []
        self.running_return_eps = 0

        self.past_lengths = []
        self.running_length = 0

        self.past_actions = []
        self.running_actions = []

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.first_obs = obs

        self.running_traj_obs = [obs]
        self.running_return = 0
        self.running_return_ext = 0
        self.running_return_eps = 0
        self.running_length = 0
        self.running_actions = []

        return obs, info

    def step(self, action, *args, **kwargs):
        obs, reward, term, trunc, info = self.env.step(action, *args, **kwargs)
        self.running_traj_obs.append(obs)
        self.running_return += reward
        self.running_return_ext += info['rew_ext']
        self.running_return_eps += info['rew_eps']
        self.running_length += 1
        self.running_actions.extend([action])

        if term or trunc:
            self.past_traj_obs = np.stack(self.running_traj_obs)
            self.past_returns.append(self.running_return)
            self.past_returns_ext.append(self.running_return_ext)
            self.past_returns_eps.append(self.running_return_eps)
            self.past_lengths.append(self.running_length)
            self.past_actions.append(self.running_actions)

        return obs, reward, term, trunc, info


class EpisodicCoverageRewardMiner(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # assert hasattr(env, 'running_traj_obs')
        # self.mask_global = self.env.running_traj_obs[-1].max(axis=-1) <-1e9

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.pobs = obs
        self.mask_episodic = (np.abs(obs - self.pobs) > 1e-3).any(axis=-1)
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        mask_change = (np.abs(obs - self.pobs) > 1e-3).any(axis=-1)
        info['rew_eps'] = np.sign((mask_change & (~self.mask_episodic)).mean())
        info['rew_epd'] = info['rew_eps'] - (5. if term else 0)
        info['rew_int'] = info['rew_eps']
        self.mask_episodic = self.mask_episodic | mask_change
        self.pobs = obs
        return obs, reward, term, trunc, info


class RewardSelector(gym.Wrapper):
    def __init__(self, env, reward_fn='ext'):
        super().__init__(env)
        self.reward_fn = reward_fn

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        reward = info[f'rew_{self.reward_fn}']
        return obs, reward, term, trunc, info


def make_single_env(env_name='miner', level_start=0, n_levels=1,
                    seed=0, video_folder=None, reward_fn='ext', **kwargs):
    # it's okay to call reset when a term/trunc is emitted (SyncVectorEnv), but that's it
    env = gym_old.make(f'procgen-{env_name}-v0', num_levels=n_levels, start_level=level_start, **kwargs)
    env = MyProcgenEnv(env)
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_folder)
    env = MinerActionRestriction(env)
    env = EpisodicCoverageRewardMiner(env)
    env = RewardSelector(env, reward_fn=reward_fn)
    env = EpisodeStats(env)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = RescaleObservation(env)
    env = gym.wrappers.FrameStack(env, 1)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    return env


def make_env(n_envs=10, env_name='miner', level_start=0, n_levels=1,
             seed=0, video_folder=None, reward_fn='ext', async_=False, **kwargs):
    if isinstance(env_name, str):
        env_name = [env_name for _ in range(n_envs)]
    if isinstance(level_start, int):
        level_start = [level_start for _ in range(n_envs)]
    if isinstance(n_levels, int):
        n_levels = [n_levels for _ in range(n_envs)]
    assert isinstance(env_name, list) and len(env_name) == n_envs
    assert isinstance(level_start, list) and len(level_start) == n_envs
    assert isinstance(n_levels, list) and len(n_levels) == n_envs

    env_fns = [partial(make_single_env, env_name=env_name[i], level_start=level_start[i], n_levels=n_levels[i],
                       seed=seed * 100 + i, video_folder=video_folder if i == 0 else None,
                       reward_fn=reward_fn, **kwargs) for i in range(n_envs)]
    env = gym.vector.AsyncVectorEnv(env_fns) if async_ else gym.vector.SyncVectorEnv(env_fns)
    return env


if __name__=='__main__':
    make_env()