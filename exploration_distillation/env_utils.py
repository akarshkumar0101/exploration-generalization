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
        if name=='render_mode':
            return 'rgb_array'
        return super().__getattr__(name)
    def reset(self, *args, **kwargs):
        obs = self.env.reset()
        self.last_obs = obs
        return obs, {}
    def step(self, *args, **kwargs):
        obs, reward, done, info = self.env.step(*args, **kwargs)
        self.last_obs = obs
        return obs, reward, done, False, info
    def render(self):
        return self.last_obs
    
class MinerActionRestriction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        udlr = {'UP', 'DOWN', 'LEFT', 'RIGHT'}
        self.possible_actions = []
        for i, action in enumerate(env.unwrapped.env.env.combos):
            if len(action)==0 or (len(action)==1 and action[0] in udlr):
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
        
        self.past_lengths = []
        self.running_length = 0
        
        self.past_actions = []
        self.running_actions = []
        
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.first_obs = obs
        
        self.running_traj_obs = [obs]
        self.running_return = 0
        self.running_length = 0
        self.running_actions = []
        
        return obs, info
        
    def step(self, action, *args, **kwargs):
        obs, reward, term, trunc, info = self.env.step(action, *args, **kwargs)
        self.running_traj_obs.append(obs)
        self.running_return += reward
        self.running_length += 1
        self.running_actions.extend([action])

        if term or trunc:
            self.past_traj_obs = np.stack(self.running_traj_obs)
            self.past_returns.append(self.running_return)
            self.past_lengths.append(self.running_length)
            self.past_actions.append(self.running_actions)

        return obs, reward, term, trunc, info

class EpisodicCoverageRewardMiner(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert hasattr(env, 'running_traj_obs')
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        o = np.stack(self.env.running_traj_obs[-2:])
        self.mask = o.std(axis=0).mean(axis=-1)>1e-3
        return obs, info
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        o = np.stack(self.env.running_traj_obs[-2:])
        mask_change = o.std(axis=0).mean(axis=-1)>1e-3
        reward = (mask_change & (~self.mask)).mean()
        if reward > 0:
            reward = 1.
        # reward *= np.prod(obs.shape[:2])/9. # ~= 1.
        self.mask = self.mask | mask_change
        return obs, reward, term, trunc, info
        
def make_single_env(env_name='procgen-miner-v0', level_id=0, seed=0, video_folder=None, reward_fn='ext'):
    # it's okay to call reset when a term/trunc is emitted (SyncVectorEnv), but that's it
    env = gym_old.make(env_name, num_levels=1, start_level=level_id, distribution_mode='hard')
    env = MyProcgenEnv(env)
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_folder)
    env = MinerActionRestriction(env)
    env = EpisodeStats(env)
    if reward_fn=='eps':
        env = EpisodicCoverageRewardMiner(env)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = RescaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    return env

def make_env(n_envs=10, env_name='procgen-miner-v0', level_id=0, seed=0, video_folder=None, async_=False, reward_fn='ext'):
    if isinstance(level_id, int):
        level_id = [level_id for _ in range(n_envs)]
    env_fns = [partial(make_single_env, env_name=env_name, level_id=level_id[seed],
                       seed=seed*100+i, video_folder=video_folder if seed==0 else None,
                       reward_fn=reward_fn) for i in range(n_envs)]
    env = gym.vector.AsyncVectorEnv(env_fns) if async_ else gym.vector.SyncVectorEnv(env_fns)
    return env