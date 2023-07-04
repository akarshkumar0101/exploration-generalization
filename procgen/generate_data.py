import gym
from procgen import ProcgenEnv

import numpy as np


# Converts vanilla procgen env into something much better
class ProcgenWrapper(gym.Wrapper):
    def __init__(self, env):
        # env = gym.wrappers.TransformObservation(env, lambda obs: obs["rgb"])
        super().__init__(env)
        self.single_action_space = env.action_space
        self.action_space = gym.spaces.MultiDiscrete([self.single_action_space.n] * self.num_envs)
        self.single_observation_space = env.observation_space["rgb"]
        self.observation_space = None  # TODO implement this
        self.is_vector_env = True
        self.action_meanings = ["leftdown", "left", "leftup", "down", "noop", "up", "rightdown", "right", "rightup", "d", "a", "w", "s", "q", "e"]

    def reset(self):
        obs = self.env.reset()
        obs = obs["rgb"]
        return obs, {}

    def step(self, action):
        obs, rew, done, infos = self.env.step(action)
        obs = obs["rgb"]
        return obs, rew, done, {}


# Only actions that are up, down, left, right, noop
class OrdinalActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        am_udlr = ["noop", "up", "down", "left", "right"]
        self.i_actions = np.array([i for i, am in enumerate(self.action_meanings) if am in am_udlr])
        self.action_meanings = [self.action_meanings[i] for i in self.i_actions]
        self.single_action_space = gym.spaces.Discrete(len(self.action_meanings))
        self.action_space = gym.spaces.MultiDiscrete([self.single_action_space.n] * self.num_envs)

    def step(self, action):
        action = self.i_actions[action]
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info


def collect_rollouts(n_envs=64, n_steps=128, ordinal_actions=True):
    env = ProcgenEnv(n_envs, "miner", distribution_mode="hard", num_levels=0, start_level=0)
    env = ProcgenWrapper(env)
    if ordinal_actions:
        env = OrdinalActions(env)

    obs, info = env.reset()
    obss, actions = [obs], []
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        obss.append(obs)
        actions.append(action)
    obss = np.stack(obss)
    actions = np.stack(actions)

    obs_now, obs_nxt, action_now = obss[:-1], obss[1:], actions
    # obs_now    is the observation at current timestep. shape: (n_steps, n_envs, 64, 64, 3), dtype: uint8, range: [0, 255]
    # obs_nxt    is the observation at    next timestep. shape: (n_steps, n_envs, 64, 64, 3), dtype: uint8, range: [0, 255]
    # action_now is the      action at current timestep. shape: (n_steps, n_envs,          ), dtype: uint8, range: [0,  15]
    return env, obs_now, obs_nxt, action_now


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_envs, n_steps = 64, 128

    env, obs_now, obs_nxt, action_now = collect_rollouts(n_envs, n_steps, ordinal_actions=True)

    # code to visualize samples from this dataset
    for i in range(10):
        i_env, i_step = np.random.randint(0, n_envs), np.random.randint(0, n_steps)
        plt.subplot(121)
        plt.imshow(obs_now[i_step, i_env])
        plt.subplot(122)
        plt.imshow(obs_nxt[i_step, i_env])
        plt.suptitle(env.action_meanings[action_now[i_step, i_env]])
        plt.show()
