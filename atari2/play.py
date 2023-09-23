
import matplotlib.pyplot as plt

import gymnasium as gym

env = gym.make(f"ALE/Galaxian-v5", frameskip=1, repeat_action_probability=0.0, full_action_space=True)
env = gym.wrappers.AtariPreprocessing(env, noop_max=1, frame_skip=4, screen_size=210, grayscale_obs=False)

obs, info = env.reset()
print(obs.shape)
while True:
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    plt.imshow(obs)
    plt.show(block=False)
    plt.pause(1/60.)
