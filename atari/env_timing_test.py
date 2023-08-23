import numpy as np
from tqdm.auto import tqdm

import gymnasium as gym
import envpool
import time

n_envs = 64
n_steps = 1024
env_id = "MontezumaRevenge"


def env_fn():
    env = gym.make(f"ALE/{env_id}-v5", frameskip=1, repeat_action_probability=0.0, full_action_space=True)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=1, frame_skip=4, screen_size=210, grayscale_obs=False)
    return env


env_fns = [env_fn for _ in range(n_envs)]

env = gym.vector.SyncVectorEnv(env_fns)
start = time.time()
env.reset()
for t in tqdm(range(n_steps)):
    action = np.random.randint(18, size=n_envs)
    _ = env.step(action)
end = time.time()
print(f"SyncVectorEnv: ")
print(f"Time: {end - start:.2f}s")
print(f"FPS: {n_steps * n_envs / (end - start):.2f}")
print()


env = gym.vector.AsyncVectorEnv(env_fns)
start = time.time()
env.reset()
for t in tqdm(range(n_steps)):
    action = np.random.randint(18, size=n_envs)
    _ = env.step(action)
end = time.time()
print(f"AsyncVectorEnv: ")
print(f"Time: {end - start:.2f}s")
print(f"FPS: {n_steps * n_envs / (end - start):.2f}")
print()


env = envpool.make_gymnasium(
    f"{env_id}-v5", num_envs=n_envs, img_height=210, img_width=210, gray_scale=False, stack_num=1, frame_skip=4, repeat_action_probability=0.0, noop_max=1, use_fire_reset=False, full_action_space=True
)
start = time.time()
env.reset()
for t in tqdm(range(n_steps)):
    action = np.random.randint(18, size=n_envs)
    _ = env.step(action)
end = time.time()
print(f"envpool: ")
print(f"Time: {end - start:.2f}s")
print(f"FPS: {n_steps * n_envs / (end - start):.2f}")
print()
