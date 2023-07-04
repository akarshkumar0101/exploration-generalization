import gymnasium as gym


env = gym.make("ALE/Pitfall-v5", render_mode='human')
obs, info = env.reset()
while True:
    env.render()
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        obs, info = env.reset()
