# exploration-generalization

Does exploration generalizes better than RL?


In gymnasium and envpool,
the environments LostLuggage and Skiing only have 9 actions even when full_action_space=True.
Very weird.
In envpool,
the environments Frogger and KingKong do not move until the fire button is pressed (when fire_reset=False).
Very weird.
For these reason, these 4 environments are in atari_games_ignore.txt.

## Ignored Games:
### Doesn't have full action space
```python
```
- LostLuggage
- Skiing

### Needs Fire Reset in Envpool
- Frogger
- KingKong

### Does not reset RAM on actual env.reset
```python
for env_id in env_ids:
    try:
        env = gym.make(f"ALE/{env_id}-v5", frameskip=1, repeat_action_probability=0.0, full_action_space=True)
    except:
        continue
    env = gym.wrappers.AtariPreprocessing(env, noop_max=1, frame_skip=4, screen_size=210, grayscale_obs=False)
    env.reset()
    env.step(0)
    ram = env.unwrapped.ale.getRAM()
    env.reset()
    env.step(0)
    ram1 = env.unwrapped.ale.getRAM()
    if not np.allclose(ram, ram1):
        print(env_id)
```
Output:
- Assault
- BeamRider
- Berzerk
- HauntedHouse
- SirLancelot
- Solaris
- SpaceWar
- Surround
- Tennis
- YarsRevenge





