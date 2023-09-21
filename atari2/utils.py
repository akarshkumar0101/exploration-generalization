import os
base = os.path.dirname(os.path.abspath(__file__))
with open(f"{base}/atari_games_57.txt") as f:
    env_ids_57 = f.read().split("\n")
with open(f"{base}/atari_games_104.txt") as f:
    env_ids_104 = f.read().split("\n")
with open(f"{base}/atari_games_ignore.txt") as f:
    env_ids_ignore = f.read().split("\n")

env_ids_57 = sorted(env_ids_57)
env_ids_104 = sorted(env_ids_104)
env_ids_ignore = sorted(env_ids_ignore)

env_ids_57_ignore = sorted(list(set(env_ids_57) - set(env_ids_ignore)))
env_ids_104_ignore = sorted(list(set(env_ids_104) - set(env_ids_ignore)))
