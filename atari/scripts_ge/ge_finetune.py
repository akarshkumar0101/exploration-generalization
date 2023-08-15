import os
import sys
import copy

import experiment_utils

sys.path.append("../")
from train import parser

default_config = vars(parser.parse_args())

with open("../atari_games_57.txt") as f:
    env_ids_all = "".join(f.readlines()).split("\n")
with open("../atari_games_train_small.txt") as f:
    env_ids_train = "".join(f.readlines()).split("\n")
env_ids_ignore = ["Skiing", "Venture", "VideoPinball"]

env_ids = [env_id for env_id in env_ids_all if env_id not in env_ids_train and env_id not in env_ids_ignore]

configs = []
for seed in range(10):
    for env_id in env_ids:
        for strategy in ["best", "random"]:
            config = copy.deepcopy(default_config)

            config["track"] = True
            config["project"] = "ge_ftppo"
            config["name"] = f"ge_ftppo_{strategy}_{env_id}_{seed}"

            config["device"] = "cuda"
            config["seed"] = seed

            config["env_ids"] = [env_id]
            config["total_steps"] = int(40e6)
            config["n_envs"] = 32
            config["n_steps"] = 256
            config["batch_size"] = 128
            config["n_updates"] = 4

            config["model"] = "gpt"
            config["ctx_len"] = 64
            config["load_agent"] = f"./data/ge_train/ge_train_{strategy}_0000.pt"
            config["save_agent"] = f"./data/ge_ftppo/ge_ftppo_{strategy}_{env_id}_{seed}_0000.pt"
            config["strategy"] = strategy

            config["lr"] = 2.5e-4

            config["episodic_life"] = True
            config["ent_coef"] = 0.0  # 0.001

            config["n_steps_rnd_init"] = 0

            configs.append(config)

prune = True
python_command = "python train.py"
out_file = os.path.basename(__file__).replace(".py", ".sh")
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=prune, python_command=python_command, out_file=out_file)
print(command_txt)
print(f"Saved to {out_file}")
