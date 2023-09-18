# requires 4gb gpu memory
# python cluster_run.py /data/vision/phillipi/akumar01/exploration-generalization/atari/scripts_ge/ge_finetune.sh --mem-gpu 4000 --dir /data/vision/phillipi/akumar01/exploration-generalization/atari --servers freeman-titanxp-1 freeman-titanxp-2 freeman-titanxp-5 freeman-titanxp-6 freeman-titanxp-7 freeman-titanxp-8 freeman-titanxp-9 oliva-titanxp-1 oliva-titanxp-2 --conda-env egb

import os
import glob
import sys
import copy

import numpy as np

sys.path.append("../")
import experiment_utils

import ppo
import bc
import ge_bc

# import goexplore

for f in glob.glob("*.sh"):
    os.remove(f)

np.random.seed(0)
with open("../atari_games_57.txt") as f:
    env_ids = "".join(f.readlines()).split("\n")
with open("../atari_games_ignore.txt") as f:
    env_ids_ignore = "".join(f.readlines()).split("\n")
env_ids = np.array([env_id for env_id in env_ids if env_id not in env_ids_ignore])

# print(len(env_ids))
# print(env_ids)

n_train = round(len(env_ids) * 0.85)
n_test = len(env_ids) - n_train
print(f"n_train: {n_train}, n_test: {n_test}")

np.random.seed(143)
env_ids_permuted = np.random.permutation(env_ids).copy()
env_ids_trains, env_ids_tests = [], []
for i_split in range(3):
    mask_test = np.arange(len(env_ids_permuted))
    mask_test = (mask_test >= i_split * 8) & (mask_test < (i_split + 1) * 8)
    mask_train = ~mask_test

    env_ids_trains.append(sorted(list(env_ids_permuted[mask_train])))
    env_ids_tests.append(sorted(list(env_ids_permuted[mask_test])))

train = set.union(*[set(env_ids_train) for env_ids_train in env_ids_trains])
test = set.union(*[set(env_ids_test) for env_ids_test in env_ids_tests])

# print(seed)
# print(f'{len(train)} unique train envs')
# print(f'{len(test)} unique test envs')
env_ids_exploration = ["Pitfall", "MontezumaRevenge", "Venture", "Zaxxon", "Berzerk", "Asteroids", "Frostbite", "PrivateEye"]
# print([env_id in test for env_id in env_ids_exploration])
# print(seed)
print(all([env_id in test for env_id in env_ids_exploration]))

for i_split in range(3):
    print(f"Split {i_split}:")
    print(f"Train: {env_ids_trains[i_split]}")
    print(f"Test: {env_ids_tests[i_split]}")

# ------------------------ CHECKPOINT SPECIALIST ------------------------ #
np.random.seed(0)
default_config = vars(ppo.parser.parse_args())
configs = []
for seed in range(1):
    for env_id in env_ids:
        config = default_config.copy()
        config["track"] = True
        config["project"] = "egb_specialist"
        config["name"] = f"{env_id}_{seed:04d}"

        config["device"] = "cuda"
        config["seed"] = seed

        config["model"] = "cnn_4"
        config["load_ckpt"] = None
        config["save_ckpt"] = f"./data/{config['project']}/{config['name']}/ckpt_{{i_iter}}.pt"
        config["n_ckpts"] = 8

        config["lr"] = 2.5e-4

        config["env_ids"] = [env_id]
        config["n_iters"] = 10000
        config["n_envs"] = 8
        config["n_steps"] = 128
        config["batch_size"] = 256
        config["n_updates"] = 16

        configs.append(config.copy())
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python ppo.py", out_file=f"specialist.sh")
print("Done!")
# ------------------------------------------------------------ #

# ------------------------ CHECKPOINT GENERALIST ------------------------ #
np.random.seed(0)
default_config = vars(bc.parser.parse_args())
configs = []
for seed in range(1):
    for i_split, (env_ids_train, env_ids_test) in enumerate(zip(env_ids_trains, env_ids_tests)):
        for strategy in ["final_ckpts", "all_ckpts"]:
            config = default_config.copy()
            config["track"] = True
            config["project"] = "egb_generalist"
            config["name"] = f"{strategy}_{i_split}_{seed:04d}"

            config["device"] = "cuda"
            config["seed"] = seed

            config["model"] = "trans_32"
            config["load_ckpt"] = None
            config["save_ckpt"] = f"./data/{config['project']}/{config['name']}/ckpt_{{i_iter}}.pt"
            config["n_ckpts"] = 1

            config["lr"] = 2.5e-4

            config["env_ids"] = env_ids_train
            config["n_iters"] = 1000
            config["n_envs"] = 8
            config["n_steps"] = 512
            config["batch_size"] = 8*48*2 # = 768
            config["n_updates"] = 32
            
            config["model_teacher"] = "cnn_4"
            if strategy == "final_ckpts":
                env_id2teachers = lambda env_id: f"./data/egb_specialist/{env_id}_{seed:04d}/ckpt_9999.pt"
            elif strategy == "all_ckpts":
                env_id2teachers = lambda env_id: f"./data/egb_specialist/{env_id}_{seed:04d}/ckpt_*.pt"
            else:
                raise NotImplementedError
            config["load_ckpt_teacher"] = [env_id2teachers(env_id) for env_id in env_ids_train]

            config["i_split"] = i_split
            config["strategy"] = strategy

            configs.append(config.copy())
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python bc.py", out_file=f"cd_generalist.sh")
print("Done!")
# ------------------------------------------------------------ #

# ------------------------ GO-EXPLORE GENERALIST ------------------------ #
np.random.seed(0)
default_config = vars(ge_bc.parser.parse_args())
configs = []
for seed in range(1):
    for i_split, (env_ids_train, env_ids_test) in enumerate(zip(env_ids_trains, env_ids_tests)):
        for strategy in ["best", "leaf"]:
            config = default_config.copy()
            config["track"] = True
            config["project"] = "egb_ge_generalist"
            config["name"] = f"{strategy}_{i_split}_{seed:04d}"

            config["device"] = "cuda"
            config["seed"] = seed

            config["model"] = "trans_32"
            config["load_ckpt"] = None
            config["save_ckpt"] = f"./data/{config['project']}/{config['name']}/ckpt.pt"
            config["n_ckpts"] = 100

            config["lr"] = 2.5e-4

            config["env_ids"] = env_ids_train
            config["n_iters"] = 10000
            config["n_envs"] = 4
            config["n_steps"] = 512
            config["batch_size"] = 384
            config["n_updates"] = 32

            config["ge_data_dir"] = f"../atari/data/ge_specialist/"
            config["strategy"] = strategy
            config["n_archives"] = 200
            config["min_traj_len"] = 100

            config["i_split"] = i_split

            configs.append(config.copy())
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python ge_bc.py", out_file=f"ge_generalist.sh")
print("Done!")
# ------------------------------------------------------------ #

# ------------------------ GO-EXPLORE FINETUNE-BC ------------------------ #
np.random.seed(0)
default_config = vars(bc.parser.parse_args())
configs = []
for seed in range(1):
    for i_split, (env_ids_train, env_ids_test) in enumerate(zip(env_ids_trains, env_ids_tests)):
        for strategy in ["best", "leaf", None]:
            for env_id in env_ids_test:
                config = default_config.copy()
                config["track"] = True
                config["project"] = "egb_ftbc"
                config["name"] = f"{env_id}_{strategy}_{i_split}_{seed:04d}"

                config["device"] = "cuda"
                config["seed"] = seed

                config["model"] = "trans_32"
                if strategy is not None:
                    config["load_ckpt"] = f"./data/egb_ge_generalist/{strategy}_{i_split}_{0:04d}/ckpt.pt"
                config["save_ckpt"] = None
                config["n_ckpts"] = 1

                config["lr"] = 2.5e-4

                config["env_ids"] = [env_id]
                config["n_iters"] = 2000
                config["n_envs"] = 8
                config["n_steps"] = 128
                config["batch_size"] = 256
                config["n_updates"] = 16

                config["ent_coef"] = 0.0
                config["model_teacher"] = "cnn_4"
                config["load_ckpt_teacher"] = [f"./data/egb_specialist/{env_id}_{0:04d}/ckpt_9999.pt"]

                config["i_split"] = i_split
                config["strategy"] = strategy

                configs.append(config.copy())
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python bc.py", out_file=f"ge_ftbc.sh")
print("Done!")
# ------------------------------------------------------------ #

# ------------------------ GO-EXPLORE FINETUNE-PPO ------------------------ #
np.random.seed(0)
default_config = vars(ppo.parser.parse_args())
configs = []
for seed in range(10):
    for i_split, (env_ids_train, env_ids_test) in enumerate(zip(env_ids_trains, env_ids_tests)):
        for strategy in ["best", "leaf", None]:
            for env_id in env_ids_test:
                config = default_config.copy()
                config["track"] = True
                config["project"] = "egb_ftppo"
                config["name"] = f"{env_id}_{strategy}_{i_split}_{seed:04d}"

                config["device"] = "cuda"
                config["seed"] = seed

                config["model"] = "trans_32"
                if strategy is not None:
                    config["load_ckpt"] = f"./data/egb_ge_generalist/{strategy}_{i_split}_{0:04d}/ckpt.pt"
                config["save_ckpt"] = None
                config["n_ckpts"] = 1

                config["lr"] = 2.5e-4

                config["env_ids"] = [env_id]
                config["n_iters"] = 4000
                config["n_envs"] = 8
                config["n_steps"] = 128
                config["batch_size"] = 256
                config["n_updates"] = 16

                config["i_split"] = i_split
                config["strategy"] = strategy

                configs.append(config.copy())
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python ppo.py", out_file=f"ge_ftppo.sh")
print("Done!")
# ------------------------------------------------------------ #
