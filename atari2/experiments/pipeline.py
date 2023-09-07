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

np.random.seed(3)
env_ids_permuted = np.random.permutation(env_ids).copy()
env_ids_trains, env_ids_tests = [], []
for i_split in range(5):
    mask_test = np.arange(len(env_ids_permuted))
    mask_test = (mask_test >= i_split * 8) & (mask_test < (i_split + 1) * 8)
    mask_train = ~mask_test

    env_ids_trains.append(env_ids_permuted[mask_train])
    env_ids_tests.append(env_ids_permuted[mask_test])

train = set.union(*[set(env_ids_train) for env_ids_train in env_ids_trains])
test = set.union(*[set(env_ids_test) for env_ids_test in env_ids_tests])

# print(seed)
# print(f'{len(train)} unique train envs')
# print(f'{len(test)} unique test envs')
env_ids_exploration = ["Pitfall", "MontezumaRevenge", "Venture", "Zaxxon", "Berzerk", "Asteroids", "Frostbite", "PrivateEye"]
# print([env_id in test for env_id in env_ids_exploration])
print(all([env_id in test for env_id in env_ids_exploration]))

for i_split in range(5):
    print(f"Split {i_split}:")
    print(f"Train: {env_ids_trains[i_split]}")
    print(f"Test: {env_ids_tests[i_split]}")


# ------------------------ SPECIALIST ------------------------ #
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

# ------------------------ GENERALIST ------------------------ #
np.random.seed(0)
default_config = vars(bc.parser.parse_args())
configs = []
for seed in range(1):
    for i_split, (env_ids_train, env_ids_test) in enumerate(zip(env_ids_trains, env_ids_tests)):
        for strategy in ["final_ckpt", "all_ckpts"]:
            config = default_config.copy()
            config["track"] = True
            config["project"] = "egb_generalist"
            config["name"] = f"{strategy}_{i_split}_{seed:04d}"

            config["device"] = "cuda"
            config["seed"] = seed

            config["model"] = "trans_64"
            config["load_ckpt"] = None
            config["save_ckpt"] = f"./data/{config['project']}/{config['name']}/ckpt_{{i_iter}}.pt"
            config["n_ckpts"] = 1

            config["lr"] = 2.5e-4

            config["env_ids"] = env_ids_train.tolist()
            config["n_iters"] = 1000
            config["n_envs"] = 8
            config["n_steps"] = 128
            config["batch_size"] = 256
            config["n_updates"] = 16

            config["model_teacher"] = "cnn_4"

            if strategy == "final_ckpt":
                env_id2teachers = lambda env_id: f"./data/egb_specialist/{env_id}_{seed:04d}/ckpt_9999.pt"
            elif strategy == "all_ckpts":
                env_id2teachers = lambda env_id: f"./data/egb_specialist/{env_id}_{seed:04d}/ckpt_*.pt"
            config["load_ckpt_teacher"] = [env_id2teachers(env_id) for env_id in env_ids_train]

            config["i_split"] = i_split
            config["strategy"] = strategy

            configs.append(config.copy())
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python bc.py", out_file=f"generalist.sh")
print("Done!")
# ------------------------------------------------------------ #


# ------------------------ FINETUNE-PPO ------------------------ #
np.random.seed(0)
default_config = vars(ppo.parser.parse_args())
configs = []
for seed in range(1):
    for i_split, (env_ids_train, env_ids_test) in enumerate(zip(env_ids_trains, env_ids_tests)):
        for strategy in ["final_ckpt", "all_ckpts", None]:
            for env_id in env_ids_test:
                config = default_config.copy()
                config["track"] = True
                config["project"] = "egb_ftppo"
                config["name"] = f"{env_id}_{seed:04d}"

                config["device"] = "cuda"
                config["seed"] = seed

                config["model"] = "trans_64"
                if strategy is not None:
                    config["load_ckpt"] = f"./data/egb_generalist/{strategy}_{i_split}_{seed:04d}/ckpt_9999.pt"
                config["save_ckpt"] = None
                config["n_ckpts"] = 1

                config["lr"] = 2.5e-4

                config["env_ids"] = [env_id]
                config["n_iters"] = 10000
                config["n_envs"] = 8
                config["n_steps"] = 128
                config["batch_size"] = 256
                config["n_updates"] = 16

                config["i_split"] = i_split
                config["strategy"] = strategy

                configs.append(config.copy())
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python ppo.py", out_file=f"ftppo.sh")
print("Done!")
# ------------------------------------------------------------ #


# # ------------------------ SPECIALIST ------------------------ #
# # 11200*15/60/80/10 = 3.5 hours
# # python server.py --command_file=~/exploration-generalization/atari/experiments/goexplore/ge_specialist.sh --run_dir=~/exploration-generalization/atari --experiment_dir=~/experiments/ge_specialist/ --job_cpu_mem=1000 --max_jobs_cpu=1 --max_jobs_gpu=10 --conda_env=egb
# print("Creating ge_specialist.sh ...")
# np.random.seed(0)
# default_config = vars(goexplore.parser.parse_args())
# configs = []
# for seed in range(200):
#     for env_id in env_ids:
#         config = default_config.copy()
#         config["track"] = False
#         config["entity"] = None
#         config["project"] = "ge_specialist"
#         config["name"] = f"ge_specialist_{env_id}_{seed:04d}"
#         config["log_hist"] = False

#         config["seed"] = seed

#         config["env_id"] = env_id
#         config["n_iters"] = int(2e3)
#         config["p_repeat"] = 0.0

#         config["h"] = 8  # np.random.randint(4, 20)  # 8
#         config["w"] = 11  # np.random.randint(4, 20)  # 11
#         config["d"] = 8  # np.random.randint(4, 20)  # 8

#         config["max_cells"] = 50000
#         config["use_reward"] = True

#         config["save_archive"] = f"./data/{config['project']}/{config['name']}.npy"
#         configs.append(config.copy())
# command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python goexplore.py", out_file=f"ge_specialist.sh")
# print("Done!")
# # ------------------------------------------------------------ #

# # ------------------------ GENERALIST ------------------------ #
# # 15*48/2/7 = 48 hours
# # python server.py --command_file=~/exploration-generalization/atari/experiments/goexplore/ge_generalist.sh --run_dir=~/exploration-generalization/atari --experiment_dir=~/experiments/ge_generalist/ --job_cpu_mem=10000 --job_gpu_mem=10000 --max_jobs_cpu=1 --max_jobs_gpu=1 --max_jobs_node=1 --conda_env=egb
# print("Creating ge_generalist.sh ...")
# np.random.seed(0)
# default_config = vars(goexplore_train.parser.parse_args())
# configs = []
# for i_split, (env_ids_train, env_ids_test) in enumerate(zip(env_ids_trains, env_ids_tests)):
#     for seed in range(1):
#         for strategy in ["best", "leaf"]:
#             config = copy.deepcopy(default_config)
#             config["track"] = True
#             config["entity"] = None
#             config["project"] = "ge_generalist"
#             config["name"] = f"ge_generalist_{strategy}_{i_split}_{seed:04d}"
#             # config["log_hist"] = False

#             config["seed"] = seed
#             config["device"] = "cuda"

#             config["env_ids"] = env_ids_train.tolist()
#             config["n_iters"] = int(1500)
#             config["n_envs"] = 4
#             config["n_steps"] = 512
#             config["batch_size"] = 384
#             config["n_updates"] = 32

#             config["ctx_len"] = 64
#             config["save_agent"] = f"./data/{config['project']}/{config['name']}.pt"

#             config["lr"] = 1e-4

#             config["ge_data_dir"] = f"./data/ge_specialist/"
#             config["n_archives"] = 40

#             config["strategy"] = strategy
#             config["i_split"] = i_split

#             configs.append(config.copy())
# command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python goexplore_train.py", out_file=f"ge_generalist.sh")
# print("Done!")
# # ------------------------------------------------------------- #

# # ------------------------ FINE TUNING ------------------------ #

# # python server.py --command_file=~/exploration-generalization/atari/experiments/goexplore/ge_finetune_ppo.sh --run_dir=~/exploration-generalization/atari --experiment_dir=~/experiments/ge_finetune_ppo/ --job_cpu_mem=1000 --job_gpu_mem=4000 --max_jobs_cpu=1 --max_jobs_gpu=2 --max_jobs_node=30 --conda_env=egb
# print("Creating ge_finetune_ppo.sh ...")
# np.random.seed(0)
# default_config = vars(train.parser.parse_args())
# configs = []
# for i_split, (env_ids_train, env_ids_test) in enumerate(zip(env_ids_trains, env_ids_tests)):
#     if i_split == 4:  # temporary
#         continue
#     for seed in range(10):
#         for env_id in env_ids_test:
#             for strategy in ["best", "leaf", "none"]:
#                 config = copy.deepcopy(default_config)

#                 config["track"] = True
#                 config["project"] = "ge_finetune_ppo"
#                 config["name"] = f"ge_finetune_ppo_{strategy}_{i_split}_{env_id}_{seed:04d}"

#                 config["device"] = "cuda"
#                 config["seed"] = seed

#                 config["env_ids"] = [env_id]
#                 config["total_steps"] = int(10e6)
#                 config["n_envs"] = 32
#                 config["n_steps"] = 256
#                 config["batch_size"] = 128
#                 config["n_updates"] = 4

#                 config["model"] = "gpt"
#                 config["ctx_len"] = 64
#                 config["load_agent"] = f"./data/ge_generalist/ge_generalist_{strategy}_{i_split}_{seed%1:04d}.pt" if strategy != "none" else None
#                 config["save_agent"] = f"./data/{config['project']}/{config['name']}.pt"
#                 config["strategy"] = strategy

#                 config["lr"] = 2.5e-4

#                 config["episodic_life"] = True
#                 config["ent_coef"] = 0.0  # 0.001

#                 config["n_steps_rnd_init"] = 0

#                 config["i_split"] = i_split

#                 configs.append(config)
# command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python train.py", out_file=f"ge_finetune_ppo.sh")
# print("Done!")
# # ------------------------------------------------------------- #
