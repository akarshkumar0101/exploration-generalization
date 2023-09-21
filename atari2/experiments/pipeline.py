# requires 4gb gpu memory
# python cluster_run.py /data/vision/phillipi/akumar01/exploration-generalization/atari/scripts_ge/ge_finetune.sh --mem-gpu 4000 --dir /data/vision/phillipi/akumar01/exploration-generalization/atari --servers freeman-titanxp-1 freeman-titanxp-2 freeman-titanxp-5 freeman-titanxp-6 freeman-titanxp-7 freeman-titanxp-8 freeman-titanxp-9 oliva-titanxp-1 oliva-titanxp-2 --conda-env egb

import os
import glob
import sys
import copy

import numpy as np

import experiment_utils

sys.path.append("../")
import ge
import ppo
import bc
import ge_bc
import utils

for f in glob.glob("*.sh"):
    os.remove(f)

np.random.seed(0)
env_ids_tests = np.random.permutation(utils.env_ids_57_ignore).reshape(4, 14).tolist()
env_ids_tests = [sorted(env_ids_test) for env_ids_test in env_ids_tests]
env_ids_trains = [sorted(list(set(utils.env_ids_104_ignore)-set(env_ids_test))) for env_ids_test in env_ids_tests]

for i_split, (env_ids_train, env_ids_test) in enumerate(zip(env_ids_trains, env_ids_tests)):
    print(f'Split: {i_split}')
    print('------------------------------------------------------------------------')
    print(' '.join(env_ids_train))
    print()
    print(' '.join(env_ids_test))
    print('------------------------------------------------------------------------')
    print()
    print()

# ------------------------ GO-EXPLORE-RANDOM ------------------------ #
# 100*200*(15/60)*(1/40/5) = 25 hours
# python server.py --command_file=~/exploration-generalization/atari/experiments/goexplore/ge_specialist.sh --run_dir=~/exploration-generalization/atari --experiment_dir=~/experiments/ge_specialist/ --job_cpu_mem=1000 --max_jobs_cpu=1 --max_jobs_gpu=10 --conda_env=egb
np.random.seed(0)
default_config = vars(ge.parser.parse_args())
configs = []
for seed in range(200):
    for env_id in utils.env_ids_104_ignore:
        config = default_config.copy()
        config["track"] = False
        config["entity"] = None
        config["project"] = "egb_goexplore_rand"
        config["name"] = f"{env_id}_{seed:04d}"
        config["log_hist"] = False

        config["seed"] = seed

        config["env_id"] = env_id
        config["n_iters"] = int(2e3)
        config["p_repeat"] = 0.0

        config["h"] = np.random.randint(4, 20)  # 8
        config["w"] = np.random.randint(4, 20)  # 11
        config["d"] = np.random.randint(4, 20)  # 8

        config["max_cells"] = 50000
        config["use_reward"] = True

        config["save_archive"] = f"./data/{config['project']}/{config['name']}.npy"
        configs.append(config.copy())
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python ge.py", out_file=f"goexplore_rand.sh")
print("Done!")
# ------------------------------------------------------------ #

    
# ------------------------ GO-EXPLORE ------------------------ #
# 100*200*(15/60)*(1/40/5) = 25 hours
# python server.py --command_file=~/exploration-generalization/atari/experiments/goexplore/ge_specialist.sh --run_dir=~/exploration-generalization/atari --experiment_dir=~/experiments/ge_specialist/ --job_cpu_mem=1000 --max_jobs_cpu=1 --max_jobs_gpu=10 --conda_env=egb
np.random.seed(0)
default_config = vars(ge.parser.parse_args())
configs = []
for seed in range(200):
    for env_id in utils.env_ids_104_ignore:
        config = default_config.copy()
        config["track"] = False
        config["entity"] = None
        config["project"] = "egb_goexplore"
        config["name"] = f"{env_id}_{seed:04d}"
        config["log_hist"] = False

        config["seed"] = seed

        config["env_id"] = env_id
        config["n_iters"] = int(2e3)
        config["p_repeat"] = 0.0

        config["h"] = 8  # np.random.randint(4, 20)  # 8
        config["w"] = 11  # np.random.randint(4, 20)  # 11
        config["d"] = 8  # np.random.randint(4, 20)  # 8

        config["max_cells"] = 50000
        config["use_reward"] = True

        config["save_archive"] = f"./data/{config['project']}/{config['name']}.npy"
        configs.append(config.copy())
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python ge.py", out_file=f"goexplore.sh")
print("Done!")
# ------------------------------------------------------------ #

# ------------------------ SPECIALISTS ------------------------ #
np.random.seed(0)
default_config = vars(ppo.parser.parse_args())
configs = []
for seed in range(1):
    for env_id in utils.env_ids_57_ignore:
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
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python ppo.py", out_file=f"specialists.sh")
print("Done!")
# ------------------------------------------------------------ #

# # ------------------------ CHECKPOINT GENERALIST ------------------------ #
# np.random.seed(0)
# default_config = vars(bc.parser.parse_args())
# configs = []
# for seed in range(1):
#     for i_split, (env_ids_train, env_ids_test) in enumerate(zip(env_ids_trains, env_ids_tests)):
#         for strategy in ["final_ckpts", "all_ckpts"]:
#             config = default_config.copy()
#             config["track"] = True
#             config["project"] = "egb_generalist"
#             config["name"] = f"{strategy}_{i_split}_{seed:04d}"

#             config["device"] = "cuda"
#             config["seed"] = seed

#             config["model"] = "trans_32"
#             config["load_ckpt"] = None
#             config["save_ckpt"] = f"./data/{config['project']}/{config['name']}/ckpt.pt"
#             config["n_ckpts"] = 50

#             config["lr"] = 2.5e-4

#             config["env_ids"] = env_ids_train
#             config["n_iters"] = 1000
#             config["n_envs"] = 8
#             config["n_steps"] = 512
#             config["batch_size"] = 8*48*2 # = 768
#             config["n_updates"] = 32
            
#             config["model_teacher"] = "cnn_4"
#             if strategy == "final_ckpts":
#                 env_id2teachers = lambda env_id: f"./data/egb_specialist/{env_id}_{seed:04d}/ckpt_9999.pt"
#             elif strategy == "all_ckpts":
#                 env_id2teachers = lambda env_id: f"./data/egb_specialist/{env_id}_{seed:04d}/ckpt_*.pt"
#             else:
#                 raise NotImplementedError
#             config["load_ckpt_teacher"] = [env_id2teachers(env_id) for env_id in env_ids_train]

#             config["i_split"] = i_split
#             config["strategy"] = strategy

#             configs.append(config.copy())
# command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=True, python_command="python bc.py", out_file=f"cd_generalist.sh")
# print("Done!")
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
