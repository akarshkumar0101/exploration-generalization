# requires 4 GB of GPU memory
# python cluster_run.py /data/vision/phillipi/akumar01/exploration-generalization/atari/scripts_ge/finetuning_ppo.sh --mem-gpu 4000 --dir /data/vision/phillipi/akumar01/exploration-generalization/atari --servers freeman-titanxp-1 freeman-titanxp-2 freeman-titanxp-5 freeman-titanxp-6 freeman-titanxp-7 freeman-titanxp-8 freeman-titanxp-9 oliva-titanxp-1 oliva-titanxp-2 --conda-env egb

import os
import sys
import copy

import experiment_utils

sys.path.append("../")
from train import parser

default_config = vars(parser.parse_args())

# with open("../atari_games_57.txt") as f:
#     env_ids_all = "".join(f.readlines()).split("\n")
# with open("../atari_games_train_small.txt") as f:
#     env_ids_train = "".join(f.readlines()).split("\n")
# env_ids_ignore = ["Skiing", "Venture", "VideoPinball"]
# env_ids = [env_id for env_id in env_ids_all if env_id not in env_ids_train and env_id not in env_ids_ignore]

configs = []
for seed in range(8):
    for env_id in ["MsPacman", "Pong", "SpaceInvaders", "StarGunner", "Boxing"]:
    # for env_id in ["Boxing"]:
        for pre_obj in ["ext", "rnd"]:
            for pre_teacher_last_k in [1, 8]:
                config = copy.deepcopy(default_config)

                config["track"] = True
                config["project"] = "egb-atari-ftppo-new"
                config["name"] = f"finetune_ppo_{env_id}_{pre_obj}_history_{pre_teacher_last_k}_{seed}"

                config["device"] = "cuda"
                config["seed"] = seed

                config["env_ids"] = [env_id]
                config["total_steps"] = int(1e6)
                config["n_envs"] = 16
                config["n_steps"] = 128
                config["batch_size"] = 2048
                config["n_updates"] = 4

                config["model"] = "gpt"
                config["ctx_len"] = 4
                config["load_agent"] = f"./data/egb-atari-prev/generalist_30_{pre_obj}_history_{pre_teacher_last_k}_{seed%4}/agent_000000320.pt"
                config["save_agent"] = None

                config["lr"] = 2.5e-4

                config["episodic_life"] = True
                config["ent_coef"] = 0.0  # 0.001

                config["pre_obj"] = pre_obj

                config["n_steps_rnd_init"] = 0

                configs.append(config)

prune = True
python_command = "python train.py"
out_file = os.path.basename(__file__).replace(".py", ".sh")
command_txt = experiment_utils.create_command_txt_from_configs(configs, default_config, prune=prune, python_command=python_command, out_file=out_file)
print(command_txt)
print(f"Saved to {out_file}")
