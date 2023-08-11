# python cluster_run.py /data/vision/phillipi/akumar01/exploration-generalization/atari/scripts/specialist.sh --mem-gpu 5000 --dir /data/vision/phillipi/akumar01/exploration-generalization/atari --servers freeman-titanxp-1 freeman-titanxp-2 fr    eeman-titanxp-3 freeman-titanxp-5 freeman-titanxp-6 freeman-titanxp-7 torralba-titanxp-1 torralba-titanxp-2 torralba-titanxp-3 torralba-titanxp-4 torralba-titanxp-5 torralba-titanxp-7 --conda-env egb
# requires 4 GB of GPU memory


python_file = "train.py"

import sys
import os

sys.path.append("../")
import train

default_config = vars(train.parser.parse_args())
config = default_config.copy()


def get_arg_list(config):
    command_list = []
    for key, val in config.items():
        # key = key.replace("_", "-")
        key = f"--{key}"

        if isinstance(val, list):
            command_list.append(f'{key} {" ".join(val)}')
        else:
            if isinstance(val, str) and (" " in val or "=" in val or "[" in val or "]" in val):
                command_list.append(f'{key}="{val}"')
            else:
                command_list.append(f"{key}={val}")
    return command_list


def get_commands(commands):
    n, l = len(commands), len(commands[0])
    str_lens = [max([len(command[i]) for command in commands]) for i in range(l)]
    commands = [" ".join([command[i].ljust(str_lens[i]) for i in range(l)]) for command in commands]
    commands = "\n".join(commands)
    return commands


# ---------- SHARED CONFIG ---------- #
config["track"] = True
config["entity"] = None
config["project"] = "egb-atari"
config["name"] = "specialist_{env_ids[0]}_{obj}_{seed}"
config["log_video"] = False
config["log_hist"] = False

config["device"] = "cuda"
config["seed"] = 0

config["env_ids"] = "Asteroids"
config["obj"] = "ext"
config["total_steps"] = int(100e6)
config["n_envs"] = 128
config["n_steps"] = 128
config["batch_size"] = 4096
config["n_updates"] = 16

config["model"] = "cnn"
config["ctx_len"] = 4
config["load_agent_history"] = False
config["load_agent"] = None
config["save_agent"] = "./data/{project}/{name}/"
config["full_action_space"] = True

config["lr"] = 2.5e-4
config["lr_warmup"] = True
config["lr_decay"] = "none"
config["max_grad_norm"] = 1.0

config["episodic_life"] = False
config["norm_rew"] = True
config["gamma"] = 0.99
config["gae_lambda"] = 0.95
config["norm_adv"] = True
config["ent_coef"] = 0.001
config["clip_coef"] = 0.1
config["clip_vloss"] = True
config["vf_coef"] = 0.5
config["max_kl_div"] = None

config["pre_obj"] = "ext"
config["train_klbc"] = False
config["model_teacher"] = "cnn"
config["ctx_len_teacher"] = 4
config["load_agent_teacher"] = None
config["teacher_last_k"] = 1
config["pre_teacher_last_k"] = 1

config["n_steps_rnd_init"] = int(100e3)

shared_config = config.copy()

# ---------- STARTING SWEEPING ---------- #
configs = []
sweep_seed = [0]

# sweep_env_id = ["Asteroids", "Alien"]
sweep_env_id = []
with open("../atari_games_train_small.txt") as f:
    sweep_env_id += [line.strip() for line in f.readlines()]
# with open("../atari_games_test.txt") as f:
# sweep_env_id += [line.strip() for line in f.readlines()]

sweep_obj = ["ext", "rnd"]
for seed in sweep_seed:
    for env_id in sweep_env_id:
        for obj in sweep_obj:
            config = default_config.copy()
            config.update(shared_config.copy())
            config["seed"] = seed
            config["env_ids"] = [env_id]
            config["obj"] = obj

            for k in ["project", "name", "load_agent", "save_agent", "load_agent_teacher"]:
                if config[k] is not None:
                    config[k] = config[k].format(**config)
            configs.append(config.copy())


# assert all configs have same key set
assert all(c.keys() == configs[0].keys() for c in configs)

for k in list(configs[0].keys()):
    if all([c[k] == default_config[k] for c in configs]):
        for c in configs:
            del c[k]

arg_lists = [["python train.py"] + get_arg_list(c) for c in configs]
commands = get_commands(arg_lists)

print(commands)
out_file = os.path.basename(__file__).replace(".py", ".sh")
with open(out_file, "w") as f:
    f.write(commands)
