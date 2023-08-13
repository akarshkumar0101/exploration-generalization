# python cluster_run.py /data/vision/phillipi/akumar01/exploration-generalization/atari/scripts/specialist.sh --mem-gpu 5000 --dir /data/vision/phillipi/akumar01/exploration-generalization/atari --servers freeman-titanxp-1 freeman-titanxp-2 fr    eeman-titanxp-3 freeman-titanxp-5 freeman-titanxp-6 freeman-titanxp-7 torralba-titanxp-1 torralba-titanxp-2 torralba-titanxp-3 torralba-titanxp-4 torralba-titanxp-5 torralba-titanxp-7 --conda-env egb
# requires 4 GB of GPU memory

python_file = "goexplore.py"

import sys
import os
import numpy as np

sys.path.append("../")
import goexplore

default_config = vars(goexplore.parser.parse_args())
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
    return commands


# ---------- SHARED CONFIG ---------- #
config["track"] = False
config["entity"] = None
config["project"] = "goexplore"
config["name"] = None
config["log_hist"] = False

config["seed"] = 0

config["env_id"] = "Asteroids"
config["n_iters"] = int(1e3)
config["p_repeat"] = 0.0

config["h"] = 8
config["w"] = 11
config["d"] = 8

config["max_cells"] = 50000
config["use_reward"] = True

config["save_archive"] = None

shared_config = config.copy()

# ---------- STARTING SWEEPING ---------- #
configs = []
sweep_seed = np.arange(100)
sweep_env_id = []
with open("../atari_games_57.txt") as f:
    sweep_env_id += [line.strip() for line in f.readlines()]
# sweep_env_id = sweep_env_id[:40]
# with open("../atari_games_test.txt") as f:
# sweep_env_id += [line.strip() for line in f.readlines()]

np.random.seed(0)
for seed in sweep_seed:
    for env_id in sweep_env_id:
        config = default_config.copy()
        config.update(shared_config.copy())
        config["seed"] = seed
        config["env_id"] = env_id
        # for k in ["project", "name", "load_agent", "save_agent", "load_agent_teacher"]:
        # if config[k] is not None:
        # config[k] = config[k].format(**config)
        config["save_archive"] = f"./data/goexplore/{env_id}_{seed:04d}.npy"
        configs.append(config.copy())

# assert all configs have same key set
assert all(c.keys() == configs[0].keys() for c in configs)

for k in list(configs[0].keys()):
    if all([c[k] == default_config[k] for c in configs]):
        for c in configs:
            del c[k]

arg_lists = [[f"python {python_file}"] + get_arg_list(c) for c in configs]
commands = get_commands(arg_lists)
commands = [f"{c} &" for c in commands]
# commands is a list of strings
# add a "wait" line every 10 commands
wait_every_k = 40
commands = [commands[i : i + wait_every_k] for i in range(0, len(commands), wait_every_k)]
commands = ["\n".join(c) + "\nwait\n" for c in commands]
commands = "\n".join(commands)

print(commands)
out_file = os.path.basename(__file__).replace(".py", ".sh")
with open(out_file, "w") as f:
    f.write(commands)
