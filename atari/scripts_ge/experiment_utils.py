import copy
import itertools


def nstr(a):
    return None if a.lower() == "none" else str(a)


def nint(a):
    return None if a.lower() == "none" else int(a)


def uargs_to_dict(uargs):
    return dict([tuple(uarg.replace("--", "").split("=")) for uarg in uargs])


def dict_product(data):
    data = {key: (val if isinstance(val, list) else [val]) for key, val in data.items()}
    return (dict(zip(data, vals)) for vals in itertools.product(*data.values()))


def align_configs(configs, default_config, prune=True):
    configs = copy.deepcopy(configs)

    # make sure all configs have the default keys
    for k in default_config.keys():
        for config in configs:
            if k not in config:
                config[k] = default_config[k]
    # assert all(c.keys() == default_config.keys() for c in configs)

    # prune away keys where all configs have the default value
    if prune:
        for k in default_config.keys():
            if all([config[k] == default_config[k] for config in configs]):
                for config in configs:
                    del config[k]

    return configs


def create_arg_list(config):
    arg_list = []
    for key, val in config.items():
        key = f"--{key}"
        if isinstance(val, list):
            arg_list.append(f'{key} {" ".join(val)}')
        else:
            if isinstance(val, str):  # if isinstance(val, str) and (" " in val or "=" in val or "[" in val or "]" in val):
                arg_list.append(f'{key}="{val}"')
            else:
                arg_list.append(f"{key}={val}")
    return arg_list


def create_command_txt(arg_lists, python_command=None, out_file=None):
    n_coms, n_args = len(arg_lists), len(arg_lists[0])
    alens = [max([len(arg_lists[i_com][i_arg]) for i_com in range(n_coms)]) for i_arg in range(n_args)]
    commands = [" ".join([arg_lists[i_com][i_arg].ljust(alens[i_arg]) for i_arg in range(n_args)]) for i_com in range(n_coms)]
    if python_command is not None:
        commands = [f"{python_command} {com}" for com in commands]
    command_txt = "\n".join(commands)
    if out_file is not None:
        with open(out_file, "w") as f:
            f.write(command_txt)
    return command_txt


def create_command_txt_from_configs(configs, default_config=None, prune=True, python_command=None, out_file=None):
    if default_config is not None:
        configs = align_configs(configs, default_config, prune=prune)
    arg_lists = [create_arg_list(config) for config in configs]
    command_txt = create_command_txt(arg_lists, python_command=python_command, out_file=out_file)
    return command_txt
