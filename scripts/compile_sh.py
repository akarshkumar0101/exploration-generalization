import argparse
import itertools
import re

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)


def list_commands(base):
    split = base.split()
    loops = {}
    for i_part, part in enumerate(split):
        if re.fullmatch(r"L\[\d+:\d+:\d+\]", part):
            start, stop, skip = [int(i) for i in part[2:-1].split(":")]
            loops[i_part] = list(range(start, stop, skip))
        if re.fullmatch(r"C\[\w+(,\w+)*\]", part):
            loops[i_part] = part[2:-1].split(",")

    keys, values = list(loops.keys()), list(loops.values())
    combos = list(itertools.product(*values))

    commands = []
    for combo in combos:
        for i_part, val in zip(keys, combo):
            if isinstance(val, int):
                split[i_part] = f"{val:05d}"
            elif isinstance(val, float):
                split[i_part] = f"{val:.3f}"
            elif isinstance(val, str):
                split[i_part] = val
        commands.append(" ".join(split))
    return commands


def main(args):
    with open(args.file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]
        lines = [line for line in lines if not line.startswith("#")]
    commands = []
    for line in lines:
        commands.append(f"# {line}")
        for command in list_commands(line):
            commands.append(command)
        commands.append("")
    for command in commands:
        print(command)


import yaml
import itertools
import argparse


def dict_product(data):
    data = {key: (val if isinstance(val, list) else [val]) for key, val in data.items()}
    return (dict(zip(data, vals)) for vals in itertools.product(*data.values()))


parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)

if __name__ == "__main__":
    # main(parser.parse_args())

    args = parser.parse_args()

    with open(args.file, "r") as f:
        data = yaml.safe_load(f)
    data = dict_product(data)

    commands = []
    for datai in data:
        datai = datai.copy()
        file = datai["python_file"]
        del datai["python_file"]
        commands.append(["python"] + [file] + [f"{key}={val}" for key, val in datai.items()])
    n, l = len(commands), len(commands[0])
    str_lens = [max([len(command[i]) for command in commands]) for i in range(l)]
    commands = [" ".join([command[i].ljust(str_lens[i]) for i in range(l)]) for command in commands]
    commands = "\n".join(commands)

    with open(args.file.replace(".yaml", ".sh"), "w") as f:
        f.write(commands)
