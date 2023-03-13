
import argparse
import itertools
import re

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)

def list_commands(base):
    split = base.split()
    loops = {}
    for i_part, part in enumerate(split):
        if re.fullmatch(r'L\[\d+:\d+:\d+\]', part):
            start, stop, skip = [int(i) for i in part[2:-1].split(':')]
            loops[i_part] = list(range(start, stop, skip))
        if re.fullmatch(r'C\[\w+(,\w+)*\]', part):
            loops[i_part] = part[2:-1].split(',')
        
    keys, values = list(loops.keys()), list(loops.values())
    combos = list(itertools.product(*values))

    commands = []
    for combo in combos:
        for i_part, val in zip(keys, combo):
            if isinstance(val, int):
                split[i_part] = f'{val:05d}'
            elif isinstance(val, float):
                split[i_part] = f'{val:.3f}'
            elif isinstance(val, str):
                split[i_part] = val
        commands.append(' '.join(split))
    return commands

def main(args):
    with open(args.file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]
    commands = []
    for line in lines:
        for command in list_commands(line):
            commands.append(command)
    for command in commands:
        print(command)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
