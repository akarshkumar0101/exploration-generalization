import argparse
from distutils.util import strtobool

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
# general parameters
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--project", type=str, default='project-name')
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default=None)
# viz parameters
parser.add_argument("--freq_viz", type=int, default=100)
parser.add_argument("--freq_save", type=int, default=100)
# algorithm parameters
parser.add_argument("--n_steps", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--coef_entropy", type=float, default=1e-2)

def main(args):
    print(f'Starting run with args: {args}')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.name is not None:
        args.name = args.name.format(**args.__dict__)

    if args.track:
        run = wandb.init(config=args, name=args.name, save_code=True)

    pbar = tqdm(range(args.n_steps))
    for i_step in pbar:
        data = dict(loss=1.1, reward=0.0)

        # if args.track and i_step%args.freq_viz==0:
            # data['viz'] = wandb.Image(...)
        # if args.track and i_step%args.freq_save==0:
            # torch.save(...)

        pbar.set_postfix({k: v for k, v in data.items() if isinstance(v, int) or isinstance(v, float)})
        if args.track:
            wandb.log(data)

    if args.track:
        run.finish()

    return locals()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
