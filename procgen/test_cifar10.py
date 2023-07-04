import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch import nn
from tqdm.auto import tqdm

import wandb
from agent_procgen import IDM
from env_procgen import make_env


def create_net():
    net = nn.Sequential(
        *[
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(10),
        ]
    )
    return net


def main(args):
    if args.name:
        args.name = args.name.format(**args.__dict__)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.track:
        wandb.init(project=args.project, name=args.name, config=args, save_code=True)

    net = create_net()
    net = net.to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)  # , weight_decay=1e-5)

    # if args.track:
    # wandb.watch(idm, log="all", log_freq=args.n_steps // 100)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    losses, accs = [], []

    pbar = tqdm(trainloader)
    for x, y in pbar:
        x, y = x.to(args.device), y.to(args.device)
        logits = net(x)

        ce = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        loss = ce.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        accuracy = (logits.argmax(dim=-1) == y).sum().item() / len(y)
        data = dict(loss=loss.item(), accuracy=accuracy, ppl=np.exp(loss.item()))

        losses.append(loss.item())
        accs.append(accuracy)

        keys_tqdm = ["loss", "accuracy", "ppl"]
        pbar.set_postfix({k: data[k] for k in keys_tqdm})
        if args.track:
            wandb.log(data)

    plt.subplot(121)
    plt.plot(losses)
    plt.subplot(122)
    plt.plot(accs)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="e3b_idm_test2")
parser.add_argument("--name", type=str, default="e3bidmtest_{idm_merge}_{lr}")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--track", default=False, action="store_true")

parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--n-steps", type=lambda x: int(float(x)), default=int(5e3))
parser.add_argument("--batch-size", type=int, default=2048)

parser.add_argument("--idm-merge", type=str, default="both")
parser.add_argument("--idm-normalize", type=lambda x: x.lower() == "true", default=True)
parser.add_argument("--actions", type=str, default="ordinal")

parser.add_argument("--freq-collect", type=lambda x: int(float(x)), default=128)
parser.add_argument("--freq-batch", type=lambda x: int(float(x)), default=1)


import torch
import torchvision

if __name__ == "__main__":
    main(parser.parse_args())
