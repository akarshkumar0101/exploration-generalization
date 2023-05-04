import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

import wandb
from agent_procgen import IDM
from env_procgen import make_env


def get_batch(batch_size=128):
    env = make_env("miner", "ext", 128, 0, 0, "easy", gamma=0.999, latent_keys=[], device="cpu", cov=False, actions="ordinal")
    obs, info = env.reset()
    x, y = [obs], []
    for i in range(256):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        x.append(obs)
        y.append(action)
    x = np.stack(x)
    y = np.stack(y)

    n_steps, n_envs = y.shape
    i_step, i_env = np.random.randint(0, n_steps, size=batch_size), np.arange(batch_size) % n_envs
    obs_now, obs_nxt, act_now = x[i_step, i_env], x[i_step + 1, i_env], y[i_step, i_env]
    return obs_now, obs_nxt, act_now


def create_net():
    net = nn.Sequential(
        *[
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
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

    pbar = tqdm(range(args.n_steps))
    for i_batch in pbar:
        obs_now, obs_nxt, act_now = get_batch(args.batch_size)

        obs_now = torch.from_numpy(obs_now).to(args.device)
        obs_nxt = torch.from_numpy(obs_nxt).to(args.device)
        act_now = torch.from_numpy(act_now).to(args.device)
        x = torch.cat([obs_now, obs_nxt], dim=-1).permute(0, 3, 1, 2) / 255.0
        logits = net(x)

        ce = torch.nn.functional.cross_entropy(logits, act_now, reduction="none")
        loss = ce.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        # v1len, v2len = v1.norm(dim=-1).mean(), v2.norm(dim=-1).mean()
        # vdist = (v1 - v2).norm(dim=-1).mean()
        accuracy = (logits.argmax(dim=-1) == act_now).sum().item() / len(act_now)
        # data = dict(loss=loss.item(), v1len=v1len.item(), v2len=v2len.item(), vdist=vdist.item(), accuracy=accuracy)
        data = dict(loss=loss.item(), accuracy=accuracy, ppl=np.exp(loss.item()))

        keys_tqdm = ["loss", "accuracy", "ppl"]
        pbar.set_postfix({k: data[k] for k in keys_tqdm})
        if args.track:
            wandb.log(data)


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

if __name__ == "__main__":
    main(parser.parse_args())


class IDM(nn.Module):
    def __init__(self):
        super().__init__()
        print('updated')
        self.encoder = nn.Sequential(
            *[
                nn.LayerNorm([3, 64, 64], elementwise_affine=False),
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.LayerNorm([32, 64, 64], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.LayerNorm([32, 64, 64], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3),
                nn.LayerNorm([32, 21, 21], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.LayerNorm([32, 21, 21], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3),
                nn.LayerNorm([32, 7, 7], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.LayerNorm([32, 7, 7], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3),
                nn.Flatten(),
                nn.LayerNorm([128], elementwise_affine=False),
                nn.LazyLinear(64),
                nn.ReLU(),
            ]
        )
        self.idm = nn.Sequential(
            *[
                nn.Linear(64 * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 5),
            ]
        )

    def forward(self, obs_now, obs_nxt):
        obs_now = obs_now.permute(0, 3, 1, 2) / 255.0
        obs_nxt = obs_nxt.permute(0, 3, 1, 2) / 255.0
        latent_now = self.encoder(obs_now)
        latent_now = latent_now / latent_now.norm(dim=-1, keepdim=True)
        latent_nxt = self.encoder(obs_nxt)
        latent_nxt = latent_nxt / latent_nxt.norm(dim=-1, keepdim=True)
        latent = torch.cat([latent_now, latent_nxt], dim=-1)
        logits = self.idm(latent)
        return logits
