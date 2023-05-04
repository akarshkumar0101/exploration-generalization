import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchinfo
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


# def create_net():
#     net = nn.Sequential(
#         *[
#             nn.Conv2d(6, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(3),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(3),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(3),
#             nn.Flatten(),
#             nn.LazyLinear(64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 5),
#         ]
#     )
#     return net
class IDM(nn.Module):
    def __init__(self, init="kaiming"):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                # nn.LayerNorm([3, 64, 64], elementwise_affine=False),
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                # nn.LayerNorm([32, 64, 64], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                # nn.LayerNorm([32, 64, 64], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3),
                # nn.LayerNorm([32, 21, 21], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                # nn.LayerNorm([32, 21, 21], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3),
                # nn.LayerNorm([32, 7, 7], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                # nn.LayerNorm([32, 7, 7], elementwise_affine=False),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3),
                nn.Flatten(),
                # nn.LayerNorm([128], elementwise_affine=False),
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
        self.encode(torch.randn(1, 64, 64, 3))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if init == "kaiming_in":
                    nn.init.kaiming_normal_(m.weight, mode="fan_in")
                elif init == "kaiming_out":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                elif init == "orthogonal":
                    nn.init.orthogonal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, obs):
        obs = obs.permute(0, 3, 1, 2) / 255.0
        latent = self.encoder(obs)
        latent = latent / latent.norm(dim=-1, keepdim=True)
        return latent

    def forward(self, obs_now, obs_nxt):
        latent_now = self.encode(obs_now)
        latent_nxt = self.encode(obs_nxt)
        latent = torch.cat([latent_now, latent_nxt], dim=-1)
        logits = self.idm(latent)
        return logits, latent_now, latent_nxt


def plot_grad(net):
    gs = []

    for m in net.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            gs.append(m.weight.grad.flatten().abs().mean().item())
            if m in net.encoder.modules():
                start_idm = len(gs) - 0.5

    plt.figure(figsize=(10, 3))
    plt.suptitle("Weight gradient magnitude vs Layers")
    plt.subplot(121)
    plt.plot(gs)
    plt.axvline(x=start_idm)
    plt.subplot(122)
    plt.plot(gs)
    plt.axvline(x=start_idm)
    plt.yscale("log")
    return plt.gcf()


def main(args):
    if args.name:
        args.name = args.name.format(**args.__dict__)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.track:
        wandb.init(project=args.project, name=args.name, config=args, save_code=True)

    net = IDM(args.init)
    torchinfo.summary(net, [(64, 64, 3), (64, 64, 3)])

    net = net.to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)  # , weight_decay=1e-5)

    # if args.track:
    # wandb.watch(idm, log="all", log_freq=args.n_steps // 100)

    pbar = tqdm(range(args.n_steps))
    for i_step in pbar:
        obs_now, obs_nxt, act_now = get_batch(args.batch_size)

        obs_now = torch.from_numpy(obs_now).to(args.device)
        obs_nxt = torch.from_numpy(obs_nxt).to(args.device)
        act_now = torch.from_numpy(act_now).to(args.device)
        logits, v1, v2 = net(obs_now, obs_nxt)

        ce = torch.nn.functional.cross_entropy(logits, act_now, reduction="none")
        loss = ce.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        v1len, v2len = v1.norm(dim=-1).mean(), v2.norm(dim=-1).mean()
        vdist = (v1 - v2).norm(dim=-1).mean()
        accuracy = (logits.argmax(dim=-1) == act_now).sum().item() / len(act_now)
        data = dict(loss=loss.item(), accuracy=accuracy, ppl=np.exp(loss.item()), v1len=v1len.item(), v2len=v2len.item(), vdist=vdist.item())

        if i_step % (args.n_steps // 10) == 0:
            fig = plot_grad(net)
            data["network grad"] = wandb.Image(fig)
            plt.close("all")

        keys_tqdm = ["loss", "accuracy", "ppl"]
        pbar.set_postfix({k: data[k] for k in keys_tqdm})
        if args.track:
            wandb.log(data)


parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="test_e3b_idm_test")
parser.add_argument("--name", type=str, default="run_{lr}_{batch_size}_{seed}")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--track", default=False, action="store_true")

parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--n-steps", type=lambda x: int(float(x)), default=int(2e3))
parser.add_argument("--batch-size", type=int, default=2048)

parser.add_argument("--init", type=str, default="none")

# parser.add_argument("--idm-merge", type=str, default="both")
# parser.add_argument("--idm-normalize", type=lambda x: x.lower() == "true", default=True)
# parser.add_argument("--actions", type=str, default="ordinal")

# parser.add_argument("--freq-collect", type=lambda x: int(float(x)), default=128)
# parser.add_argument("--freq-batch", type=lambda x: int(float(x)), default=1)

if __name__ == "__main__":
    main(parser.parse_args())
