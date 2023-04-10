import argparse

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

from agent_procgen import E3B
from env_procgen import make_env

action_strs_miner = ["nothing", "left", "down", "up", "right"]
action_list_miner = np.array([4, 1, 3, 5, 7])


def collect_rollouts():
    env = make_env("miner", "ext", 64, 0, 0, "hard", 0.999, e3b=None, device="cpu")
    obs, info = env.reset()
    obss, actions = [obs], []

    for i in range(256):
        action = np.random.randint(low=0, high=len(action_list_miner), size=env.num_envs)
        obs, rew, done, info = env.step(action_list_miner[action])
        obss.append(obs)
        actions.append(action)
    obss = np.stack(obss)
    actions = np.stack(actions)
    return obss, actions


def main(args):
    if args.name:
        args.name = args.name.format(**args.__dict__)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.track:
        wandb.init(project=args.project, name=args.name, config=args, save_code=True)

    e3b = E3B(64, obs_shape=(64, 64, 3), n_actions=len(action_list_miner), n_features=100, lmbda=0.1)
    e3b.to(args.device)
    opt = torch.optim.Adam(e3b.idm.parameters(), lr=args.lr)  # , weight_decay=1e-5)

    pbar = tqdm(range(args.n_steps))
    for i_batch in pbar:
        if i_batch % args.freq_collect == 0:
            obss, actions = collect_rollouts()
        if i_batch % args.freq_batch == 0:
            i_step = torch.randint(low=0, high=len(obss) - 1, size=(args.batch_size,))
            i_env = torch.randint(low=0, high=64, size=(args.batch_size,))

        obs_now = obss[i_step, i_env]
        obs_nxt = obss[i_step + 1, i_env]
        action_now = actions[i_step, i_env]
        obs_now, obs_nxt = torch.from_numpy(obs_now).to(args.device), torch.from_numpy(obs_nxt).to(args.device)
        action_now = torch.from_numpy(action_now).to(args.device)
        logits = e3b.idm(obs_now, obs_nxt)

        with torch.no_grad():
            v1 = e3b.idm.calc_features(obs_now)
            v2 = e3b.idm.calc_features(obs_nxt)
            v1len = v1.norm(dim=-1).mean()
            v2len = v2.norm(dim=-1).mean()
            vdist = (v1 - v2).norm(dim=-1).mean()

        ce = torch.nn.functional.cross_entropy(logits, action_now, reduction="none")
        loss = ce.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        data = dict(loss=loss.item(), v1len=v1len.item(), v2len=v2len.item(), vdist=vdist.item())
        pbar.set_postfix(**data)
        if args.track:
            wandb.log(data)


parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="e3b_idm_test")
parser.add_argument("--name", type=str, default="e3bidmtest_{freq_collect}_{freq_batch}_{lr}")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--track", default=False, action="store_true")

parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--n-steps", type=lambda x: int(float(x)), default=int(1e6))
parser.add_argument("--batch-size", type=int, default=256)

parser.add_argument("--freq-collect", type=lambda x: int(float(x)), default=64)
parser.add_argument("--freq-batch", type=lambda x: int(float(x)), default=1)

if __name__ == "__main__":
    main(parser.parse_args())
