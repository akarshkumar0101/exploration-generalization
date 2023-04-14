import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

import wandb
from agent_procgen import IDM
from env_procgen import make_env


def collect_rollouts(actions="all"):
    env = make_env("miner", "ext", 64, 0, 0, "hard", 0.999, encoder=None, device="cpu", actions=actions)
    obs, info = env.reset()
    obss, actions = [obs], []
    for i in range(1024):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        obss.append(obs)
        actions.append(action)
    obss = np.stack(obss)
    actions = np.stack(actions)
    return env, obss, actions


def main(args):
    if args.name:
        args.name = args.name.format(**args.__dict__)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.track:
        wandb.init(project=args.project, name=args.name, config=args, save_code=True)

    env = make_env("miner", "ext", 64, 0, 0, "hard", 0.999, encoder=None, device="cpu", actions=args.actions)
    idm = IDM(obs_shape=(64, 64, 3), n_actions=env.single_action_space.n, n_features=64, normalize=args.idm_normalize, merge=args.idm_merge)
    idm.to(args.device)
    opt = torch.optim.Adam(idm.parameters(), lr=args.lr)  # , weight_decay=1e-5)

    if args.track:
        wandb.watch(idm, log="all", log_freq=args.n_steps // 100)

    pbar = tqdm(range(args.n_steps))
    for i_batch in pbar:
        if i_batch % args.freq_collect == 0:
            env, obss, actions = collect_rollouts(actions=args.actions)
        if i_batch % args.freq_batch == 0:
            i_step = torch.randint(low=0, high=len(obss) - 1, size=(args.batch_size,))
            i_env = torch.randint(low=0, high=64, size=(args.batch_size,))

        obs_now = torch.from_numpy(obss[i_step, i_env]).to(args.device)
        obs_nxt = torch.from_numpy(obss[i_step + 1, i_env]).to(args.device)
        action_now = torch.from_numpy(actions[i_step, i_env]).to(args.device)
        v1, v2, logits = idm.forward(obs_now, obs_nxt)

        ce = torch.nn.functional.cross_entropy(logits, action_now, reduction="none")
        loss = ce.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        v1len, v2len = v1.norm(dim=-1).mean(), v2.norm(dim=-1).mean()
        vdist = (v1 - v2).norm(dim=-1).mean()
        accuracy = (logits.argmax(dim=-1) == action_now).sum().item() / len(action_now)
        data = dict(loss=loss.item(), v1len=v1len.item(), v2len=v2len.item(), vdist=vdist.item(), accuracy=accuracy)

        if i_batch % (args.n_steps // 50) == 0:
            n_actions = env.single_action_space.n
            fig = plt.figure()
            plt.barh(np.arange(n_actions), [np.e**ce[action_now==i].mean().item() for i in range(n_actions)])
            plt.yticks(np.arange(n_actions), labels=env.action_meanings)
            plt.xlim(0, n_actions)
            plt.grid()
            data['action_losses'] = wandb.Image(fig)
            plt.close('all')

        keys_tqdm = ['loss', 'v1len', 'v2len', 'vdist', 'accuracy']
        pbar.set_postfix({k: data[k] for k in keys_tqdm})
        if args.track:
            wandb.log(data)


parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="e3b_idm_test2")
parser.add_argument("--name", type=str, default="e3bidmtest_{idm_merge}_{lr}")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--track", default=False, action="store_true")

parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--n-steps", type=lambda x: int(float(x)), default=int(5e4))
parser.add_argument("--batch-size", type=int, default=2048)

parser.add_argument("--idm-merge", type=str, default="both")
parser.add_argument("--idm-normalize", type=lambda x: x.lower()=='true', default=True)
parser.add_argument("--actions", type=str, default="ordinal")

parser.add_argument("--freq-collect", type=lambda x: int(float(x)), default=128)
parser.add_argument("--freq-batch", type=lambda x: int(float(x)), default=1)

if __name__ == "__main__":
    main(parser.parse_args())
