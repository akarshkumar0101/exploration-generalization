import argparse
import time

import torch

from agent_procgen import IDM, Agent
from env_procgen import make_env


def test_env(args):
    encoder = IDM((64, 64, 3), 15, 10, merge="diff").to(args.device)
    # for env_id in ["miner", "heist", "jumper", "coinrun", "fruitbot", "caveflyer"]:
    for env_id in ["miner"]:
        for distribution_mode in ["easy", "hard"]:
            n_steps, n_envs = 256, 64
            env = make_env(env_id, "ext", n_envs, 0, 0, distribution_mode, 0.999, encoder=encoder, device=args.device, cov=True)
            start = time.time()
            obs, info = env.reset()
            for t in range(n_steps):
                obs, rew, done, info = env.step(env.action_space.sample())
            sps = n_steps * n_envs / (time.time() - start)
            print(f"{env_id:>15s}, {distribution_mode:>10s}, {sps:12.2f} SPS")


def test_rollout(args):
    n_steps, n_envs = 256, 64
    x = torch.randint(0, 255, (n_envs, 64, 64, 3), dtype=torch.uint8, device=args.device)
    agent = Agent((64, 64, 3), 15).to(args.device)
    start = time.time()
    for t in range(n_steps):
        with torch.no_grad():
            agent.get_action_and_value(x)
    sps = n_steps * n_envs / (time.time() - start)
    print(f"{'agent forward':>15s}, {'nolstm':>10s}, {sps:12.2f} SPS")


def test_train(args):
    n_steps, n_envs = 256, 64

    x = torch.randint(0, 255, (256, 8, 64, 64, 3), dtype=torch.uint8, device=args.device)
    agent = Agent((64, 64, 3), 15).to(args.device)
    idm = IDM((64, 64, 3), 15, 10, merge="both").to(args.device)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)

    start = time.time()
    for i_epoch in range(3):
        for i_batch in range(8):
            xb = x.reshape(2048, 64, 64, 3)
            _, logprob, _, _, _ = agent.get_action_and_value(xb)
            # logits = idm(xb, xb)
            v1, v2, logits = idm.forward_smart(x)
            loss = logprob.sum()+logits.sum()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            opt.step()
    sps = n_steps * n_envs / (time.time() - start)
    print(f"{'agent learn':>15s}, {'nolstm':>10s}, {sps:12.2f} SPS")


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")

if __name__ == "__main__":
    args = parser.parse_args()
    start = time.time()
    test_env(args)
    test_rollout(args)
    test_train(args)
    sps = 256 * 64 / (time.time() - start)
    print(sps)
