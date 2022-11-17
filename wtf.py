import argparse
from distutils.util import strtobool

import torch
from torch import nn
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from gym import spaces

import wandb

from env_lava_grid import LavaGrid


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class ImitationExplorer(nn.Module):
    def __init__(self, env, force_random=False):
        super().__init__()
        self.n_inputs = np.prod(env.observation_space.shape)
        self.n_outputs = env.action_space.n
        self.encoder = nn.Sequential(
            nn.Flatten(-2, -1),
            layer_init(nn.Linear(self.n_inputs, 20)),
            nn.Tanh(),
            layer_init(nn.Linear(20, 20)),
            nn.Tanh(),
            layer_init(nn.Linear(20, 20)),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(20, 20)),
            nn.Tanh(),
            layer_init(nn.Linear(20, 1)),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(20, 20)),
            nn.Tanh(),
            layer_init(nn.Linear(20, self.n_outputs)),
        )
        self.force_random = force_random
        
    def get_logits_values(self, x):
        x = self.encoder(x)
        logits, values = self.actor(x), self.critic(x)
        return logits, values
    
    def get_action_and_value(self, x, action=None, ret_logits=False):
        x = self.encoder(x)
        logits, values = self.actor(x), self.critic(x)
        if self.force_random:
            logits, values = torch.zeros_like(logits), torch.zeros_like(values)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values
    
def step_policy(ge, explorer, opt, calc_prod, n_steps, batch_size=100, coef_entropy=1e-1, viz=False, device=None):
    # list of tuples (snapshot, obs, action, reward, done)
    obs = torch.stack([trans[1] for node in ge for trans in node.traj])
    action = torch.stack([trans[2] for node in ge for trans in node.traj])
    # mask_done = torch.stack([trans[4] for node in ge for trans in node.traj])
    mask_done = torch.stack([node.done for node in ge])
    prod = calc_prod(ge)
    prod_norm = (prod-prod[~mask_done].mean())/(prod[~mask_done].std()+1e-9)
    r = prod_norm[1:]
    
    losses = []
    entropies = []
    logits_list = []
    for i_batch in range(n_steps):
        idx = torch.randperm(len(obs))[:batch_size]
        b_obs, b_action, b_r = obs[idx].to(device), action[idx].to(device), r[idx].to(device)
        # if norm_batch:
            # b_prod = (b_prod-b_prod.mean())/(b_prod.std()+1e-9)
            
        logits, values = explorer.get_logits_values(b_obs)
        # logits_aug = b_r[:, None]*logits
        # logits_aug = (1./b_r[:, None])*logits
        logits_aug = torch.sign(b_r[:, None])*logits
        dist = torch.distributions.Categorical(logits=logits_aug)
        log_prob = dist.log_prob(b_action)
        entropy = dist.entropy()
        
        loss_data = (-log_prob*b_r.abs()).mean()
        loss_entropy = -entropy.mean()
        
        loss = loss_data + coef_entropy*loss_entropy

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        entropies.append(entropy.mean().item())
        logits_list.append(logits.mean().item())
        
        # pbar.set_postfix(loss=loss.item())
    losses = torch.tensor(losses)
    # print(f'Reduced loss from {losses[0].item()} to {losses[-1].item()}')

    if viz:
        plt.figure(figsize=(15, 5))
        plt.subplot(131); plt.plot(losses); plt.title('loss vs time')
        plt.subplot(132); plt.plot(entropies); plt.title('entropy vs time')
        plt.subplot(133); plt.plot(logits_list); plt.title('logits mean vs time')
        plt.show()
        
        # plt.plot(logits.softmax(dim=-1).mean(dim=-2).detach().cpu().numpy())
        # plt.hist(logits.argmax(dim=-1).detach().cpu().numpy())
        # plt.ylim(.23, .27)
        # plt.title('avg prob distribution')
        # plt.show()
        
        # plt.scatter(b_action.detach().cpu().numpy(), log_prob.detach().cpu().numpy())
        # plt.show()
        # plt.scatter(b_prod.cpu().numpy(), loss1.detach().cpu().numpy())
        # plt.show()
        
        # for i in range(4):
            # print(f'Action {i}')
            # print(log_probs[batch_actions==i].mean().item())
    

# def train_agent(args):
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
    
#     callbacks = []
#     if args.track:
#         run = wandb.init(
#             config=args,
#             name=args.exp_name,
#             sync_tensorboard=True,
#             monitor_gym=True,
#             save_code=True)
#         callback = WandbCallback(verbose=2)
#                                  # model_save_path=f"models/{run.id}", 
#                                  # model_save_freq=args.n_steps//5,
#                                  # gradient_save_freq=args.n_steps//5)
#                                  # model_save_freq=None,
#                                  # gradient_save_freq=None,)
#         callbacks.append(callback)
#     # folder_tensorboard = f"runs/{run.id}/tb" if args.track else None
#     # folder_video = f"runs/{run.id}/videos" if args.track else None
#     folder_tensorboard = f"runs/{run.id}" if args.track else None
#     folder_video = f"videos/{run.id}" if args.track else None
    
#     all_elems = ['empty', 'sand', 'water', 'wall', 'plant', 'stone', 'lava']
#     kwargs_pcg = dict(hw=(64,64), elems=all_elems[:args.n_elems], num_tasks=args.num_tasks, 
#                       num_lines=args.n_lines, num_circles=args.n_circles, num_squares=args.n_squares, has_empty_path=False)
    
#     if args.env=='sand':
#         env = envs.PWSandEnv(False, kwargs_pcg, flat_action=args.flat_action, device=args.device)
#     elif args.env=='draw':
#         # temporary different settings for draw enviornment
#         kwargs_pcg = dict(hw=(64,64), elems=['sand'], num_tasks=args.num_tasks, 
#                           num_lines=args.n_lines, num_circles=args.n_circles, num_squares=args.n_squares, has_empty_path=False)
#         env = envs.PWDrawEnv(False, kwargs_pcg, flat_action=args.flat_action, device=args.device)
#     elif args.env=='destroy':
#         env = envs.PWDestroyEnv(False, kwargs_pcg, flat_action=args.flat_action, device=args.device)
        
#     print(f'Action Space: {env.action_space}')
        
#     env = VecMonitor(env)
#     if args.track:
#         env = VecVideoRecorder(env, folder_video,
#                                record_video_trigger=lambda x: x%2000==0, video_length=500)
    
#     policy_kwargs = dict(
#         features_extractor_class=CustomCNN,
#         features_extractor_kwargs=dict(features_dim=256),
#     )
    
#     model = PPO('CnnPolicy', env, learning_rate=0.00006, clip_range=.1,
#                 policy_kwargs=policy_kwargs, n_steps=128, batch_size=2048, verbose=2,
#                 tensorboard_log=folder_tensorboard, seed=args.seed, device=args.device)
#     model.learn(total_timesteps=args.n_steps, callback=callbacks, progress_bar=True)
    
#     if args.track:
#         run.finish()
        

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default='cuda:0')

parser.add_argument("--n_steps", type=int, default=500_000)

parser.add_argument("--learn", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--reward", type=str, default='novelty')
parser.add_argument("--lr", type=float, default=1e-3)

parser.add_argument("--freq_learn", type=int, default=10)
parser.add_argument("--n_udpates_learn", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--coef_entropy", type=float, default=1e-2)


def main():
    args = parser.parse_args()
    print(args)

if __name__=='__main__':
    main()


