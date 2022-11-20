import matplotlib.pyplot as plt
import torch

import goexplore_discrete


def train_bc_elite(ge, net, opt, n_nodes_select, n_steps, batch_size=1000, coef_entropy=1e-1, device=None):
    """
    Train using Behavior Cloning
    """
    trajs = [node.get_full_trajectory() for node in ge.select_nodes(n_nodes_select)]
    obs = torch.stack([trans[1] for traj in trajs for trans in traj])
    action = torch.stack([trans[2] for traj in trajs for trans in traj])

    net = net.to(device)
    losses = []
    entropies = []
    logits_std = []
    for i_batch in range(n_steps):
        idx = torch.randperm(len(obs))[:batch_size]
        b_obs, b_action = obs[idx].to(device), action[idx].to(device)
        logits, values = net.get_logits_values(b_obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(b_action)
        entropy = dist.entropy()
        
        # b_r.abs tells us how much to care about specific examples
        loss_data = (-log_prob).mean()
        loss_entropy = -entropy.mean()
        
        loss = loss_data + coef_entropy*loss_entropy

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        entropies.append(entropy.mean().item())
        logits_std.append(logits.std().item())
        
    return torch.tensor(losses), torch.tensor(entropies), torch.tensor(logits_std)

def train_contrastive_bc(ge, net, opt, n_steps, batch_size=1000, coef_entropy=1e-1, device=None):
    obs = torch.stack([trans[1] for node in ge for trans in node.traj])
    action = torch.stack([trans[2] for node in ge for trans in node.traj])
    mask_done = torch.stack([trans[4] for node in ge for trans in node.traj])
    reward = goexplore_discrete.calc_reward_novelty(ge)
    reward = torch.stack([reward[i] for i, node in enumerate(ge) for trans in node.traj])
    reward = (reward-reward[~mask_done].mean())/(reward[~mask_done].std()+1e-9)

    return train_fancy_contrastive(net, opt, obs, action, reward, n_steps, batch_size, coef_entropy, device)

def train_fancy_contrastive(net, opt, obs, action, reward, n_steps, batch_size, coef_entropy, device=None):
    """
    Train using contrastive Behavior Cloning

    net is the network
    opt is the optimizer
    obs is the observations of shape (N, ...)
    action is the actions of shape (N, ...)
    reward is the rewards of shape (N, ...)

    n_steps is the number of gradient steps
    batch_size is the batch size
    coef_entropy is the coefficient for the entropy term
    device is the device to use
    data is a dictionary to store the data
    """
    # list of tuples (snapshot, obs, action, reward, done)
    net = net.to(device)

    losses = []
    entropies = []
    logits_std = []
    for i_batch in range(n_steps):
        idx = torch.randperm(len(obs))[:batch_size]
        b_obs, b_action, b_r = obs[idx].to(device), action[idx].to(device), reward[idx].to(device)
        # if norm_batch:
            # b_prod = (b_prod-b_prod.mean())/(b_prod.std()+1e-9)
            
        logits, values = net.get_logits_values(b_obs)
        # logits_aug = b_r[:, None]*logits
        # logits_aug = (1./b_r[:, None])*logits

        # sign tells us to increase or decrease the probability of the action
        logits_aug = torch.sign(b_r[:, None])*logits
        dist = torch.distributions.Categorical(logits=logits_aug)
        log_prob = dist.log_prob(b_action)
        entropy = dist.entropy()
        
        # b_r.abs tells us how much to care about specific examples
        loss_data = (-log_prob*b_r.abs()).mean()
        loss_entropy = -entropy.mean()
        
        loss = loss_data + coef_entropy*loss_entropy

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        entropies.append(entropy.mean().item())
        logits_std.append(logits.std().item())
        
    return torch.tensor(losses), torch.tensor(entropies), torch.tensor(logits_std)

    # if viz:
    #     plt.figure(figsize=(15, 5))
    #     plt.subplot(131); plt.plot(losses); plt.title('loss vs time')
    #     plt.subplot(132); plt.plot(entropies); plt.title('entropy vs time')
    #     plt.subplot(133); plt.plot(logits_std); plt.title('logits std vs time')
    #     plt.show()
        
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



