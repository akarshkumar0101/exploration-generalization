import numpy as np
import torch
from torch import nn

import sklearn

import goexplore

class RandomExplorer():
    def __init__(self):
        self.goexplore = None
        
    def get_action(self, state_sim, obs, latent, action=None):
        # return self.env.action_space.sample()
        logits = torch.log(torch.ones(4))
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), logits

    def update_policy(self, nodes, *args, **kwargs):
        pass

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(2, 10)),
            nn.Tanh(),
            layer_init(nn.Linear(10, 10)),
            nn.Tanh(),
            layer_init(nn.Linear(10, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(2, 10)),
            nn.Tanh(),
            layer_init(nn.Linear(10, 10)),
            nn.Tanh(),
            layer_init(nn.Linear(10, 4), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)



class PolicyExplorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.goexplore = None
        
        self.net = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 4),
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.losses = []


    def forward(self, x):
        return self.net(x)
        
    def get_action(self, state_sim, obs, latent, action=None):
        # p = obs[:, [0,1]]
        # d = nn.functional.one_hot(obs[:, 2], 4)
        # x = torch.cat([p, d], dim=-1).float()
        logits = self(latent)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), logits


    def compute_gae(self, rewards, values, gamma=0.99, gae_lambda=0.95):
        """
        Generalized Advantage Estimation
        rewards, values should have shape (n_trajs, len_traj)
        """
        t = torch.arange(rewards.shape[1]).expand(rewards.shape[0], -1)

        values_now, values_next = values[..., :-1], values[..., 1:]
        rewards_now, rewards_next = rewards[..., :-1], rewards[..., 1:]
        t_now, t_next = t[..., :-1], t[..., 1:]

        # Compute TD residual
        # shape: (n_trajs, len_traj-1)
        deltas_now = rewards_now + gamma * values_next - values_now
        def reverse_cumsum(x, dim=0):
            return x + x.sum(dim=dim, keepdims=True) - torch.cumsum(x, dim=dim)
        advantages = ((gamma*gae_lambda)**t_now) * deltas_now
        advantages = reverse_cumsum(advantages, dim=-1)
        returns = advantages + values_now
        return advantages, returns

    def update_policy(self, nodes, n_epochs_policy, batch_size_policy):
        # print(f'updating policy with {len(nodes)} nodes')
        # nodes_history = self.goexplore.archive.nodes
        n_trajs, len_traj = len(nodes), len(nodes[0].traj)

        obss = torch.stack([torch.stack([trans[1] for trans in node.traj]) for node in nodes]).float()
        actions = torch.stack([torch.stack([trans[2] for trans in node.traj]) for node in nodes]).float()
        values = torch.zeros(n_trajs, len_traj)
        # shape: n_trajs, len_traj, ...
        # rewards = (prods - prods.mean())/(prods.std() + 1e-6)
        rewards = torch.zeros(n_trajs, len_traj)
        # prods = compute_productivities(nodes)
        # prods = torch.tensor([prods[node] for node in nodes]).float()
        # prods = prods[:, None].expand(-1, len_traj)
        advantages, returns = compute_gae(rewards, values)
        # returns = (prods - prods.mean())/(prods.std()+1e-6)
    # nodes = self.goexplore.archive.nodes[1:] # ignore root because it has not trajectory
        # shape: n_trajs, len_traj, ...
        
        obss_f = obss.reshape(-1, obss.shape[-1])
        actions_f = actions.reshape(-1)
        values_f = values.reshape(-1)
        rewards_f = rewards.reshape(-1)
        advantages_f = advantages.reshape(-1)
        returns_f = returns.reshape(-1)

        for i_epoch in range(n_epochs_policy):
            for i_batch, idx_b in enumerate(torch.randperm(n_trajs*len_traj).split(batch_size_policy)):

                obss_b = obss_f[idx_b]
                actions_b = actions_f[idx_b]
                values_b = values_f[idx_b]
                rewards_b = rewards_f[idx_b]
                advantages_b = advantages_f[idx_b]
                returns_b = returns_f[idx_b]

                # latent_b = self.goexplore.state2latent(None, obss_b)

                # print(obss_b.shape, actions_b.shape, returns_b.shape)
                
                _, newlogprob_b, entropy_b, newvalues_b = self.get_action(None, obss_b, latent_b, actions_b)
                logratio = newlogprob_b - log_probs_b
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if True:
                    advantages_b = (advantages_b - advantages_b.mean())/(advantages_b.std() + 1e-8)
                # Policy loss
                pg_loss1 = -advantages_b * ratio
                pg_loss2 = -advantages_b * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if args.clip_vloss:
                    v_loss_unclipped = (newvalues_b - returns_b) ** 2
                    v_clipped = values_b + (newvalues_b - values_b).clamp(-args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - returns_b) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy_b.mean()
                loss = 1*pg_loss - args.ent_coef*entropy_loss + args.vf_coef*v_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                self.opt.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

    def collect_traj(self, node_start, len_traj):
        traj = [] # list of tuples (state, obs, action, log_probs, reward)
        # state_sim, obs = node_start.state_sim, node_start.obs
        state_sim, obs, reward, done, info = self.env.goto_state_sim(node_start.state_sim)
        for i_trans in range(len_traj):
            latent = self.state2latent(state_sim, obs)

            with torch.no_grad():
                action, logprob, _, value = self.get_action_and_value(next_obs)
                # values[step] = value.flatten()
            # action, log_prob, _ = self.explorer.get_action(state_sim, obs, latent)
            
            state_sim_next, obs_next, reward, done, info = self.env.step(action.cpu().numpy())
            traj.append((state_sim, obs, done, action, log_prob, reward))
            state_sim, obs = state_sim_next, obs_next
            if done:
                print('DONE!!!!!!!')
                break

        latent = self.state2latent(state_sim, obs)
        node = Node(node_start, traj, state_sim, obs, done, latent, node_start.depth+1)
        return node, done

    def collect_trajs(self, n_trajs, len_traj):
        nodes = []
        for i_traj in range(n_trajs):
            node_start = self.select_node()
            self.env.reset(node_start.state_sim)
            node_end, done = self.collect_traj(node_start, len_traj)
            nodes.append(node_end)
        return nodes

    def step(self, n_trajs, len_traj, n_epochs_policy, batch_size_policy):
        # TODO: anneal lr, 
        # TODO normalize obs
        # collect new policy/exploration data
        nodes = self.perform_multiple_trajs(n_trajs, len_traj)
        
        # update policy using this data
        self.explorer.update_policy(nodes, n_epochs_policy, batch_size_policy)

        



        # add the end nodes to the archive
        for node in nodes:
            self.archive.add_node(node)
        self.latents = torch.stack([node.latent for node in self.archive.nodes])

        self.returner.fit(self.latents)






def compute_productivities(nodes, alpha=0.01, child_aggregate='mean'):
    """
    Compute the productivity of each node in the archive.

    alpha determines how much I care about myself vs my children.
    """
    latents = torch.stack([node.latent for node in nodes])
    
    gm = sklearn.mixture.GaussianMixture(n_components=1)
    gm = gm.fit(latents.numpy())
    
    novelties = -gm.score_samples(latents.numpy())
    novelties = {node: n for node, n in zip(nodes, novelties)}
    
    productivities = {}

    if child_aggregate == 'mean':
        child_aggregate = np.mean
    elif child_aggregate == 'max':
        child_aggregate = np.max
    
    # gamma = 1.
    # alpha = .01 # how much do I care about myself vs children?
    
    def dfs_productivity(node, depth=0):
        # assert node.depth==depth
        
        productivity = novelties[node]
        if node.children:
            prods_child = []
            for child in node.children:
                # if child not in nodes:
                    # continue
                dfs_productivity(child, depth=depth+1)
                prods_child.append(productivities[child])
            # should we use mean or max here?
            # should we consider it good to have one good children or do all children have to be good?
            productivity = alpha*productivity + (1-alpha)*child_aggregate(prods_child)
        else:
            productivity = alpha*productivity + (1-alpha)*productivity # no children, so productivity is just novelty
        productivities[node] = productivity
    
    for node in nodes:
        if node not in productivities:
            dfs_productivity(node)
    
    return productivities

    

