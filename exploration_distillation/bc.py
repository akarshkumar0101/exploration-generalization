import matplotlib.pyplot as plt
import numpy as np
import torch


def train_bc_agent(agent, x_train, y_train, batch_size=32, n_steps=10, lr=1e-3, coef_entropy=0.0,
                   device=None, tqdm=None, callback_fn=None):
    """
    Behavior Cloning
    """
    agent = agent.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    pbar = range(n_steps)
    if tqdm is not None: pbar = tqdm(pbar, leave=False)
    for i_step in pbar:
        idxs_batch = torch.randperm(len(x_train))[:batch_size]
        x_batch, y_batch = x_train[idxs_batch].float().to(device), y_train[idxs_batch].long().to(device)
        dist, values = agent.get_dist_and_values(x_batch)

        loss_bc = loss_fn(dist.logits, y_batch).mean()
        loss_entropy = dist.entropy().mean()
        loss = loss_bc - coef_entropy * loss_entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        if tqdm is not None:
            pbar.set_postfix(loss_bc=loss_bc.item(), entropy=loss_entropy.item())
        if callback_fn is not None:
            callback_fn(**locals())
