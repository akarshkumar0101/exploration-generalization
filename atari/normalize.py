# Adapted from gymnasium.wrappers.normalize
"""Set of wrappers for normalizing actions and observations."""

import gymnasium as gym
import torch


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self):
        """Tracks the mean, variance and count of values."""
        self.mean, self.var, self.count = None, None, 0

    def normalize(self, x, eps=1e-4):
        return (x - self.mean) / torch.sqrt(self.var + eps)

    def unnormalize(self, x, eps=1e-4):
        return x * torch.sqrt(self.var + eps) + self.mean

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        x = x.float()
        if self.mean is None:
            self.mean = torch.zeros_like(x.mean(dim=0))
            self.var = torch.ones_like(x.var(dim=0))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, correction=0)  # correction to match numpy implementation
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, eps: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, eps=eps)
        gym.Wrapper.__init__(self, env)

        self.rms_obs = RunningMeanStd()
        self.eps = eps

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rew, term, trunc, info = self.env.step(action)
        self.rms_obs.update(info["obs"])
        """Normalises the observation using the running mean and variance of the observations."""
        info["obs_norm"] = self.rms_obs.normalize(info["obs"], eps=self.eps)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)
        self.rms_obs.update(info["obs"])
        """Normalises the observation using the running mean and variance of the observations."""
        info["obs_norm"] = self.rms_obs.normalize(info["obs"], eps=self.eps)
        return obs, info


class NormalizeReward(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(self, env, key_rew="ext", gamma=0.99, eps=1e-8):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, eps=eps)
        gym.Wrapper.__init__(self, env)

        self.rms_returns = RunningMeanStd()
        self.returns = None
        self.key_rew = key_rew
        self.gamma = gamma
        self.eps = eps

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rew, term, trunc, info = self.env.step(action)

        r, d = info[f"rew_{self.key_rew}"], info["done"]
        if self.returns is None:
            self.returns = torch.zeros_like(r)
        # self.returns = self.returns * self.gamma * (1.0 - d.float()) + r
        self.returns = self.returns * self.gamma + r
        self.rms_returns.update(self.returns)
        """Normalizes the rewards with the running mean rewards and their variance."""
        r = r / (self.rms_returns.var + self.eps).sqrt()
        info[f"rew_{self.key_rew}_norm"] = r
        return obs, rew, term, trunc, info




class TorchRunningMeanStd:
    """For RND networks"""

    def __init__(self, shape=(), device='cpu'):
        self.device = device
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = 0

        self.deltas = []
        self.min_size = 10

    @torch.no_grad()
    def update(self, x):
        x = x.to(self.device)
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)

        # update count and moments
        n = x.shape[0]
        self.count += n
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count
        m_a = self.var * (self.count - n)
        m_b = batch_var * n
        M2 = m_a + m_b + torch.square(delta) * self.count * n / self.count
        self.var = M2 / self.count

    @torch.no_grad()
    def update_single(self, x):
        self.deltas.append(x)

        if len(self.deltas) >= self.min_size:
            batched_x = torch.concat(self.deltas, dim=0)
            self.update(batched_x)

            del self.deltas[:]

    @torch.no_grad()
    def normalize(self, x):
        return (x.to(self.device) - self.mean) / torch.sqrt(self.var + 1e-8)

