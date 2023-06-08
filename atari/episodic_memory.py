import torch
import collections

import gymnasium as gym
from torch import nn
import numpy as np

import gym.wrappers.normalize


class EpisodicBonus(gym.Wrapper):
    def __init__(self, env, encode_fn):
        super().__init__(env)
        self.encode_fn = encode_fn
        self._cdist_normalizer = gym.wrappers.normalize.RunningMeanStd(shape=(10,))

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.memories = [collections.deque(maxlen=30000) for _ in range(self.n_envs)]
        return obs, info

    @torch.no_grad()
    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)

        latents = self.encode_fn(info["obs"])  # n, d
        for i, (latent, memory) in enumerate(zip(latents, self.memories)):
            memory = torch.stack(list(memory))  # m, d
            self.memories[i].append(latent)
            if len(memory) < 10:
                rew[i] = 0.0
                continue

            dsquared = (latent - memory).norm(dim=-1).pow(2)  # m
            nn_distances_sq = dsquared.topk(k=10, largest=False).values  # k

            self._cdist_normalizer.update(nn_distances_sq.cpu().numpy())

            # Normalize distances with running mean dₘ².
            distance_rate = nn_distances_sq / self._cdist_normalizer.mean

            # The distance rate becomes 0 if already small: r <- max(r-ξ, 0).
            distance_rate = (distance_rate - 0.008).clamp(0, None)

            # Compute the Kernel value K(xₖ, x) = ε/(rate + ε).
            kernel_output = 0.0001 / (distance_rate + 0.0001)

            # Compute the similarity for the embedding x:
            # s = √(Σ_{xₖ ∈ Nₖ} K(xₖ, x)) + c
            similarity = torch.sqrt(torch.sum(kernel_output)) + 0.001

            if torch.isnan(similarity):
                rew[i] = 0.0
                continue

            # Compute the intrinsic reward:
            # r = 1 / s.
            if similarity > 8:
                rew[i] = 0.0
                continue
            rew[i] = (1 / similarity).cpu().item()

        return obs, rew, term, trunc, info


# class EpisodicBonusModule:
#     """Episodic memory for calculate intrinsic bonus, used in NGU and Agent57."""

#     def __init__(
#         self,
#         embedding_network: torch.nn.Module,
#         device: torch.device,
#         capacity: int,
#         num_neighbors: int,
#         kernel_epsilon: float = 0.0001,
#         cluster_distance: float = 0.008,
#         max_similarity: float = 8.0,
#         c_constant: float = 0.001,
#     ) -> None:
#         self._embedding_network = embedding_network.to(device=device)
#         self._device = device

#         self._memory = collections.deque(maxlen=capacity)

#         # Compute the running mean dₘ².
#         self._cdist_normalizer = normalizer.Normalizer(eps=0.0001, clip_range=(-10, 10), device=self._device)

#         self._num_neighbors = num_neighbors
#         self._kernel_epsilon = kernel_epsilon
#         self._cluster_distance = cluster_distance
#         self._max_similarity = max_similarity
#         self._c_constant = c_constant

#     @torch.no_grad()
#     def compute_bonus(self, s_t: torch.Tensor) -> float:
#         """Compute episodic intrinsic bonus for given state."""
#         base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)

#         embedding = self._embedding_network(s_t).squeeze(0)

#         memory = list(self._memory)

#         # Insert single embedding into memory.
#         self._memory.append(embedding)

#         if len(memory) < self._num_neighbors:
#             return 0.0

#         memory = torch.stack(memory, dim=0)
#         knn_query_result = knn_query(embedding, memory, self._num_neighbors)
#         # neighbor_distances from knn_query is the squared Euclidean distances.
#         nn_distances_sq = knn_query_result.neighbor_distances

#         # Update the running mean dₘ².
#         self._cdist_normalizer.update(nn_distances_sq[..., None])

#         # Normalize distances with running mean dₘ².
#         distance_rate = nn_distances_sq / self._cdist_normalizer.mean

#         # The distance rate becomes 0 if already small: r <- max(r-ξ, 0).
#         distance_rate = torch.min((distance_rate - self._cluster_distance), torch.tensor(0.0))

#         # Compute the Kernel value K(xₖ, x) = ε/(rate + ε).
#         kernel_output = self._kernel_epsilon / (distance_rate + self._kernel_epsilon)

#         # Compute the similarity for the embedding x:
#         # s = √(Σ_{xₖ ∈ Nₖ} K(xₖ, x)) + c
#         similarity = torch.sqrt(torch.sum(kernel_output)) + self._c_constant

#         if torch.isnan(similarity):
#             return 0.0

#         # Compute the intrinsic reward:
#         # r = 1 / s.
#         if similarity > self._max_similarity:
#             return 0.0

#         return (1 / similarity).cpu().item()

#     def reset(self) -> None:
#         """Resets episodic memory"""
#         self._memory.clear()

#     def update_embedding_network(self, state_dict: Dict) -> None:
#         """Update embedding network."""
#         self._embedding_network.load_state_dict(state_dict)