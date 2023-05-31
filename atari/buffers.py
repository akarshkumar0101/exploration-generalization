import torch
import time

from einops import rearrange

# class Buffer:
#     def __init__(self, env, agent, n_envs, n_steps, device=None):
#         self.env, self.agent, self.n_envs, self.n_steps, self.device = env, agent, n_envs, n_steps, device

#         self.obs = torch.zeros((n_steps, n_envs) + env.single_observation_space.shape, dtype=torch.uint8, device=self.device)
#         self.actions = torch.zeros((n_steps, n_envs) + env.single_action_space.shape, dtype=torch.long, device=self.device)
#         self.logprobs = torch.zeros((n_steps, n_envs), device=self.device)
#         self.rewards = torch.zeros((n_steps, n_envs), device=self.device)
#         self.dones = torch.zeros((n_steps, n_envs), dtype=torch.bool, device=self.device)
#         self.values = torch.zeros((n_steps, n_envs), device=self.device)

#         # self.adv = torch.zeros((n_steps, n_envs), device=self.device)

#         _, info = env.reset()
#         self.next_obs = info["obs"]
#         self.next_done = torch.zeros(n_envs, dtype=torch.bool, device=self.device)

#     def collect(self, agent=None):
#         if agent is None:
#             agent = self.agent

#         agent.eval()
#         for i_step in range(self.n_steps):
#             self.obs[i_step] = self.next_obs
#             self.dones[i_step] = self.next_done

#             with torch.no_grad():
#                 action, logprob, _, value = agent.get_action_and_value(self.next_obs)
#             _, reward, _, _, info = self.env.step(action.cpu().numpy())
#             self.next_obs, self.next_done = info["obs"], info["done"]

#             self.values[i_step] = value.flatten()
#             self.actions[i_step] = action
#             self.logprobs[i_step] = logprob
#             self.rewards[i_step] = torch.as_tensor(reward).to(self.device)


# class MultiGameBuffer:
#     def __init__(self, args, env_id2agents):
#         self.args, self.env_id2agents = args, env_id2agents

#         n_env_ids = len(env_id2agents)

#         # simple solution
#         self.env_id2n_envs = {env_id: args.n_envs // n_env_ids for env_id in env_id2agents}

#         self.env_id2buffer = {}
#         for env_id in env_id2agents:
#             pass

#         # self.buffers

#     def collect(self):
#         pass


class Buffer:
    # TODO: change from n_steps, n_envs shape to n_envs, n_steps shape
    def __init__(self, n_steps, n_envs, ctx_len, env, device=None):
        self.n_steps, self.n_envs, self.ctx_len = n_steps, n_envs, ctx_len
        self.env = env
        self.device = device

        self.obss = torch.zeros((n_steps, n_envs) + env.single_observation_space.shape, dtype=torch.uint8, device=device)
        self.dones = torch.zeros((n_steps, n_envs), dtype=torch.bool, device=device)

        self.logits = torch.zeros((n_steps, n_envs, env.single_action_space.n), device=device)
        self.dists = None
        self.logprobs = None
        self.acts = torch.zeros((n_steps, n_envs) + env.single_action_space.shape, dtype=torch.long, device=device)
        self.vals = torch.zeros((n_steps, n_envs), device=device)
        self.rews = torch.zeros((n_steps, n_envs), device=device)
        self.advs = torch.zeros((n_steps, n_envs), device=device)
        self.rets = torch.zeros((n_steps, n_envs), device=device)

        _, info = env.reset()
        self.obs = info["obs"]
        self.done = torch.ones(n_envs, dtype=torch.bool, device=device)

    def _construct_agent_input(self, i_step):
        # only for use during inference bc that's the only time buffer rolls over (for first observation)
        assert i_step <= self.n_steps
        if i_step < self.n_steps:
            if i_step >= self.ctx_len - 1:
                idx = list(range(i_step - self.ctx_len + 1, i_step + 1))
            else:
                idx = list(range(-self.ctx_len + i_step + 1, 0)) + list(range(0, i_step + 1))
            obs, act, done = self.obss[idx], self.acts[idx[:-1]], self.dones[idx]
        else:
            obs = torch.cat([self.obss[-self.ctx_len + 1 :], self.obs[None]], dim=0)
            act = self.acts[-self.ctx_len + 1 :]
            done = torch.cat([self.dones[-self.ctx_len + 1 :], self.done[None]], dim=0)
        obs = rearrange(obs, "t n ... -> n t ...")
        act = rearrange(act, "t n ... -> n t ...")
        done = rearrange(done, "t n ... -> n t ...")
        return dict(obs=obs, act=act, done=done)

    @torch.no_grad()
    def collect(self, agent):
        agent.eval()
        dt_const, dt_inf, dt_env = 0.0, 0.0, 0.0
        for i_step in range(self.n_steps):
            self.obss[i_step] = self.obs
            self.dones[i_step] = self.done

            time1 = time.time()
            agent_input = self._construct_agent_input(i_step)
            time2 = time.time()

            self.dist, self.value = agent.act(**agent_input)
            time3 = time.time()
            action = self.dist.sample()

            _, reward, _, _, info = self.env.step(action.cpu().numpy())
            time4 = time.time()
            self.obs, self.done = info["obs"], info["done"]

            self.vals[i_step] = self.value
            self.acts[i_step] = action
            self.logits[i_step] = self.dist.logits
            self.rews[i_step] = torch.as_tensor(reward).to(self.device)

            dt_const += time2 - time1
            dt_inf += time3 - time2
            dt_env += time4 - time3
        print(f"collect: dt_const={dt_const:.3f}, dt_inf={dt_inf:.3f}, dt_env={dt_env:.3f}")
        _, self.value = agent.act(**self._construct_agent_input(i_step + 1))  # calculate one more value

        self.dists = torch.distributions.Categorical(logits=self.logits)
        self.logprobs = self.dists.log_prob(self.acts)

    @torch.no_grad()
    def calc_gae(self, gamma, gae_lambda):
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = ~self.done
                nextvalues = self.value
            else:
                nextnonterminal = ~self.dones[t + 1]
                nextvalues = self.vals[t + 1]
            delta = self.rews[t] + gamma * nextvalues * nextnonterminal - self.vals[t]
            self.advs[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        self.rets = self.advs + self.vals

    def generate_batch(self, batch_size):
        i_env = torch.randint(0, self.n_envs, (batch_size,), device=self.device)
        i_step = torch.randint(0, self.n_steps + 1 - self.ctx_len, (batch_size,), device=self.device)

        batch = {}
        batch["obs"] = torch.stack([self.obss[i : i + self.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
        batch["done"] = torch.stack([self.dones[i : i + self.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
        batch["logits"] = torch.stack([self.logits[i : i + self.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
        batch["dist"] = torch.distributions.Categorical(logits=batch["logits"])
        batch["logprobs"] = torch.stack([self.logprobs[i : i + self.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
        batch["act"] = torch.stack([self.acts[i : i + self.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
        batch["vals"] = torch.stack([self.vals[i : i + self.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
        batch["rews"] = torch.stack([self.rews[i : i + self.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
        batch["advs"] = torch.stack([self.advs[i : i + self.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
        batch["rets"] = torch.stack([self.rets[i : i + self.ctx_len, j] for i, j in zip(i_step.tolist(), i_env.tolist())])
        return batch


class MultiBuffer:
    def __init__(self, n_steps, n_envs, ctx_len, envs, device=None):
        # simple solution
        self.envs = envs
        self.n_envs = [n_envs // len(envs) for _ in envs]
        self.buffers = [Buffer(n_steps, self.n_envs[i], ctx_len, envs[i], device) for i in range(len(envs))]

    def collect(self, agents):
        for buffer, agent in zip(self.buffers, agents):
            buffer.collect(agent)

    def calc_gae(self, gamma, gae_lambda):
        for buffer in self.buffers:
            buffer.calc_gae(gamma, gae_lambda)

    def generate_batch(self, batch_size):
        data = [buffer.generate_batch(batch_size // len(self.buffers)) for buffer in self.buffers]
        # list of dicts to dict of lists
        data = {key: [di[key] for di in data] for key in data[0]}  # list of dicts to dict of lists
        for key in data:
            if isinstance(data[key][0], torch.Tensor):
                data[key] = torch.cat(data[key], dim=0)
        data["dist"] = torch.distributions.Categorical(logits=data["logits"])
        return data


if __name__ == "__main__":
    pass
