import torch
import time

from einops import rearrange


class Buffer:
    # TODO: change from n_steps, n_envs shape to n_envs, n_steps shape
    def __init__(self, n_steps, n_envs, env, device=None):
        self.n_steps, self.n_envs = n_steps, n_envs
        self.env, self.device = env, device

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

        _, info = self.env.reset()
        self.obs = info["obs"]
        self.done = torch.ones(n_envs, dtype=torch.bool, device=device)

    def _construct_agent_input(self, i_step, ctx_len):
        # only for use during inference bc that's the only time buffer rolls over (for first observation)
        assert i_step >= 0 and i_step <= self.n_steps
        if i_step < ctx_len - 1:
            idx = list(range(i_step + 1 - ctx_len, i_step + 1))
            obs, act, done = self.obss[idx], self.acts[idx[:-1]], self.dones[idx]  # this is what takes time...
        elif i_step < self.n_steps:
            obs = self.obss[i_step + 1 - ctx_len : i_step + 1]  # indexing with slices is *much* faster than indexing with lists
            act = self.acts[i_step + 1 - ctx_len : i_step]
            done = self.dones[i_step + 1 - ctx_len : i_step + 1]
        else:
            obs = torch.cat([self.obss[-ctx_len + 1 :], self.obs[None]], dim=0)
            act = self.acts[-ctx_len + 1 :]
            done = torch.cat([self.dones[-ctx_len + 1 :], self.done[None]], dim=0)
        obs = rearrange(obs, "t n ... -> n t ...")
        act = rearrange(act, "t n ... -> n t ...")
        done = rearrange(done, "t n ... -> n t ...")
        # return dict(obs=torch.zeros(8, 4, 1, 84, 84, device="mps"), act=torch.zeros(8, 3, dtype=torch.int64, device="mps"), done=torch.zeros(8, 4, dtype=bool, device="mps"))
        return dict(obs=obs, act=act, done=done)

    @torch.no_grad()
    def collect(self, agent, ctx_len):
        agent.eval()
        self.dt_const, self.dt_inf, self.dt_env = 0.0, 0.0, 0.0
        for i_step in range(self.n_steps):
            self.obss[i_step] = self.obs
            self.dones[i_step] = self.done

            time1 = time.time()
            agent_input = self._construct_agent_input(i_step, ctx_len)
            time2 = time.time()

            self.dist, self.value = agent(**agent_input)
            # only get output from last token
            self.dist, self.value = torch.distributions.Categorical(logits=self.dist.logits[:, -1]), self.value[:, -1]
            time3 = time.time()
            action = self.dist.sample()

            _, reward, _, _, info = self.env.step(action.cpu().numpy())
            time4 = time.time()
            self.obs, self.done = info["obs"], info["done"]

            self.vals[i_step] = self.value
            self.acts[i_step] = action
            self.logits[i_step] = self.dist.logits
            self.rews[i_step] = torch.as_tensor(reward).to(self.device)

            self.dt_const += time2 - time1
            self.dt_inf += time3 - time2
            self.dt_env += time4 - time3
        # print(f"collect: dt_const={self.dt_const:.3f}, dt_inf={self.dt_inf:.3f}, dt_env={self.dt_env:.3f}")
        _, self.value = agent(**self._construct_agent_input(i_step + 1, ctx_len))  # calculate one more value
        self.value = self.value[:, -1]

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

    def generate_batch_naive1(self, batch_size, ctx_len):
        # naive because for loop over batch
        i_env = torch.randint(0, self.n_envs, (batch_size,), device="cpu").tolist()
        i_step = torch.randint(0, self.n_steps + 1 - ctx_len, (batch_size,), device="cpu").tolist()

        batch = {}
        batch["obs"] = torch.stack([self.obss[i : i + ctx_len, j] for i, j in zip(i_step, i_env)])
        batch["done"] = torch.stack([self.dones[i : i + ctx_len, j] for i, j in zip(i_step, i_env)])
        batch["logits"] = torch.stack([self.logits[i : i + ctx_len, j] for i, j in zip(i_step, i_env)])
        batch["dist"] = torch.distributions.Categorical(logits=batch["logits"])
        batch["logprob"] = torch.stack([self.logprobs[i : i + ctx_len, j] for i, j in zip(i_step, i_env)])
        batch["act"] = torch.stack([self.acts[i : i + ctx_len, j] for i, j in zip(i_step, i_env)])
        batch["val"] = torch.stack([self.vals[i : i + ctx_len, j] for i, j in zip(i_step, i_env)])
        batch["rew"] = torch.stack([self.rews[i : i + ctx_len, j] for i, j in zip(i_step, i_env)])
        batch["adv"] = torch.stack([self.advs[i : i + ctx_len, j] for i, j in zip(i_step, i_env)])
        batch["ret"] = torch.stack([self.rets[i : i + ctx_len, j] for i, j in zip(i_step, i_env)])
        return batch

    def generate_batch_naive2(self, batch_size, ctx_len):
        # this flattens the 2d space into a 1d space and indexes into it and then unflattens it
        # naive because indexing with arrays instead of slices
        i_flat = self.generate_batch_i_flat(batch_size, ctx_len)
        batch = {}
        batch["obs"] = self.index_into_flat(self.obss, i_flat, batch_size)
        batch["done"] = self.index_into_flat(self.dones, i_flat, batch_size)
        batch["logits"] = self.index_into_flat(self.logits, i_flat, batch_size)
        batch["dist"] = torch.distributions.Categorical(logits=batch["logits"])
        batch["logprob"] = self.index_into_flat(self.logprobs, i_flat, batch_size)
        batch["act"] = self.index_into_flat(self.acts, i_flat, batch_size)
        batch["val"] = self.index_into_flat(self.vals, i_flat, batch_size)
        batch["rew"] = self.index_into_flat(self.rews, i_flat, batch_size)
        batch["adv"] = self.index_into_flat(self.advs, i_flat, batch_size)
        batch["ret"] = self.index_into_flat(self.rets, i_flat, batch_size)
        return batch

    def generate_batch_i_flat(self, batch_size, ctx_len):
        i_env = torch.randint(0, self.n_envs, (batch_size,), device=self.device)
        i_step = torch.randint(0, self.n_steps + 1 - ctx_len, (batch_size,), device=self.device)
        i_env = i_env[:, None] + torch.zeros(ctx_len, dtype=i_env.dtype, device=self.device)
        i_step = i_step[:, None] + torch.arange(ctx_len, device=self.device)
        i_env, i_step = i_env.flatten(), i_step.flatten()
        # i_flat = i_env * self.n_steps + i_step
        i_flat = i_step * self.n_envs + i_env
        return i_flat

    def index_into_flat(self, a, i_flat, batch_size):
        a = rearrange(a, "t b ... -> (t b) ...")
        a = a[i_flat]
        a = rearrange(a, "(b t) ... -> b t ...", b=batch_size)
        return a

    def generate_batch(self, batch_size, ctx_len):
        # optimal because it maximizes slicing
        assert batch_size % self.n_envs == 0

        i_step = torch.randint(0, self.n_steps + 1 - ctx_len, (batch_size // self.n_envs,), device="cpu").tolist()

        batch = {}
        batch["obs"] = self.index_into_temp(self.obss, i_step, ctx_len)
        batch["done"] = self.index_into_temp(self.dones, i_step, ctx_len)
        batch["logits"] = self.index_into_temp(self.logits, i_step, ctx_len)
        batch["dist"] = torch.distributions.Categorical(logits=batch["logits"])
        batch["logprob"] = self.index_into_temp(self.logprobs, i_step, ctx_len)
        batch["act"] = self.index_into_temp(self.acts, i_step, ctx_len)
        batch["val"] = self.index_into_temp(self.vals, i_step, ctx_len)
        batch["rew"] = self.index_into_temp(self.rews, i_step, ctx_len)
        batch["adv"] = self.index_into_temp(self.advs, i_step, ctx_len)
        batch["ret"] = self.index_into_temp(self.rets, i_step, ctx_len)
        return batch

    def index_into_temp(self, a, i_step, ctx_len):
        a = torch.cat([a[i : i + ctx_len, :] for i in i_step], dim=1)
        a = rearrange(a, "t b ... -> b t ...")
        return a


class MultiBuffer:
    def __init__(self, buffers=None):
        self.buffers = buffers if buffers is not None else []

    @property
    def envs(self):
        return [buffer.env for buffer in self.buffers]

    def collect(self, agents, ctx_len):
        if isinstance(agents, torch.nn.Module):
            agents = [agents for _ in self.buffers]
        for buffer, agent in zip(self.buffers, agents):
            buffer.collect(agent, ctx_len)

    def calc_gae(self, gamma, gae_lambda):
        for buffer in self.buffers:
            buffer.calc_gae(gamma, gae_lambda)

    def generate_batch(self, batch_size, ctx_len):
        data = [buffer.generate_batch(batch_size // len(self.buffers), ctx_len) for buffer in self.buffers]
        # list of dicts to dict of lists
        data = {key: [di[key] for di in data] for key in data[0]}  # list of dicts to dict of lists
        for key in data:
            if isinstance(data[key][0], torch.Tensor):
                data[key] = torch.cat(data[key], dim=0)
        data["dist"] = torch.distributions.Categorical(logits=data["logits"])
        return data


if __name__ == "__main__":
    pass
