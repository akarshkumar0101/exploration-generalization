import timers
import torch


class Buffer:
    def __init__(self, n_envs, n_steps, env, device="cpu"):
        self.n_envs, self.n_steps = n_envs, n_steps
        self.env, self.device = env, device

        self.obss = torch.zeros((n_envs, n_steps) + env.single_observation_space.shape, dtype=torch.uint8, device=device)
        self.dones = torch.zeros((n_envs, n_steps), dtype=torch.bool, device=device)

        self.logits = torch.zeros((n_envs, n_steps, env.single_action_space.n), device=device)
        self.acts = torch.zeros((n_envs, n_steps) + env.single_action_space.shape, dtype=torch.long, device=device)
        self.vals = torch.zeros((n_envs, n_steps), device=device)
        self.rews = torch.zeros((n_envs, n_steps), device=device)

        self.advs = torch.zeros((n_envs, n_steps), device=device)
        self.rets = torch.zeros((n_envs, n_steps), device=device)

        _, info = self.env.reset()
        self.obs, self.done = info["obs"], info["done"]

    def _construct_agent_input(self, i_step, ctx_len):
        # only for use during inference bc that's the only time buffer rolls over (for first observation)
        assert i_step >= 0 and i_step <= self.n_steps
        # we want i+1-c: i+1, but this doesn't work if i+1-c < 0
        if i_step + 1 < ctx_len:  # use data from end of buffer (previous episode)
            done = torch.cat([self.dones[:, i_step + 1 - ctx_len :], self.dones[:, : i_step + 1]], dim=1)
            obs = torch.cat([self.obss[:, i_step + 1 - ctx_len :], self.obss[:, : i_step + 1]], dim=1)
            act = torch.cat([self.acts[:, i_step + 1 - ctx_len :], self.acts[:, :i_step]], dim=1)
            rew = torch.cat([self.rews[:, i_step + 1 - ctx_len :], self.rews[:, :i_step]], dim=1)
        elif i_step < self.n_steps:  # fastest case... indexing with slices is *much* faster than indexing with lists
            done = self.dones[:, i_step + 1 - ctx_len : i_step + 1]
            obs = self.obss[:, i_step + 1 - ctx_len : i_step + 1]
            act = self.acts[:, i_step + 1 - ctx_len : i_step]
            rew = self.rews[:, i_step + 1 - ctx_len : i_step]
        elif i_step == self.n_steps:
            done = torch.cat([self.dones[:, i_step + 1 - ctx_len : i_step + 1], self.done[:, None]], dim=1)
            obs = torch.cat([self.obss[:, i_step + 1 - ctx_len : i_step + 1], self.obs[:, None]], dim=1)
            act = self.acts[:, i_step + 1 - ctx_len : i_step]
            rew = self.rews[:, i_step + 1 - ctx_len : i_step]
        return dict(done=done, obs=obs, act=act, rew=rew)

    @torch.no_grad()
    def collect(self, agent, ctx_len, timer):
        agent.eval()
        for i_step in range(self.n_steps):
            self.obss[:, i_step] = self.obs
            self.dones[:, i_step] = self.done

            with timer.add_time("construct_agent_input"):
                agent_input = self._construct_agent_input(i_step, ctx_len)
            with timer.add_time("agent_inference"):
                logits, value = agent(**agent_input)  # b t ...
            # only get output from last token
            logits, value = logits[:, -1, :], value[:, -1]
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()

            with timer.add_time("env_step"):
                _, rew, _, _, info = self.env.step(act)
            self.obs, self.done = info["obs"], info["done"]

            self.logits[:, i_step] = logits
            self.acts[:, i_step] = act
            self.vals[:, i_step] = value
            self.rews[:, i_step] = info["rew"]

        i_step += 1
        with timer.add_time("construct_agent_input"):
            agent_input = self._construct_agent_input(i_step, ctx_len)
        with timer.add_time("agent_inference"):
            logits, value = agent(**agent_input)  # b t ...
        self.value = value[:, -1]

        print("Collection time breakdown:")
        for key, t in timer.key2time.items():
            print(f"{key:30s}: {t:.3f}")

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

    """
    Batch Generation Strategy 1:
        - generate i_env, i_step of size batch_size
        - loop over these and stack tensors
    
    Batch Generation Strategy 2:
        - flatten buffer tensors and then index into them directly with i_batch_flat

    Batch Generation Strategy 3: (most efficient since it uses slicing rather than indexing)
        - generate i_step of size batch_size//n_envs
        - loop over these and cat tensors like cat([obs[:, i] for i in i_step])

    """

    def generate_batch(self, batch_size, ctx_len):
        assert batch_size % self.n_envs == 0

        i_step = torch.randint(0, self.n_steps + 1 - ctx_len, (batch_size // self.n_envs,), device="cpu").tolist()
        batch = {}
        batch["obs"] = self.generate_batch_tensor(self.obss, i_step, ctx_len)
        batch["done"] = self.generate_batch_tensor(self.dones, i_step, ctx_len)
        batch["logits"] = self.generate_batch_tensor(self.logits, i_step, ctx_len)
        batch["act"] = self.generate_batch_tensor(self.acts, i_step, ctx_len)
        batch["val"] = self.generate_batch_tensor(self.vals, i_step, ctx_len)
        batch["rew"] = self.generate_batch_tensor(self.rews, i_step, ctx_len)
        batch["adv"] = self.generate_batch_tensor(self.advs, i_step, ctx_len)
        batch["ret"] = self.generate_batch_tensor(self.rets, i_step, ctx_len)

        batch["dist"] = torch.distributions.Categorical(logits=batch["logits"])
        batch["logprob"] = batch["dist"].log_prob(batch["act"])
        return batch

    def generate_batch_tensor(self, data, i_step, ctx_len):
        return torch.cat([data[:, i : i + ctx_len] for i in i_step], dim=0)  # b t ...


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
