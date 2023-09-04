import timers
import torch


class Buffer:
    def __init__(self, env, agent, n_steps, device=None):
        self.env, self.agent = env, agent
        self.n_envs, self.n_steps = env.num_envs, n_steps
        self.device = device
        self.first_collect = True

        self.data = {}
        self.data["obs"] = torch.zeros((self.n_envs, self.n_steps) + env.single_observation_space.shape, dtype=torch.uint8, device=device)
        self.data["done"] = torch.zeros((self.n_envs, self.n_steps), dtype=torch.bool, device=device)

        self.data["logits"] = torch.zeros((self.n_envs, self.n_steps, env.single_action_space.n), device=device)
        self.data["act"] = torch.zeros((self.n_envs, self.n_steps) + env.single_action_space.shape, dtype=torch.long, device=device)
        self.data["val"] = torch.zeros((self.n_envs, self.n_steps), device=device)
        self.data["rew"] = torch.zeros((self.n_envs, self.n_steps), device=device)

        self.data["adv"] = torch.zeros((self.n_envs, self.n_steps), device=device)
        self.data["ret"] = torch.zeros((self.n_envs, self.n_steps), device=device)

    def _construct_agent_input(self, i_step, ctx_len):
        # only for use during inference bc that's the only time buffer rolls over (for first observation)
        assert i_step >= 0 and i_step <= self.n_steps
        # we want i+1-c: i+1, but this doesn't work if i+1-c < 0
        if i_step + 1 < ctx_len:  # use data from end of buffer (previous episode)
            done = torch.cat([self.data["done"][:, i_step + 1 - ctx_len :], self.data["done"][:, : i_step + 1]], dim=1)
            obs = torch.cat([self.data["obs"][:, i_step + 1 - ctx_len :], self.data["obs"][:, : i_step + 1]], dim=1)
            act = torch.cat([self.data["act"][:, i_step + 1 - ctx_len :], self.data["act"][:, :i_step]], dim=1)
            rew = torch.cat([self.data["rew"][:, i_step + 1 - ctx_len :], self.data["rew"][:, :i_step]], dim=1)
        elif i_step < self.n_steps:  # fastest case... indexing with slices is *much* faster than indexing with lists
            done = self.data["done"][:, i_step + 1 - ctx_len : i_step + 1]
            obs = self.data["obs"][:, i_step + 1 - ctx_len : i_step + 1]
            act = self.data["act"][:, i_step + 1 - ctx_len : i_step]
            rew = self.data["rew"][:, i_step + 1 - ctx_len : i_step]
        elif i_step == self.n_steps:
            done = torch.cat([self.data["done"][:, i_step + 1 - ctx_len : i_step + 1], self.done[:, None]], dim=1)
            obs = torch.cat([self.data["obs"][:, i_step + 1 - ctx_len : i_step + 1], self.obs[:, None]], dim=1)
            act = self.data["act"][:, i_step + 1 - ctx_len : i_step]
            rew = self.data["rew"][:, i_step + 1 - ctx_len : i_step]
        return dict(done=done, obs=obs, act=act, rew=rew)

    @torch.no_grad()
    def collect(self, pbar=None):
        timer = timers.Timer()
        if self.first_collect:
            self.first_collect = False
            self.obs, info = self.env.reset()
            self.done = torch.zeros((self.n_envs,), dtype=torch.bool, device=self.device)

        self.agent.eval()
        for i_step in range(self.n_steps):
            if pbar is not None:
                pbar.update(1)

            self.data["obs"][:, i_step] = self.obs
            self.data["done"][:, i_step] = self.done

            with timer.add_time("construct_agent_input"):
                agent_input = self._construct_agent_input(i_step, self.agent.ctx_len)
            with timer.add_time("agent_inference"):
                logits, value = self.agent(**agent_input)  # b t ...
            # only get output from last token
            logits, value = logits[:, -1, :], value[:, -1]
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()

            with timer.add_time("env_step"):
                self.obs, rew, term, trunc, info = self.env.step(act)
            self.done = term | trunc
            self.data["logits"][:, i_step, :] = dist.logits
            self.data["act"][:, i_step] = act
            self.data["val"][:, i_step] = value
            self.data["rew"][:, i_step] = rew

        i_step += 1
        with timer.add_time("construct_agent_input"):
            agent_input = self._construct_agent_input(i_step, self.agent.ctx_len)
        with timer.add_time("agent_inference"):
            logits, value = self.agent(**agent_input)  # b t ...
        self.value = value[:, -1]

        # print("Collection time breakdown:")
        # for key, t in timer.key2time.items():
        # print(f"{key:30s}: {t:.3f}")

    @torch.no_grad()
    def calc_gae(self, gamma, gae_lambda, episodic=True):
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                # nextnonterminal = ~self.done
                nextnonterminal = 1.0 - self.done.float() if episodic else 1.0
                nextvalues = self.value
            else:
                # nextnonterminal = ~self.dones[:, t + 1]
                nextnonterminal = 1.0 - self.data["done"][:, t + 1].float() if episodic else 1.0
                nextvalues = self.data["val"][:, t + 1]
            delta = self.data["rew"][:, t] + gamma * nextvalues * nextnonterminal - self.data["val"][:, t]
            self.data["adv"][:, t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        self.data["ret"] = self.data["adv"] + self.data["val"]

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
        batch = {k: self.generate_batch_tensor(v, i_step, ctx_len) for k, v in self.data.items()}
        return batch

    def generate_batch_tensor(self, data, i_step, ctx_len):
        return torch.cat([data[:, i : i + ctx_len] for i in i_step], dim=0)  # b t ...


if __name__ == "__main__":
    import timers
    import gymnasium as gym
    from my_envs import *
    from my_agents import *

    agent = StackedCNNAgent(18, 4)

    timer = timers.Timer()
    env = MyEnvpool("MontezumaRevenge-v5", num_envs=8, stack_num=1)
    env = gym.wrappers.NormalizeReward(env)
    env = RecordEpisodeStatistics(env)
    env = ToTensor(env)

    buffer = Buffer(env, agent, 1000, device="cpu")
    with timer.add_time("steps"):
        buffer.collect()
    print((env.n_envs * 1000) / timer.key2time["steps"])
