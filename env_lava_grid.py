import torch

class LavaGrid():
    def __init__(self, size_grid=100, obs_size=9, p_lava=0.15, n_envs=10, dead_screen=True):
        super().__init__()
        assert obs_size % 2==1
        self.size_grid = size_grid
        self.map = torch.rand(self.size_grid, self.size_grid) < p_lava
        self.k = k = obs_size//2
        self.map[k:k+3, k:k+3] = False
        self.map[:k] = True; self.map[:, :k] = True
        self.map[-k:] = True; self.map[:, -k:] = True
        self.n_envs = n_envs
        self.dead_screen = dead_screen

        self.action2vec = torch.tensor([[ 1,  0],
                                        [ 0,  1],
                                        [-1,  0],
                                        [ 0, -1]])
        self.reset()
        
        self.observation_space = type('', (), {})()
        self.observation_space.sample = lambda : torch.rand((self.n_envs, 2*k+1, 2*k+1), device=self.map.device)
        self.observation_space.shape = (2*k+1, 2*k+1)
        self.action_space = type('', (), {})()
        self.action_space.sample = lambda : torch.randint(0, len(self.action2vec), size=(self.n_envs, ), dtype=torch.long, device=self.map.device)
        self.action_space.n = len(self.action2vec)
        
    def to(self, *args, **kwargs):
        self.map = self.map.to(*args, **kwargs)
        self.action2vec = self.action2vec.to(*args, **kwargs)
        return self
        
    def reset(self, snapshot=None):
        if snapshot is None:
            self.snapshot = torch.full((self.n_envs, 2), self.k, dtype=torch.long, device=self.map.device)
        else:
            self.snapshot = snapshot.to(self.map.device)
        obs, done = self.calc_obs_done()
        reward = torch.zeros(self.n_envs, device=self.map.device)
        info = None # [{} for _ in range(self.n_envs)]
        return self.snapshot, obs, reward, done, info
    
    def step(self, action):
        action = self.action2vec[action]
        done = self.map[self.snapshot[:, 0], self.snapshot[:, 1]]
        self.snapshot = torch.where(done[:, None], self.snapshot, self.snapshot + action)
        self.snapshot = torch.clamp(self.snapshot, min=self.k, max=self.size_grid-self.k-1)
        
        obs, done = self.calc_obs_done()
        reward = torch.zeros(self.n_envs, device=self.map.device)
        done = self.map[self.snapshot[:, 0], self.snapshot[:, 1]]
        info = None # [{} for _ in range(self.n_envs)]
        return self.snapshot, obs, reward, done, info
    
    def calc_obs_done(self, snapshot=None):
        snapshot = self.snapshot if snapshot is None else snapshot
        obs = torch.stack([self.map[x-self.k: x+self.k+1, y-self.k: y+self.k+1] for (x, y) in snapshot]).float()
        done = self.map[snapshot[:, 0], snapshot[:, 1]]
        if self.dead_screen:
            obs[done] = 1.
        return obs, done
    
    def to_latent(self, snapshot, obs):
        done = self.map[snapshot[:, 0], snapshot[:, 1]]
        return [(tuple(a.tolist()) if not d else (-1, -1)) for a, d in zip(snapshot, done)]
    
    
    