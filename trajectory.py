


class Trajectory:
    pass



class TrajectoryElement:
    def __init__(self, state_sim, obs, latent, action, reward, done, info):
        self.state_sim = state_sim
        self.obs = obs
        self.latent = latent
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info
