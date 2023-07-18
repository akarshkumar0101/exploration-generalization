import numpy as np
import gymnasium as gym

from collections import defaultdict
import cv2
import copy


e1 = 0.001
e2 = 0.00001


def cellfn(frame):
    cell = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cell = cv2.resize(cell, (11, 8), interpolation=cv2.INTER_AREA)
    cell = cell // 32
    return cell


def hashfn(cell):
    return hash(cell.tobytes())


class Weights:
    times_chosen = 0.1
    times_chosen_since_new = 0
    times_seen = 0.3


class Powers:
    times_chosen = 0.5
    times_chosen_since_new = 0.5
    times_seen = 0.5


class Cell(object):
    def __init__(self):
        self.times_chosen = 0
        self.times_chosen_since_new = 0
        self.times_seen = 0

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if key != "score" and hasattr(self, "times_seen"):
            self.score = self.cellscore()

    def cntscore(self, a):
        w = getattr(Weights, a)
        p = getattr(Powers, a)
        v = getattr(self, a)
        return w / (v + e1) ** p + e2

    def cellscore(self):
        return self.cntscore("times_chosen") + self.cntscore("times_chosen_since_new") + self.cntscore("times_seen") + 1

    def visit(self):
        self.times_seen += 1
        return self.times_seen == 1

    def choose(self):
        self.times_chosen += 1
        self.times_chosen_since_new += 1
        return self.ram, self.reward, self.trajectory



env = gym.make(f"ALE/{'MontezumaRevenge'}-v5", frameskip=4, repeat_action_probability=0.0)


_, info = env.reset()
max_lives = info["lives"]

archive = defaultdict(lambda: Cell())
highscore = 0
frames = 0
iterations = 0

env = gym.make(f"ALE/{'MontezumaRevenge'}-v5", frameskip=4, repeat_action_probability=0.0)
env.seed(0)
env.action_space.seed(0)
np.random.seed(0)
frame = env.reset()
score = 0
action = 0
trajectory = []
my_iterations = 0

ff = None
for i in range(100):
    found_new_cell = False
    episode_length = 0

    for i in range(100):
        if np.random.random() > 0.95:
            action = env.action_space.sample()

        frame, reward, terminal, trunc, info = env.step(action)
        score += reward
        terminal |= info["lives"] < max_lives
        terminal |= trunc

        trajectory.append(action)
        episode_length += 4

        if score > highscore:
            highscore = score
            best_cell = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if terminal:
            frames += episode_length
            break
        else:
            cell = cellfn(frame)
            cellhash = hashfn(cell)
            cell = archive[cellhash]
            first_visit = cell.visit()
            if first_visit or score > cell.reward or score == cell.reward and len(trajectory) < len(cell.trajectory):
                if ff is None:
                    ff = cellhash
                if cellhash==ff:
                    print('---------', ff)
                    print(first_visit)
                    if not first_visit:
                        print(score > cell.reward)
                        if not score > cell.reward:
                            print(score == cell.reward and len(trajectory) < len(cell.trajectory))
                            print(score, cell.reward, len(trajectory), len(cell.trajectory))
                cell.ram = env.clone_state(True)
                cell.reward = score
                cell.trajectory = copy.deepcopy(trajectory.copy())
                cell.times_chosen = 0
                cell.times_chosen_since_new = 0
                new_cell = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                found_new_cell = True
                if cellhash==ff:
                    print('-')
                    print(first_visit)
                    if not first_visit:
                        print(score > cell.reward)
                        if not score > cell.reward:
                            print(score == cell.reward and len(trajectory) < len(cell.trajectory))
                            print(score, cell.reward, len(trajectory), len(cell.trajectory))
                

    if found_new_cell and my_iterations > 0:
        restore_cell.times_chosen_since_new = 0

    scores = np.array([cell.score for cell in archive.values()])
    hashes = [cellhash for cellhash in archive.keys()]
    probs = scores / scores.sum()
    restore = np.random.choice(hashes, p=probs)
    restore_cell = archive[restore]
    ram, score, trajectory = restore_cell.choose()
    if restore==ff:
        print('restoring root ', len(trajectory))
    env.reset()
    env.restore_state(ram)
    my_iterations += 1
    iterations += 1
