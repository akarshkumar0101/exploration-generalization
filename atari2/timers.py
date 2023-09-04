import time
from contextlib import contextmanager
from collections import defaultdict


class Timer:
    def __init__(self):
        self.key2time = defaultdict(float)

    @contextmanager
    def add_time(self, key):
        before = time.time()
        yield None
        after = time.time()
        self.key2time[key] += after - before

    def clear(self):
        self.key2time.clear()
