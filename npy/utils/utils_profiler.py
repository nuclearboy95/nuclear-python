import cProfile
from collections import defaultdict
from contextlib import contextmanager
import time


__all__ = ['Profiler', 'Clocks']


class Profiler:
    def __init__(self, profile=True, sortby="cumtime"):
        self.pr = cProfile.Profile()
        self.sortby = sortby
        self.detailed = profile
        self.start_time = 0

    def __enter__(self):
        if self.detailed:
            self.pr.enable()
        else:
            self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.detailed:
            self.pr.disable()
            self.pr.print_stats(self.sortby)
        else:
            print("Profile ended. (Took %.1fs)" % (time.time() - self.start_time))


class Clocks:
    def __init__(self):
        self.accums = defaultdict(float)
        self.callcount = defaultdict(int)

    @contextmanager
    def __call__(self, key):
        self.callcount[key] += 1
        s = time.time()
        yield
        e = time.time()
        dur = e - s
        self.accums[key] += dur

    def get(self, N=1):
        return {k: v * 1000 / N for k, v in self.accums.items()}

    def counts(self, N=1):
        return {
            k: v // N if int(v / N) == v // N else v / N
            for k, v in self.callcount.items()
        }

    @property
    def d(self):
        return self.accums
