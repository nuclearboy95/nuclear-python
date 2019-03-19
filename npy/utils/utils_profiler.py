import cProfile
import time


class Profiler:
    def __init__(self, profile=True, sortby='cumtime'):
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
            print('Profile ended. (Took %.1fs)' % (time.time() - self.start_time))
