from collections import defaultdict


__all__ = ['AverageMeter', 'AverageMeters', 'MaxMeter', 'MinMeter']


class MaxMeter:
    def __init__(self):
        self._max = float('-inf')

    def update(self, v):
        self._max = max(self._max, v)

    @property
    def max(self):
        return self._max


class MinMeter:
    def __init__(self):
        self._min = float('inf')

    def update(self, v):
        self._min = max(self._min, v)

    @property
    def min(self):
        return self._min


class AverageMeter:
    def __init__(self):
        self.count = 0
        self.value = 0

    def update(self, v, count=1):
        self.count += count
        self.value += v

    def reset(self):
        self.count = 0
        self.value = 0

    @property
    def avg(self):
        if self.count == 0:
            return 0

        else:
            return self.value / self.count


class AverageMeters:
    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, key, v, count=1):
        self.meters[key].update(v, count)

    def reset(self):
        for key in self.meters.keys():
            self.meters[key].reset()

    def avg(self, key):
        return self.meters[key].avg

    @property
    def avgs(self):
        return {
            key: self.meters[key].avg
            for key in self.meters.keys()
        }
