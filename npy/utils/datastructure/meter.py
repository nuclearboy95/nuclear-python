from collections import defaultdict


__all__ = ['AverageMeter', 'AverageMeters', 'MaxMeter', 'MinMeter']


class MaxMeter:
    def __init__(self):
        self._max = float('-inf')

    def update(self, value):
        self._max = max(self._max, value)

    @property
    def max(self):
        return self._max


class MinMeter:
    def __init__(self):
        self._min = float('inf')

    def update(self, value):
        self._min = max(self._min, value)

    @property
    def min(self):
        return self._min


class AverageMeter:
    def __init__(self):
        self.count = 0
        self.accumulate = 0
        self.last_value = 0

    def update(self, value, count=1):
        self.count += count
        self.accumulate += value
        self.last_value = value

    def reset(self):
        self.count = 0
        self.accumulate = 0
        self.last_value = 0

    @property
    def avg(self):
        if self.count == 0:
            return 0

        else:
            return self.accumulate / self.count


class AverageMeters:
    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, key, value, count=1):
        self.meters[key].update(value, count)

    def reset(self):
        for key in self.meters.keys():
            self.meters[key].reset()

    def keys(self):
        return sorted(self.meters.keys())

    ###########################

    def avg(self, key):
        return self.meters[key].avg

    def last_value(self, key):
        return self.meters[key].last_value

    @property
    def avgs(self):
        return {
            key: self.meters[key].avg
            for key in self.meters.keys()
        }

    @property
    def last_values(self):
        return {
            key: self.meters[key].last_value
            for key in self.meters.keys()
        }
