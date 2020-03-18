__all__ = ['AccuracyMeter', 'AverageMeter', 'MaxMeter', 'MinMeter']


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


class AccuracyMeter:
    def __init__(self):
        self.count = 0
        self.correct = 0

    def update(self, correct, count):
        self.correct += correct
        self.count += count

    def reset(self):
        self.count = 0
        self.correct = 0

    @property
    def accuracy(self):
        if self.count == 0:
            return 0

        else:
            return self.correct / self.count


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
    def average(self):
        if self.count == 0:
            return 0

        else:
            return self.value / self.count
