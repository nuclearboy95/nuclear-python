__all__ = ['AccuracyMeter']


class AccuracyMeter:
    def __init__(self):
        self.count = 0
        self.correct = 0

    def add(self, correct, count):
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
