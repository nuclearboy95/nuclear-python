__all__ = ['interval_first', 'interval_last']


def interval_last(i, interval):
    return i % interval == interval - 1


def interval_first(i, interval):
    return i % interval == 0
