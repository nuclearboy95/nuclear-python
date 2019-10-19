TRACK_FLAG = False
TRACK_TIME_FLAG = False


__all__ = ['on', 'off']


def on(watch_time=False):
    global TRACK_FLAG
    global TRACK_TIME_FLAG

    TRACK_FLAG = True
    TRACK_TIME_FLAG = watch_time


def off():
    global TRACK_FLAG
    TRACK_FLAG = False
