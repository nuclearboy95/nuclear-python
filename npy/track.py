

_TRACK_FLAG = False


__all__ = ['on', 'off']


def on():
    global _TRACK_FLAG
    _TRACK_FLAG = True


def off():
    global _TRACK_FLAG
    _TRACK_FLAG = False
