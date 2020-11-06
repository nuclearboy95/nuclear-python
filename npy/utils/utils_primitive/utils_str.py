import re

__all__ = ['str2float', 'str2int', 'str2int_l', 'countchar', 'rmchar']


def str2float(v: str) -> float:
    v = v.strip()
    if v.startswith('(') and v.endswith(')'):
        return -str2int(v[1:-1])

    v = ''.join(re.findall(r'[\d.\-]+', v))
    if not v:
        return 0.
    elif v == '-':
        return 0
    else:
        return float(v)


def str2int(v: str) -> int:
    v = v.strip()
    if v.startswith('(') and v.endswith(')'):
        return -str2int(v[1:-1])

    v = ''.join(re.findall(r'[\d\-]+', v))
    if not v:
        return 0
    elif v == '-':
        return 0
    else:
        return int(v)


def str2int_l(l: list) -> list:
    return list(map(str2int, l))


def countchar(s: str, c: str) -> int:
    return sum([int(cc == c) for cc in s])


def rmchar(s: str, c: str) -> str:
    return s.replace(c, '')
