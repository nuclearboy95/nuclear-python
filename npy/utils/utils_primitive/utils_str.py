import re

__all__ = ['str2float', 'str2int', 'int_or_str2int', 'str2int_l',
           'countchar', 'countcharfront', 'rmchar', 'rmchars', 'rmws',
           'include_exclude', 'endswith_any', 'startswith_any']


def str2float(s: str) -> float:
    if isinstance(s, float) or isinstance(s, int):
        return s

    s = s.strip()
    if s.startswith('(') and s.endswith(')'):
        return -str2int(s[1:-1])

    s = ''.join(re.findall(r'[\d.\-]+', s))
    if not s:
        return 0.
    elif s == '-':
        return 0
    else:
        return float(s)


def str2int(s: str) -> int:
    if isinstance(s, int):
        return s
    elif isinstance(s, float):
        return int(s)

    s = s.strip()
    if s.startswith('(') and s.endswith(')'):
        return -str2int(s[1:-1])

    s = ''.join(re.findall(r'[\d\-]+', s))
    if not s:
        return 0
    elif s == '-':
        return 0
    else:
        return int(s)


def int_or_str2int(v) -> int:
    if isinstance(v, str):
        return str2int(v)
    else:
        return v


def str2int_l(l: list) -> list:
    return list(map(str2int, l))


def countchar(s: str, c: str) -> int:
    return sum([int(cc == c) for cc in s])


def countcharfront(s: str, c: str) -> int:
    count = 0
    for v in s:
        if v == c:
            count += 1
        elif not v.isspace():
            return count
    return count


def rmchar(s: str, c: str) -> str:
    return s.replace(c, '')


def rmchars(s: str, cs: list) -> str:
    for c in cs:
        s = rmchar(s, c)
    return s


def rmws(s: str) -> str:
    return rmchars(s, ['\n', ' ', '\t'])


def include_exclude(s: str, includes: list = None, excludes: list = None) -> bool:
    if includes is not None:
        if any([v not in s for v in includes]):
            return False

    if excludes is not None:
        if any([v in s for v in excludes]):
            return False

    return True


def endswith_any(s: str, prefix_list: list) -> bool:
    return any(s.endswith(prefix) for prefix in prefix_list)


def startswith_any(s: str, prefix_list: list) -> bool:
    return any(s.startswith(prefix) for prefix in prefix_list)
