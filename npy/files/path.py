import os


__all__ = ['rlistdir', 'listdir_path', 'home_path', 'home_rsc_path', 'makedirpath']


def rlistdir(path='.', endswith=None, absolute=False):
    """
    returns a generator that iterates over all sub files (exclude dir) of a given path.

    :param str path:
    :param str endswith:
    :param bool absolute:
    :return:
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if endswith is None or file.endswith(endswith):
                fpath = os.path.join(root, file)
                if absolute:
                    fpath = os.path.abspath(fpath)
                yield fpath


def listdir_path(dpath, prefix=None, file_only=False, absolute=False) -> list:
    """
    Similar to os.listdir() with path suffix.

    :param dpath:
    :param str prefix:
    :param bool file_only:
    :param bool absolute:
    :return:
    """
    fnames = sorted(os.listdir(dpath))

    if prefix is not None:
        fnames = list(filter(lambda fname: fname.startswith(prefix), fnames))

    fpaths = [os.path.join(dpath, fname) for fname in fnames]

    if file_only:
        fpaths = list(filter(os.path.isfile, fpaths))

    if absolute:
        fpaths = list(map(os.path.abspath, fpaths))

    return fpaths


def home_path() -> str:
    return os.path.expanduser('~')


def home_rsc_path() -> str:
    return os.path.join(home_path(), '.nucpy')


def makedirpath(fpath: str):
    dpath = os.path.dirname(fpath)
    if dpath:
        os.makedirs(dpath, exist_ok=True)
