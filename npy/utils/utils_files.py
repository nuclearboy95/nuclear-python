import os


def sub_files(path, endswith=None):
    for root, dirs, files in os.walk(path):
        for file in files:
            if endswith is None or file.endswith(endswith):
                yield os.path.join(root, file)


def path_listdir(dirname, prefix=None, only_files=False, absolute=True):
    result = [os.path.join(dirname, basename)
              for basename in os.listdir(dirname)
              if not prefix or basename.startswith(prefix)]
    result.sort()

    if only_files:
        result = list(filter(os.path.isfile, result))

    if absolute:
        result = list(map(os.path.abspath, result))

    return result


def rpath_listdir2(dname, only_files=False):
    fdnames = path_listdir(dname)
    fdnames.sort()
    fnames = list(filter(os.path.isfile, fdnames))
    dnames = list(filter(os.path.isdir, fdnames))
    result = list()
    result += fnames

    if not only_files:
        result += dnames

    for child in map(lambda dn: rpath_listdir2(os.path.join(dname, dn), only_files=only_files), dnames):
        result += child
    return result


def rpath_listdir(dname, only_files=False, absolute=True):
    result = list(sub_files(dname))
    result.sort()
    if only_files:
        result = list(filter(os.path.isfile, result))

    if absolute:
        result = list(map(os.path.abspath, result))

    return result


def home_path():
    return os.path.expanduser('~')
