import os
from .. import load_binary, save_binary
import shutil
from glob import glob
from collections import defaultdict


__all__ = ['fdict', 'FileDict', 'SpawningCache', 'ddict']


class ddict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)

        else:
            ret = self[key] = self.default_factory(key)
            return ret


class fdict:
    def __init__(self, root_folder='./'):
        self.root_folder = root_folder
        os.makedirs(root_folder, exist_ok=True)

    def __getitem__(self, key):  # value = d[key]
        key = self._preprocess_key(key)
        path = self._get_path(key)

        if os.path.exists(path):
            if os.path.isfile(path):
                try:
                    return load_binary(path)
                except EOFError:
                    print('EOFError during loading. Key:', key)
                    raise

            else:
                return fdict(path)

        else:
            return fdict(path)

    def __setitem__(self, key, value):  # d[key] = value
        key = self._preprocess_key(key)
        path = self._get_path(key)
        dirname = os.path.dirname(path)

        # 1. make a directory
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        if os.path.exists(path):
            if os.path.isdir(path):  # 2. if a directory
                shutil.rmtree(path)
        save_binary(value, path)

    def __delitem__(self, key):  # del d[key]
        key = self._preprocess_key(key)
        path = self._get_path(key)

        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)

    @staticmethod
    def _preprocess_key(key):
        return str(key)

    def _get_path(self, key):
        return os.path.join(self.root_folder, key)

    @staticmethod
    def _is_file(path):
        return os.path.exists(path) and os.path.isfile(path)

    def keys(self, recursive=False):
        if recursive:
            allnames = glob(os.path.join(self.root_folder, '**'), recursive=True)
        else:
            allnames = os.listdir(self.root_folder)
        filenames = list(filter(os.path.isfile, allnames))
        return sorted(filenames)


FileDict = fdict


class SpawningCache:
    def __init__(self):
        self.d = {}

    def get(self, key, spawner):
        try:
            return self.d[key]
        except KeyError:
            self.d[key] = spawner
            return self.d[key]

    def set(self, key, value):
        self.d[key] = value
