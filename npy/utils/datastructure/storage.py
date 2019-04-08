import os
from .. import load_binary, save_binary
import shutil
from glob import glob


class FileDict:
    def __init__(self, root_folder='./'):
        self.root_folder = root_folder
        os.makedirs(root_folder, exist_ok=True)

    def __getitem__(self, key):
        key = self._preprocess_key(key)
        path = self._get_path(key)

        if self._is_file(path):
            try:
                return load_binary(path)
            except EOFError:
                print('EOFError during loading. Key:', key)
                raise
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        key = self._preprocess_key(key)
        path = self._get_path(key)
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        save_binary(value, path)

    def __delitem__(self, key):
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

    def keys(self):
        allnames = glob('**', recursive=True)
        filenames = list(filter(os.path.isfile, allnames))
        filenames.sort()
        return filenames


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
