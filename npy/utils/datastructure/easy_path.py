import os
from ..utils_files import home_path


class EasyPath:
    def __init__(self, path=''):
        if not path:
            path = home_path()
        if not self.is_valid(path):
            raise ValueError('Invalid path: {}'.format(path))
        self.path = path

    def __getattr__(self, item):
        return EasyPath(os.path.join(self.path, item))

    def __str__(self):
        return self.path

    as_str = __str__

    def __repr__(self):
        return 'EasyPath("{}")'.format(self.path)

    @staticmethod
    def is_valid(path):
        return True
