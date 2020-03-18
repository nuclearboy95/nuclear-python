import _pickle as p
import json
import yaml
import multiprocessing as mp
import os
from .utils_files import home_rsc_path
from .utils_files import makedirpath
import functools


def ensure_dir_exist(f):
    @functools.wraps(f)
    def wrapper(content, fpath, *args, **kwargs):
        makedirpath(fpath)
        return f(content, fpath, *args, **kwargs)
    return wrapper


def load_binary(fpath, encoding='ASCII'):
    with open(fpath, 'rb') as f:
        return p.load(f, encoding=encoding)


def load_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)


def load_yaml(fpath):
    with open(fpath, 'r') as f:
        return yaml.load(f)


def load_txt(fpath):
    with open(fpath, 'r') as f:
        return f.read()


@ensure_dir_exist
def save_binary(d, fpath):
    with open(fpath, 'wb') as f:
        p.dump(d, f)


@ensure_dir_exist
def save_json(d, fpath):
    with open(fpath, 'w') as f:
        json.dump(d, f, indent=4)


@ensure_dir_exist
def save_yaml(d, fpath):
    with open(fpath, 'w') as f:
        yaml.dump(d, f)


@ensure_dir_exist
def save_txt(s, fpath):
    with open(fpath, 'w') as f:
        return f.write(s)


class AsyncWriter:
    def __init__(self, processes=2):
        self.pool = mp.Pool(processes)
        self.res = None

    def __enter__(self):
        self.pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.res is not None:  # wait until all write is done.
            self.res.get()
        return self.pool.__exit__(exc_type, exc_val, exc_tb)

    def save_binary(self, d, path):
        self.res = self.pool.apply_async(save_binary, (d, path))


def load_rsc(*args):
    path = os.sep.join(args)
    path_rsc = home_rsc_path()
    os.makedirs(path_rsc, exist_ok=True)
    path = os.path.join(path_rsc, path)
    try:
        return load_binary(path)
    except FileNotFoundError:
        return None


def save_rsc(d, *args):
    path = os.sep.join(args)
    path_rsc = home_rsc_path()
    os.makedirs(path_rsc, exist_ok=True)
    path = os.path.join(path_rsc, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_binary(d, path)


ldj = load_json
ldb = load_binary
svj = save_json
svb = save_binary
