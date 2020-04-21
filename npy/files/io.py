import _pickle as p
import json
import yaml
import os
import functools

from .path import home_rsc_path, makedirpath


def ensure_dir_exist(f):
    @functools.wraps(f)
    def wrapper(content, fpath, *args, **kwargs):
        makedirpath(fpath)
        return f(content, fpath, *args, **kwargs)
    return wrapper


def load_binary(fpath, encoding='ASCII'):
    with open(fpath, 'rb') as f:
        return p.load(f, encoding=encoding)


def load_json(fpath, encoding=None):
    with open(fpath, 'r', encoding=encoding) as f:
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
def save_json(d, fpath, ensure_ascii=True):
    with open(fpath, 'w') as f:
        json.dump(d, f, indent=4, ensure_ascii=ensure_ascii)


@ensure_dir_exist
def save_yaml(d, fpath):
    with open(fpath, 'w') as f:
        yaml.dump(d, f)


@ensure_dir_exist
def save_txt(s, fpath):
    with open(fpath, 'w') as f:
        return f.write(s)


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
