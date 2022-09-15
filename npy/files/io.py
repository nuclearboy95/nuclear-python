import _pickle as p
import json
import yaml
import os
import functools
import cv2
import numpy as np

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


def load_video(fpath):
    vidcap = cv2.VideoCapture(fpath)

    ret = []
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            ret.append(image)

    ret = np.stack(ret)
    if ret.shape[-1] == 3:
        ret = ret[..., [2, 1, 0]]
    return ret


@ensure_dir_exist
def save_video(arr, fpath, fourcc='DIVX', fps=24, **kwargs):
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    N, H, W = arr.shape[:3]

    if arr.shape[-1] == 3:  # BGR to RGB
        arr = arr[..., [2, 1, 0]]

    vid = cv2.VideoWriter(fpath, fourcc, fps, (W, H), **kwargs)

    for frame in arr:
        vid.write(frame)
    vid.release()


@ensure_dir_exist
def save_binary(d, fpath):
    with open(fpath, 'wb') as f:
        p.dump(d, f)


@ensure_dir_exist
def save_json(d, fpath, ensure_ascii=True, encoding=None):
    with open(fpath, 'w', encoding=encoding) as f:
        json.dump(d, f, indent=4, ensure_ascii=ensure_ascii)


@ensure_dir_exist
def save_yaml(d, fpath):
    with open(fpath, 'w') as f:
        yaml.dump(d, f)


@ensure_dir_exist
def save_txt(s, fpath):
    with open(fpath, 'w') as f:
        return f.write(s)


def get_rsc_path(*args):
    fpath = os.sep.join(args)
    path_rsc = home_rsc_path()
    fpath = os.path.join(path_rsc, fpath)
    return fpath


def load_rsc(*args):
    fpath = get_rsc_path(*args)
    try:
        return load_binary(fpath)
    except FileNotFoundError:
        return None


def load_rsc_json(*args, encoding=None):
    fpath = get_rsc_path(*args) + '.json'
    try:
        return load_json(fpath, encoding=encoding)
    except FileNotFoundError:
        return None


def save_rsc(d, *args):
    fpath = get_rsc_path(*args)
    save_binary(d, fpath)


def save_rsc_json(d, *args, ensure_ascii=True):
    fpath = get_rsc_path(*args) + '.json'
    save_json(d, fpath, ensure_ascii=ensure_ascii)


ldj = load_json
ldb = load_binary
svj = save_json
svb = save_binary
