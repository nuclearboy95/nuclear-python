# target for torch
import numpy as np
import torch

__all__ = ["info2"]


def apply_depth(s, depth):
    return "\n".join(["    " * depth + v for v in s.split("\n")])


def _info_ndarray(obj, depth=0, **kwargs) -> str:
    """

    :param np.ndarray obj:
    :param str name:
    :return:
    """
    res_str = ""
    res_str += f"np: {obj.shape} {obj.dtype}"
    if np.prod(obj.shape) != 0:
        if not kwargs.get("no_range", False):
            res_str += f" {obj.min():.3g} ~ {obj.max():.3g}"
    else:
        res_str += f" {obj}"

    return res_str


def _info_list(obj, depth=0, **kwargs) -> str:
    """

    :param list obj:
    :param str name:
    :return:
    """
    length = len(obj)
    res_str = ""

    if length == 0:
        res_str += "[]"
    elif length == 1:
        res_str += f"[{type(obj[0])}]"
    else:
        res_str += f"list: len={length}"

        if length > 8:
            res_str += f"type([0])={type(obj[0])}"
        else:  # todo
            res_str += (
                " [\n"
                + apply_depth(
                    ",\n".join([info2(v, depth=depth + 1, **kwargs) for v in obj]),
                    depth,
                )
                + "\n]"
            )

    return res_str


def _info_tuple(obj, depth=0, **kwargs) -> str:
    """

    :param tuple obj:
    :param str name:
    :return:
    """
    length = len(obj)
    res_str = ""

    if length == 0:
        res_str += "(,)"
    elif length == 1:
        res_str += f"({info2(obj[0], depth=depth, **kwargs)},)"
    else:
        res_str += f"tuple: len={length}"
        if length > 10:
            res_str += f" type([0])={type(obj[0])}"
        else:  # todo
            res_str += (
                " (\n"
                + apply_depth(
                    ",\n".join([info2(v, depth=depth + 1, **kwargs) for v in obj]),
                    depth,
                )
                + "\n)"
            )

    return res_str


def _info_dict(obj, depth=0, **kwargs) -> str:
    """

    :param tuple obj:
    :param str name:
    :return:
    """
    length = len(obj)
    res_str = ""

    def format_one(k, v):
        if kwargs.get('dict_equal', False):
            return f'{k}={info2(v, depth=depth + 1, **kwargs)}'
        else:
            return f'"{k}": {info2(v, depth=depth + 1, **kwargs)}'

    if length == 0:
        res_str += "{}"

    elif length == 1:
        k = list(obj.keys())[0]
        res_str += f'{{{format_one(k, obj[k])}}}'

    else:
        res_str += f"dict: len={length}"
        res_str += (
            " {\n"
            + apply_depth(
                ",\n".join([format_one(k, v) for k, v in obj.items()]),
                depth,
            )
            + "\n}"
        )

    return res_str


def _info_torchtensor(obj, depth=0, **kwargs) -> str:
    """
    :param torch.Tensor obj:
    :param str name:
    :return:
    """

    dtype = str(obj.dtype).split(".")[-1]
    dtype = dtype.replace("float", "f").replace("int", "i").replace("bool", "b")

    res_str = "torch"

    if not kwargs.get("no_device", False):
        res_str += f"[{obj.device}]"

    res_str += ":"

    if not kwargs.get("no_dtype", False):
        res_str += f"{dtype}"

    if not kwargs.get("no_shape", False):
        res_str += f"{tuple(obj.shape)}"

    if np.prod(obj.shape) != 0:
        if not kwargs.get("no_range", False):
            res_str += f", {obj.min():.3g} ~ {obj.max():.3g}"
    else:
        res_str += f"={obj}"

    return res_str


def info2(obj, depth=1, **kwargs) -> str:
    import torch

    if isinstance(obj, np.ndarray):
        return _info_ndarray(obj, depth=depth, **kwargs)

    elif isinstance(obj, torch.Tensor):
        return _info_torchtensor(obj, depth=depth, **kwargs)

    elif isinstance(obj, tuple):
        return _info_tuple(obj, depth=depth, **kwargs)

    elif isinstance(obj, dict):
        return _info_dict(obj, depth=depth, **kwargs)

    elif isinstance(obj, list):
        return _info_list(obj, depth=depth, **kwargs)

    return str(obj)
