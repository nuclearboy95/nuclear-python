"""
hook factories. not hooks
"""

import inspect as ins
from npy import info2


def get_codeline(frame):
    return f"{frame.filename}:{frame.lineno};  {frame.code_context[0].strip()}"


def print_stack(N=4):
    def hook(module, input_, output_):
        frames = ins.stack()[::-1]

        if N > 0:
            frames = frames[:-N]

        for frame in frames:
            print(get_codeline(frame))

        print("-" * 20)

    return hook


def log_pre(
    inspect=True,
    stack_depth=5,
    with_kwargs=True,
    range=False,
    device=False,
    dtype=False,
):
    info_kwargs = {
        "dict_equal": True,
        "no_range": not range,
        "no_device": not device,
        "no_dtype": not dtype,
    }

    if with_kwargs:

        def hook(module, input_, inputkwargs):
            frame = ins.stack()[::-1][-stack_depth]
            res = ""
            if inspect:
                res += get_codeline(frame) + "; "

            s = []

            if len(input_):
                s.append(f"args={info2(input_, **info_kwargs)}")

            if len(inputkwargs):
                s.append(f"kwargs={info2(inputkwargs, **info_kwargs)}")

            res += f"In: {', '.join(s)}"
            print(res)

        return hook

    else:

        def hook(module, input_):
            frame = ins.stack()[::-1][-stack_depth]
            res = ""
            if inspect:
                res += get_codeline(frame) + "; "

            s = []

            if len(input_):
                s.append(f"args={info2(input_, **info_kwargs)}")

            res += f"In: {', '.join(s)}"
            print(res)

    return hook


def log_post(
    inspect=True,
    stack_depth=5,
    with_kwargs=True,
    range=False,
    device=False,
    dtype=False,
):
    info_kwargs = {
        "dict_equal": True,
        "no_range": not range,
        "no_device": not device,
        "no_dtype": not dtype,
    }

    if with_kwargs:

        def hook(module, input_, inputkwargs, output_):
            frame = ins.stack()[::-1][-stack_depth]
            res = ""

            if inspect:
                res += get_codeline(frame) + "; "

            s = []

            if len(input_):
                s.append(f"args={info2(input_, **info_kwargs)}")

            if len(inputkwargs):
                s.append(f"kwargs={info2(inputkwargs, **info_kwargs)}")

            res += f"In: {', '.join(s)} => "

            if output_:
                res += f"{info2(output_, **info_kwargs)}"

            print(res)

        return hook
    else:

        def hook(module, input_, output_):
            frame = ins.stack()[::-1][-stack_depth]
            res = ""
            if inspect:
                res += get_codeline(frame) + "; "

            s = []

            if len(input_):
                s.append(f"args={info2(input_, **info_kwargs)}")

            res += f"In: {', '.join(s)} => "

            if output_:
                res += f"{info2(output_, **info_kwargs)}"

            print(res)

        return hook
