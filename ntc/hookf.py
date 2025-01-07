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
    stack_depth=6,
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
            print_forward(
                module,
                in_args=input_,
                in_kwargs=inputkwargs,
                inspect=inspect,
                stack_depth=stack_depth,
                info_kwargs=info_kwargs,
            )

        return hook
    else:

        def hook(module, input_):
            print_forward(
                module,
                in_args=input_,
                inspect=inspect,
                stack_depth=stack_depth,
                info_kwargs=info_kwargs,
            )

        return hook


def log_post(
    stack_depth=6,
    inspect=True,
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
            print_forward(
                module,
                in_args=input_,
                in_kwargs=inputkwargs,
                output=output_,
                inspect=inspect,
                stack_depth=stack_depth,
                info_kwargs=info_kwargs,
            )

        return hook
    else:

        def hook(module, input_, output_):
            print_forward(
                module,
                in_args=input_,
                output=output_,
                inspect=inspect,
                stack_depth=stack_depth,
                info_kwargs=info_kwargs,
            )

        return hook


def print_forward(
    module,
    in_args=None,
    in_kwargs=None,
    output=None,
    inspect=True,
    stack_depth=6,
    info_kwargs=None,
):
    frame = ins.stack()[::-1][-stack_depth]
    res = ""

    if inspect:
        res += get_codeline(frame) + "; "

    any_args = in_args is not None and len(in_args)
    any_kwargs = in_kwargs is not None and len(in_kwargs)
    
    l = []
    if any_args:
        for v in in_args:
            l.append(info2(v, **info_kwargs))

    if any_kwargs:
        for k, v in in_kwargs.items():
            l.append(f"{k}={info2(v, **info_kwargs)}")
            
    res += f"f({', '.join(l)})"    

    if output is not None:
        res += f" => {info2(output, **info_kwargs)}"

    print(res)
