import pandas as pd
import numpy as np
import time
import torch
from collections import defaultdict
from contextlib import contextmanager

from torch.profiler import record_function
from npy import info2, task


__all__ = ["ModuleAnalyzer"]


class ModuleAnalyzer:
    def __init__(self, module, f_module_filter=None):
        self.module = module
        self.f_module_filter = f_module_filter
        self.stimes = {}

        self.res_runtime = defaultdict(float)
        self.res_paramcount = defaultdict(int)
        self.res_outshape = defaultdict(str)

        self.d_recordfunctions = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        pass

    ##############

    def hookf_flops_pre(self, key):
        def hook(module, input_):
            self.d_recordfunctions[key] = record_function(key)
            self.d_recordfunctions[key].__enter__()

        return hook

    def hookf_flops_post(self, key):
        def hook(module, input_, output_):
            self.d_recordfunctions[key].__exit__(None, None, None)

        return hook

    ########

    def hookf_runtime_pre(self, key):
        def hook(module, input_):
            s = time.time()
            self.stimes[key] = s

        return hook

    def hookf_runtime_post(self, key):
        def hook(module, input_, output_):
            e = time.time()
            dur = e - self.stimes[key]
            self.stimes[key] = 0  # del self.stimes[key]
            self.res_runtime[key] += dur

        return hook

    def hookf_others_post(self, key):
        def hook(module, input_, output_):

            with task("2. Param count"):
                c = 0
                for p in module.parameters():
                    c += np.prod(tuple(p.shape))

                self.res_paramcount[key] = c

            with task("3. Output shape"):
                if isinstance(output_, tuple) and len(output_) == 1:
                    output_ = output_[0]

                if isinstance(output_, torch.Tensor):
                    shape = tuple(output_.shape)
                else:
                    shape = ""
                self.res_outshape[key] = shape

        return hook

    #################

    @property
    def result(self):
        ks = self.res_paramcount.keys()
        ret = [
            dict(
                key=k,
                params=self.res_paramcount[k],
                outshape=self.res_outshape[k],
                runtime=self.res_runtime[k],
            )
            for k in ks
        ]
        df = pd.DataFrame(ret).set_index("key")
        return df

    @contextmanager
    def record_runtime(self, N=1):
        handles = []
        for name, m in self.module.named_modules():
            if self.f_module_filter is not None:
                if not self.f_module_filter(name, m):
                    continue

            handles.append(m.register_forward_pre_hook(self.hookf_runtime_pre(name)))
            handles.append(m.register_forward_hook(self.hookf_runtime_post(name)))

        yield

        self.res_runtime = {k: v * 1_000_000 / N for k, v in self.res_runtime.items()}

        for handle in handles:
            handle.remove()

    @contextmanager
    def record_others(self):
        handles = []
        for name, m in self.module.named_modules():
            if self.f_module_filter is not None:
                if not self.f_module_filter(name, m):
                    continue

            handles.append(m.register_forward_hook(self.hookf_others_post(name)))

        yield

        for handle in handles:
            handle.remove()

    @contextmanager
    def record_flops(self):
        handles = []
        for name, m in self.module.named_modules():
            if self.f_module_filter is not None:
                if not self.f_module_filter(name, m):
                    continue

            handles.append(m.register_forward_pre_hook(self.hookf_flops_pre(name)))
            handles.append(m.register_forward_hook(self.hookf_flops_post(name)))

        yield

        for handle in handles:
            handle.remove()
