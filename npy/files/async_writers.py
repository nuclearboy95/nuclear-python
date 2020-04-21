import multiprocessing as mp
from imageio import imsave
from .io import save_binary


__all__ = ['AsyncWriter', 'AsyncImageWriter']


class AsyncWriter:
    def __init__(self, processes=1):
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


class AsyncImageWriter:
    def __init__(self, processes=1):
        self.pool = mp.Pool(processes)
        self.res = None

    def __enter__(self):
        self.pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.res is not None:  # wait until all write is done.
            self.res.get()
        return self.pool.__exit__(exc_type, exc_val, exc_tb)

    def imsave(self, path, image):
        self.res = self.pool.apply_async(imsave, (path, image))
