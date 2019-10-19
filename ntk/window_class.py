from abc import abstractmethod, ABCMeta
from tkinter import Tk, Frame, messagebox
from contextlib import contextmanager


__all__ = ['Window', 'T']


@contextmanager
def T(widget):
    yield widget


class Window(metaclass=ABCMeta):
    def __init__(self):
        self.root = Tk()
        self.canvas = dict()
        self.popup = messagebox

    # set functions

    def set_title(self, title):
        self.root.title(title)

    def set_geometry(self, w=1200, h=800, wo=100, ho=100):
        self.root.geometry('%dx%d+%d+%d' % (w, h, wo, ho))

    def set_resizable(self, h, w):
        self.root.resizable(h, w)

    # core

    def main(self):
        self.set_layout()
        self.root.mainloop()

    # implement

    @abstractmethod
    def set_layout(self):
        raise NotImplementedError
