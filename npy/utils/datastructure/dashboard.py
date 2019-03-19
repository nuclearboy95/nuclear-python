import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import os


class Dashboard:
    __metaclass__ = ABCMeta

    @abstractmethod
    def draw(self):
        pass

    def show(self):
        self.draw()
        plt.show()

    def save(self, save_path, **kwargs):
        self.draw()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, **kwargs)
