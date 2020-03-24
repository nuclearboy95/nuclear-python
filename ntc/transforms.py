import numpy as np
import torchvision.transforms.functional as TF


__all__ = ['RandomRotation']


class RandomRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = np.random.choice(self.angles)
        return TF.rotate(x, angle)
