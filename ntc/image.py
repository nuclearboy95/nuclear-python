import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from .constants import MEAN_IMAGENET, STD_IMAGENET

__all__ = ['show_imagenet_tensor',
           'preprocess_imagenet', 'unpreprocess_imagenet',
           'NHWC2NCHW', 'NCHW2NHWC', 'HWC2CHW', 'CHW2HWC',
           'as_numpy'
           ]


# Plots image from tensor
def show_imagenet_tensor(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array(MEAN_IMAGENET)
    std = np.array(STD_IMAGENET)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


def preprocess_imagenet(images) -> torch.Tensor:
    """

    :param np.ndarray images:
    :return:
    """
    images = images.astype(np.float32)
    images /= 255.
    images -= np.array(MEAN_IMAGENET)
    images /= np.array(STD_IMAGENET)
    images = NHWC2NCHW(images)
    images = torch.from_numpy(images)
    return images


def unpreprocess_imagenet(images) -> np.ndarray:
    """

    :param torch.Tensor images:
    :return:
    """
    images = images.detach().cpu().numpy()
    images = NCHW2NHWC(images)
    images *= np.array(STD_IMAGENET)
    images += np.array(MEAN_IMAGENET)
    images *= 255.
    np.clip(images, 0, 255, out=images)
    images = images.astype(np.uint8)
    return images


def NCHW2NHWC(images: np.ndarray) -> np.ndarray:
    return np.transpose(images, [0, 2, 3, 1])


def NHWC2NCHW(images: np.ndarray) -> np.ndarray:
    return np.transpose(images, [0, 3, 1, 2])


def HWC2CHW(image: np.ndarray) -> np.ndarray:
    return np.transpose(image, [2, 0, 1])


def CHW2HWC(image: np.ndarray) -> np.ndarray:
    return np.transpose(image, [1, 2, 0])


def as_numpy(image_or_images: torch.Tensor) -> np.ndarray:
    image_or_images = image_or_images.detach().cpu().numpy()
    image_or_images = np.clip(np.transpose(image_or_images, [0, 2, 3, 1]) * 255, 0, 255).astype(np.uint8)
    return image_or_images
