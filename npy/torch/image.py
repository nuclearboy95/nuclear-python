import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


__all__ = ['show_imagenet_tensor', 'preprocess_imagenet_pil', 'read_imagenet_tensor',
           'preprocess_imagenet',
           ]


MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


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


preprocess_imagenet_pil = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_IMAGENET,
                                     std=STD_IMAGENET),
            ])


preprocess_imagenet = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_IMAGENET,
                                     std=STD_IMAGENET),
            ])


read_imagenet_tensor = transforms.Compose([
    lambda x: Image.open(x),
    preprocess_imagenet_pil,
    lambda x: torch.unsqueeze(x, 0)
])
