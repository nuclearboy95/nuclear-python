from scipy.misc import imread
import os


__all__ = ['ostrich', 'saxophone']


def _get_example_image_folder():
    current_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_path)
    return os.path.join(current_dir, 'example_images')


def ostrich():
    image_dir = _get_example_image_folder()
    path = os.path.join(image_dir, 'ostrich.png')
    return imread(path)


def saxophone():
    image_dir = _get_example_image_folder()
    path = os.path.join(image_dir, 'saxophone.png')
    return imread(path)
