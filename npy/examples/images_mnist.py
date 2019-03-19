from scipy.misc import imread
from functools import partial
import os


__all__ = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


def _get_example_image_folder():
    current_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_path)
    return os.path.join(current_dir, 'example_images/mnist')


def load_digit(digit):
    image_dir = _get_example_image_folder()
    path = os.path.join(image_dir, '%d.png' % digit)
    return imread(path)


zero = partial(load_digit, digit=0)
one = partial(load_digit, digit=1)
two = partial(load_digit, digit=2)
three = partial(load_digit, digit=3)
four = partial(load_digit, digit=4)
five = partial(load_digit, digit=5)
six = partial(load_digit, digit=6)
seven = partial(load_digit, digit=7)
eight = partial(load_digit, digit=8)
nine = partial(load_digit, digit=9)