import numpy as np
import os
import shutil
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from imageio import imread, imsave
import tensorflow as tf
import npy
from npy.ns import *
import ntf

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

set_tf_log(5)
