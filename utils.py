import tensorflow as tf
import numpy as np
import random

def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)
    