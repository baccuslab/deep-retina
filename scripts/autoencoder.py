%load_ext autoreload
%autoreload 1

import os
import functools
import argparse
import tensorflow as tf
K = tf.keras.backend
import tableprint as tp
import numpy as np
from deepretina import config

import deepretina
import deepretina.experiments
import deepretina.models
import deepretina.core
import deepretina.metrics

# %aimport deepretina
# %aimport deepretina.experiments
# %aimport deepretina.models
# %aimport deepretina.core
# %aimport deepretina.metrics
