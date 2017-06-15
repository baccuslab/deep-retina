"""
fit BN_CNN models
"""
import os
import argparse
import tableprint as tp
import tensorflow as tf
from deepretina.core import train
from deepretina.models import bn_cnn


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a BN_CNN model')
    parser.add_argument('--expt', help='Experiment date (e.g. 15-10-07)')
    parser.add_argument('--stim', help='Stimulus class (e.g. naturalscene)')
    args = parser.parse_args()

    tp.banner(f'Training BN-CNN, expt {args.expt}, {args.stim}')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # expts = ('15-10-07', '15-11-21a', '15-11-21b')
    # stims = ('whitenoise', 'naturalscene')

    train(bn_cnn, args.expt, args.stim, lr=1e-2, nb_epochs=250, val_split=0.05)
