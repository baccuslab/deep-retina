"""
fit LN and BN_CNN models
"""
import os
import functools
import argparse
import tableprint as tp
from deepretina.core import train
from deepretina.models import bn_cnn, linear_nonlinear


def fit_bn_cnn(expt, stim):
    train(bn_cnn, expt, stim, lr=1e-2, nb_epochs=250, val_split=0.05)


def fit_ln(expt, stim, activation, ci, l2_reg=0.1):
    model = functools.partial(linear_nonlinear, activation=activation, l2_reg=l2_reg)
    tp.banner(f'Training LN-{activation}, expt {args.expt}, {args.stim}, cell {ci+1:02d}')
    train(model, expt, stim, lr=1e-2, nb_epochs=250, val_split=0.05, cells=[ci])


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # expts = ('15-10-07', '15-11-21a', '15-11-21b')
    # stims = ('whitenoise', 'naturalscene')

    parser = argparse.ArgumentParser(description='Train a BN_CNN model')
    parser.add_argument('--expt', help='Experiment date (e.g. 15-10-07)')
    parser.add_argument('--stim', help='Stimulus class (e.g. naturalscene)')
    parser.add_argument('--model', help='Model architecture (e.g. BN_CNN or LN_softplus)')
    parser.add_argument('--cell', help='Cell to train on (only used for LN models)')
    args = parser.parse_args()

    if args.model.upper() == 'BN_CNN':
        tp.banner(f'Training BN_CNN, expt {args.expt}, {args.stim}')
        fit_bn_cnn(args.expt, args.stim)

    elif args.model.split('_')[0].upper() == 'LN':
        activation = args.model.split('_')[1]
        fit_ln(args.expt, args.stim, activation, args.cell)
