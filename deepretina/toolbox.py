"""
Toolbox with helpful utilities for exploring Keras models
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import theano
import h5py
import os
import tableprint
from keras.models import model_from_json
from .experiments import loadexpt
from . import metrics
from .utils import notify
from .visualizations import visualize_convnet, visualize_glm
from functools import partial
import pandas as pd
import json


class Models:

    def __init__(self, basedir):
        """Creates a data structure to hold the list of models"""

        self.basedir = basedir
        self.models = {}

        # scan base directory for models folders
        with notify('Scanning {} for all models'.format(basedir)):
            for folder in os.listdir(basedir):

                try:
                    # load the description string
                    with open(os.path.join(basedir, folder, 'README.md'), 'r') as f:
                        lines = f.readlines()
                        desc_index = [line.strip() for line in lines].index('### description')
                        self.models[folder] = lines[desc_index + 1].strip()

                except (NotADirectoryError, FileNotFoundError, ValueError):
                    pass

    def search(self, text, disp=True):
        """Searches model descriptions for the given text"""

        keys = list()
        for key, value in self.models.items():
            if text in value:
                keys.append(key)
                if disp:
                    highlighted = value.replace(text, '\033[94m{}\033[0m'.format(text))
                    print('{}: {}'.format(key, highlighted))

        if len(keys) == 0:
            print('Did not find "{}" in any model descriptions!'.format(text))

        return keys

    def filepath(self, *args):
        """Returns the filepath of the given key + arguments"""
        return os.path.join(self.basedir, *args)

    def weights(self, key, filename='best_weights.h5'):
        """Returns a function that loads weights from the given filename, for the given key"""
        return partial(get_weights, self.filepath(key, filename))


class Model:

    def __init__(self, basedir, key):
        self.basedir = basedir
        self.key = key
        self.hashkey, self.modeltype = key.split(' ')

    def filepath(self, *args):
        """Returns the filepath of the given key + arguments"""
        return os.path.join(self.basedir, self.key, *args)

    def __getitem__(self, filename):
        """Loads the given file from this model's directory"""

        # add file extension if the filename is one of these special cases
        if filename in ('metadata', 'architecture', 'experiment'):
            filename += '.json'
        elif filename in ('best_weights'):
            filename += '.h5'
        elif filename in ('train', 'validation'):
            filename += '.csv'

        # get the name and extension
        _, ext = filename.split('.')
        fullpath = self.filepath(filename)

        if ext == 'json':
            with open(fullpath, 'r') as f:
                return json.load(f)

        elif ext == 'h5':
            return h5py.File(fullpath, 'r')

        elif ext == 'csv':
            return pd.read_csv(fullpath)

        else:
            raise ValueError('Could not parse extension "{}"'.format(ext))

    def plot(self, filename='best_weights.h5'):
        """Plots the parameters of this model"""

        if self.modeltype == 'convnet':
            figures = visualize_convnet(self[filename], self['architecture.json']['layers'])

        elif self.modeltype.lower() == 'glm':
            figures = visualize_glm(self[filename])

        else:
            raise ValueError("I don't know how to plot a model of type '{}'".format(self.modeltype))

        return figures


def load_model(model_path, weight_filename):
    """
    Loads a Keras model using:
    - an architecture.json file
    - an h5 weight file, for instance 'epoch018_iter01300_weights.h5'

    INPUT:
        model_path		the full path to the saved weight and architecture files, ending in '/'
        weight_filename	an h5 file with the weights
        OUTPUT:
        returns keras model
    """

    architecture_filename = 'architecture.json'
    with open(os.path.join(model_path, architecture_filename), 'r') as architecture_data:
        architecture_string = architecture_data.read()
        model = model_from_json(architecture_string)
        model.load_weights(os.path.join(model_path, weight_filename))

    return model


def load_partial_model(model, layer_id):
    """
    Returns the model up to a specified layer.

    INPUT:
        model       a keras model
        layer_id    an integer designating which layer is the new final layer

    OUTPUT:
        a theano function representing the partial model
    """

    # create theano function to generate activations of desired layer
    return theano.function([model.layers[0].input], model.layers[layer_id].get_output(train=False))


def list_layers(model_path, weight_filename):
    """
    Lists the layers in the model with their children.

    This provides an easy way to see how many "layers" in the model there are, and which ones
    have weights attached to them.

    Layers without weights and biases are relu, pool, or flatten layers.

    INPUT:
        model_path		the full path to the saved weight and architecture files, ending in '/'
        weight_filename	an h5 file with the weights

    OUTPUT:
        an ASCII table using tableprint
    """
    weights = h5py.File(model_path + weight_filename, 'r')
    layer_names = list(weights)

    # print header
    print(tableprint.hr(3))
    print(tableprint.header(['layer', 'weights', 'biases']))
    print(tableprint.hr(3))

    params = []
    for l in layer_names:
        params.append(list(weights[l]))
        if params[-1]:
            print(tableprint.row([l.encode('ascii', 'ignore'),
                  params[-1][0].encode('ascii', 'ignore'),
                  params[-1][1].encode('ascii', 'ignore')]))
        else:
            print(tableprint.row([l.encode('ascii', 'ignore'), '', '']))

    print(tableprint.hr(3))


def get_test_responses(model, stim_type='whitenoise', cells=[0], exptdate='15-10-07'):
    """Get a list of [true_responses, model_responses] on the same test data."""
    test_data = loadexpt(cells, stim_type, 'test', 40, exptdate=exptdate)

    truth = []
    predictions = []
    for X, y in datagen(50, *test_data, shuffle=False):
        truth.extend(y)
        predictions.extend(model.predict(X))

    truth = np.array(truth)
    predictions = np.array(predictions)

    return [truth, predictions]


def get_correlation(model, stim_type='natural', cells=[0], metric='cc'):
    """Get Pearson's r correlation."""
    truth, predictions = get_test_responses(model, stim_type=stim_type, cells=cells)

    metric_func = getattr(metrics, metric)

    test_cc = []
    for c in cells:
        test_cc.append(metric_func(truth[:, c], predictions[:, c]))

    return test_cc


def get_performance(model, stim_type='natural', cells=[0], metric='cc'):
    """
        Get correlation coefficient on held-out data for deep-retina.

        INPUT:
            model           Keras model
            stim_type       'natural' or 'white'; which test data to draw from?
            cells           list of cell indices
            metric          'cc' (scipy.stats.pearsonr),
                            'lli' (Log-likelihood improvement over a mean rate model in bits per spike),
                            'rmse' (Root mean squared error),
                            'fev' (Fraction of explained variance; note this does not take into account
                                    the variance from trial-to-trial)
    """
    truth, predictions = get_test_responses(model, stim_type=stim_type, cells=cells)

    # metric (function computing a score between true and predicted rates)
    metric_func = getattr(metrics, metric)

    # compute the test results
    test_results = [metric_func(truth[:, c], predictions[:, c]) for c in cells]

    return test_results


def get_weights(filepath, layer=0, param=0):
    """
    Return the weights from a saved .h5 file.

    Parameters
    ----------
    filepath : str
        Path to the weights h5 file

    layer : int or str
        name of the layer. If an int, assumed to be 'layer_x', where x is the int,
        otherwise, the given string is used

    param : int or str
        name of the parameter within this layer. If an int, assumed to be 'param_x',
        where x is the int, otherwise, the given string is used. param_0 stores the
        weights, and param_1 stores the biases
    """

    assert isinstance(layer, (str, int)), "Layer argument must be an int or a string"
    assert isinstance(param, (str, int)), "Param argument must be an int or a string"

    if isinstance(layer, int):
        layer = 'layer_{:d}'.format(layer)

    if isinstance(param, int):
        param = 'param_{:d}'.format(param)

    with h5py.File(filepath, 'r') as f:
        weights = np.array(f[layer][param]).astype('float64')

    return weights
