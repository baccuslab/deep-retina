"""
Toolbox with helpful utilities for exploring Keras models
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import theano
import h5py
import os
import re
import tableprint
from keras.models import model_from_json, model_from_config
from .experiments import loadexpt
from . import metrics
from .visualizations import visualize_convnet, visualize_glm
import pandas as pd
import json
from tqdm import tqdm
from operator import attrgetter


def scandb(directory):
    """Scans the given directory for any model folders and returns a
    list of Model objects for each folder in the directory
    """
    models = []

    # scan base directory for models folders, only load those with a README
    regex = re.compile(r'^([a-z0-9]){6} ')
    for folder in tqdm(filter(regex.match, os.listdir(directory))):
        if os.path.isdir(os.path.join(directory, folder)):
            models.append(Model(directory, folder))

    return sorted(models, key=attrgetter('timestamp'), reverse=True)


def matcher(desc, key):
    """Returns a fucntion that matches a model given the desc and key"""

    def wrapper(model):

        # return False if the model key is not matched
        if not (key.lower() in model.key.lower()):
            return False

        if model.description is None:

            # description is given, so this model cannot be matched
            if len(desc) > 0:
                return False

        else:

            # return False if the description is not matched
            if not(desc.lower() in model.description.lower()):
                return False

        # everything matches, return True
        return True

    return wrapper


def search(models, desc='', key='', disp=True):
    """Searches a list of models for the given text

    Parameters
    ----------
    models : list
        A list of Model objects to search through

    desc : str, optional
        If given, this (case insensitive) text is searched for in the README
        description of each model (Default: '')

    key : str, optional
        The model hashkeys (hexadecimal strings) are searched using this text
        (Default: '')

    Returns
    -------
    matches : list
        A list of matching model objects
    """
    matches = list(filter(matcher(desc, key), models))

    if disp:

        if len(matches) == 0:
            print('Did not find match any models!')

        for model in matches:
            hltext = '\033[94m{}\033[0m'
            keystr = model.key.replace(key, hltext.format(key)) if len(key) > 0 else model.key
            if model.description is None:
                descstr = ''
            else:
                descstr = model.description.replace(desc, hltext.format(desc)) if len(desc) > 0 else model.description
            print('{}: {}'.format(keystr, descstr))

    return matches


class Model:

    def __init__(self, directory, key):
        """Creates a Model object that interfaces with a folder
        in the deep-retina/database directory

        Parameters
        ----------
        directory : str
            filepath of the deep-retina/database folder

        key : str
            the name of the model folder in the database directory
        """
        self.basedir = directory
        self.key = key
        self.hashkey, self.modeltype = key.split(' ')

        # load files from the model directory
        self.metadata = load_json(self.filepath('metadata.json'))
        self.architecture = load_json(self.filepath('architecture.json'))
        self.experiment = load_json(self.filepath('experiment.json'))
        self.train = load_csv(self.filepath('train.csv'))
        self.validation = load_csv(self.filepath('validation.csv'))

        # load the README description
        try:
            with open(self.filepath('README.md'), 'r') as f:
                lines = f.readlines()
                desc_index = [line.strip() for line in lines].index('### description')
                self.description = lines[desc_index + 1].strip()
        except (FileNotFoundError, ValueError):
            self.description = None

        # force every model to have a timestamp
        if self.metadata is not None:
            self.timestamp = self.metadata['timestamp']
        else:
            self.timestamp = os.path.getmtime(self.filepath())

    def __str__(self):
        """Returns a string representation of this model"""

        if self.metadata is not None:
            datestr = 'Created on: {} {}'.format(self.metadata['date'], self.metadata['time'])
        else:
            datestr = ''

        return '\n'.join([self.key, self.description, datestr])

    def __repr__(self):
        return self.key

    def __iter__(self):
        """Iterate over the files in this directory"""
        return iter(os.listdir(self.filepath()))

    def filepath(self, *args):
        """Returns the filepath of the given key + arguments, if it exists"""
        return os.path.join(self.basedir, self.key, *args)

    @property
    def results(self):
        return load_h5(self.filepath('results.h5'))

    def bestiter(self, on='validation', metric='lli'):
        if self.results is not None:
            return np.array(self.results[on][metric]).mean(axis=1).argmax()

    def performance(self, idx, stimulus, on='test', metric='lli'):
        return np.array(self.results[on][stimulus][metric][idx])

    def weights(self, filename='best_weights.h5'):
        """Loads the given weights file from this model's directory"""

        # add file extension if necessary
        if not filename.endswith('.h5'):
            filename += '.h5'

        return load_h5(self.filepath(filename))

    def keras(self, weights='best_weights.h5'):
        """Returns a Keras model with the architecture and weights file"""

        if self.architecture is None:
            raise ValueError('Architecture not found. Is this a Keras model?')

        # Load model architecture
        mdl = model_from_config(self.architecture)

        # load the weights
        if weights is not None:
            mdl.load_weights(self.filepath(weights))

        return mdl

    def plot(self, filename='best_weights.h5'):
        """Plots the parameters of this model"""
        weights = self.weights(filename)

        if self.modeltype == 'convnet':
            figures = visualize_convnet(weights, self.architecture['layers'])

        elif self.modeltype.lower() == 'glm':
            figures = visualize_glm(weights)

        else:
            raise ValueError("I don't know how to plot a model of type '{}'".format(self.modeltype))

        return figures


def load_json(filepath):
    """Loads the given json file, if it exists, otherwise returns None"""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        return json.load(f)


def load_csv(filepath):
    """Loads the given csv file as a pandas DataFrame, if it exists, otherwise returns None"""
    if not os.path.exists(filepath):
        return None

    return pd.read_csv(filepath)


def load_h5(filepath):
    if not os.path.exists(filepath):
        return None

    return h5py.File(filepath, 'r')


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


def load_partial_model(model, stop_layer, start_layer=0):
    """
    Returns the model up to a specified layer.

    INPUT:
        model       a keras model
        layer_id    an integer designating which layer is the new final layer

    OUTPUT:
        a theano function representing the partial model
    """

    # create theano function to generate activations of desired layer
    start = model.layers[start_layer].input
    stop = model.layers[stop_layer].get_output(train=False)
    return theano.function([start], stop)


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
