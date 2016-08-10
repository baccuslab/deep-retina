"""
Toolbox with helpful utilities for exploring Keras models
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import theano
import h5py
import os
import sys
import re
import time
import tableprint
from keras.models import model_from_json, model_from_config
from .experiments import loadexpt, rolling_window
from .models import sequential
from . import metrics
from .visualizations import visualize_convnet, visualize_glm, visualize_ln
import pandas as pd
import json
from tqdm import tqdm
from operator import attrgetter
from itertools import takewhile
from .utils import xcorr, pairs
from scipy.stats import sem


def scandb(directory, nmax=-1):
    """Scans the given directory for any model folders and returns a
    list of Model objects for each folder in the directory
    """
    models = []

    # scan base directory for models folders, only load those with a README
    regex = re.compile(r'^([a-z0-9]){6} ')
    for folder in tqdm(list(filter(regex.match, os.listdir(directory)))[:nmax]):
        if os.path.isdir(os.path.join(directory, folder)):
            models.append(Model(directory, folder))

    return sorted(models, key=attrgetter('timestamp'), reverse=True)


def select(models, keys):
    """Selects the models with the given keys from the list of models"""
    return filter(lambda mdl: mdl.hashkey in keys, models)


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


def zoo(models, filename='modelzoo-{}-{}.csv', metric='fev'):
    """Generates a performance CSV file"""
    headers = 'key,type,exptdate,date,train_datasets,bestiter,train,validation,wn_test,ns_test'

    def row(mdl):
        try:
            data = ["'{}'".format(mdl.hashkey),
                    mdl.modeltype,
                    mdl.experiment['date'],
                    mdl.metadata['date'],
                    mdl.experiment['train_datasets'],
                    ]

            ix = mdl.bestiter(metric=metric)
            train = mdl.train[metric.upper()][ix]
            val = mdl.validation[metric.upper()][ix]

            wn_test = mdl.performance(ix, 'whitenoise', on='test', metric=metric).mean() \
                if 'whitenoise' in mdl.results['test'] else None

            ns_test = mdl.performance(ix, 'naturalscene', on='test', metric='fev').mean() \
                if 'naturalscene' in mdl.results['test'] else None

            data.extend(map(str, (ix, train, val, wn_test, ns_test)))
            return ','.join(data)
        except:
            # fuck
            return ','.join(['NaN']*10)

    skiplist = ('11795a', '65a38b', 'e4790c', '35b009', '3ccf99', '781506', 'd4ef12',
                '48dc18', '27998b', '4b3624', 'c37935')
    stop_at = '2fcc94'

    subselected = takewhile(lambda m: m.hashkey != stop_at, models)
    filtered = filter(lambda m: m.hashkey not in skiplist, subselected)
    rows = map(row, filtered)

    with open(filename.format(time.strftime('%Y-%m-%d'), metric), 'x') as f:
        f.write(headers + '\n')
        f.write('\n'.join(tqdm(rows)))


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

        if self.modeltype in ('convnet', 'multilayered_convnet'):
            figures = visualize_convnet(weights, self.architecture['layers'])

        elif self.modeltype.lower() == 'glm':
            figures = visualize_glm(weights)

        elif self.modeltype.lower() in ('ln_cutout', 'ln'):
            figures = visualize_ln(weights)

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


def modify_model(model_path, weight_filename, changed_params):
    """
    Loads a Keras model using:
    - an architecture.json file
    - an h5 weight file, for instance 'epoch018_iter01300_weights.h5'

    INPUT:
        model_path		the full path to the saved weight and architecture files, ending in '/'
        weight_filename	an h5 file with the weights
        changed_params  dictionary of new parameters.
                        e.g. {'loss': 'poisson', 'lr': 0.1, 'dropout': 0.25, 'name': 'Adam',
                        'layers': [{'layer_id': 0, 'trainable': false}]}
        OUTPUT:
        returns keras model
    """

    # if params have changed, load old json file, make changes, save revised json file
    with open(os.path.join(model_path, 'architecture.json'), 'r') as architecture_data:
        arch = json.load(architecture_data)
        for key in changed_params:
            # keys that are flat and at the highest hierarchy
            if key in ['loss', 'name', 'class_mode', 'sample_weight_mode']:
                arch[key] = changed_params[key]
            # keys that are in optimizer
            elif key in ['beta_1', 'beta_2', 'epsilon', 'lr', 'name']:
                arch['optimizer'][key] = changed_params[key]
            # key is dropout
            elif key in ['dropout']:
                idxs = [i for i in range(len(arch['layers'])) if arch['layers'][i]['name'] == 'Dropout']
                for i in idxs:
                    arch['layers'][i]['p'] = changed_params['dropout']
            # key is sigma for GaussianNoise layers
            elif key in ['sigma']:
                idxs = [i for i in range(len(arch['layers'])) if arch['layers'][i]['name'] == 'GaussianNoise']
                for count,i in enumerate(idxs):
                    arch['layers'][i]['sigma'] = changed_params['sigma'][count]
            # change parameters of individual layers
            elif key in ['layers']:
                # changed_params['layers'] should be a list of dicts
                for l in changed_params['layers']:
                    layer_id = l['layer_id']
                    for subkey in l.keys():
                        if subkey not in ['layer_id']:
                            arch['layers'][layer_id][subkey] = l[subkey]
            else:
                print('Key %s not recognized by load_model at this time.' % key)
                sys.stdout.flush()

        # write modified architecture to string
        architecture_string = json.dumps(arch)

    model = model_from_json(architecture_string)
    model.load_weights(os.path.join(model_path, weight_filename))

    return model


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


def load_partial_model(model, stop_layer=None, start_layer=0):
    """
    Returns the model up to a specified layer.

    INPUT:
        model           a keras model
        stop_layer      index of the final layer
        start_layer     index of the start layer

    OUTPUT:
        a theano function representing the partial model
    """
    if start_layer == 0:
        if stop_layer is None:
            return model.predict
        else:
            # create theano function to generate activations of desired layer
            start = model.layers[start_layer].input
            stop = model.layers[stop_layer].get_output(train=False)
            return theano.function([start], stop)
    else:
        # to have the partial model start at an arbitrary layer
        # we need to redefine the model
        layers = model.layers
        new_layers = layers[start_layer:stop_layer]
        new_model = sequential(new_layers, 'adam', loss='poisson')
        new_model.compile(optimizer='adam', loss='poisson')
        for idl, l in enumerate(new_model.layers):
            l.set_weights(new_layers[idl].get_weights())

        return new_model.predict


class CompositeModel(object):
    def __init__(self, model1, model2, history=None):
        """Initializes a model that is the composition of two existing models"""

        # get the function for each model
        self.func1 = model1.predict if hasattr(model1, 'predict') else model1
        self.func2 = model2.predict if hasattr(model2, 'predict') else model2

        # store the inner rolling window history
        self.history = history

    def predict(self, X):
        """Returns the prediction of the composition model2(model1(X))"""

        # pass through the first function
        intermediate = self.func1(X)

        # apply rolling window if necessary
        if self.history is not None:
            intermediate = rolling_window(intermediate, self.history)

        # return the output of the second function
        return self.func2(intermediate)


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
    print(tableprint.header(['layer', 'weights', 'biases']))

    params = []
    for l in layer_names:
        params.append(list(weights[l]))
        if params[-1]:
            print(tableprint.row([l.encode('ascii', 'ignore'),
                  params[-1][0].encode('ascii', 'ignore'),
                  params[-1][1].encode('ascii', 'ignore')]))
        else:
            print(tableprint.row([l.encode('ascii', 'ignore'), '', '']))

    print(tableprint.bottom(3))


def computecorr(data, maxlag, dt=1e-2):
    """Computes pairwise correlation

    Parameters
    ----------
    data : array_like
        data must have shape (ncells, nrepeats, ntimesteps)

    Returns
    -------
    lags : array_like
        array of time lags (in seconds)

    both : dict
        dictionary mapping from a tuple (pair) of indices, to
        an array containing the stimulus+noise correlations across
        N repeats for each of the M time lags

    stim : dict
        dictionary mapping from a tuple (pair) of indices, to
        an array containing the stimulus only correlations across
        (N)(N-1)/2 pairs of repeats for each of the M time lags
    """
    ncells, nrepeats, ntimesteps = data.shape

    # store results in a dictionary from pairs -> arrays
    both = {}
    stim = {}

    # compute lags array
    lags = np.arange(-maxlag, maxlag + 1).astype('float') * dt

    for pair in pairs(ncells):
        i, j = pair

        # compute stimulus+noise correlations
        both[pair] = np.stack([xcorr(data[i, r], data[j, r],
                                     maxlag, normalize=True)[1]
                               for r in range(nrepeats)])

        # compute stimulus only correlations for each unique pair of repeats
        stim[pair] = np.stack([xcorr(data[i, a], data[j, b],
                                     maxlag, normalize=True)[1]
                               for a, b in pairs(nrepeats)])

    return lags, both, stim


def noise_correlations(both, stim):
    """Computes noise correlations given stimulus+noise and
    just stimulus correlations

    Returns the difference in the means and the difference in
    the standard error of each pair in both
    """
    mu = dict()
    sigma = dict()

    for pair in both.keys():
        mu[pair] = np.mean(both[pair], axis=0) - np.mean(stim[pair], axis=0)
        sigma[pair] = sem(both[pair], axis=0) - sem(stim[pair], axis=0)

    return mu, sigma


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


def inject_noise(keras_model, noise_strength, stimulus, ntrials=10, target_layer=0):
    '''
        Inject noise into a keras model at a given layer to
        measure shared parameters and noise correlations
        in deep retina.

        INPUTS:
        keras_model         a keras model
        noise_strength      std of noise injection
        stimulus            visual stimulus signal to model; (time, 40, 50, 50)
        ntrials             number of trials to inject different noise instances
        target_layer        layer to inject noise into; default is pixels

        OUTPUT:
        out                 np array of model responses (response of ganglion cells)
                            with each row a different trial
    '''
    # split model into two parts
    if target_layer > 0:
        model_part1 = load_partial_model(keras_model, stop_layer=target_layer)
        model_part2 = load_partial_model(keras_model, start_layer=target_layer+1)

        stimulus_response = model_part1(stimulus)
    # unless one of the two parts is trivial
    else:
        model_part2 = keras_model.predict
        stimulus_response = stimulus

    noisy_responses = []
    for t in range(ntrials):
        noise = noise_strength * np.random.randn(*stimulus_response.shape)
        noisy_responses.append(model_part2(stimulus_response + noise))

    # return noise repeats as (ncells, nrepeats, ntimesteps)
    return np.rollaxis(np.stack(noisy_responses), 2, 0)
