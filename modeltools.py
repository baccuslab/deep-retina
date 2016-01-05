import numpy as np
import theano
import h5py
import tableprint
from keras.models import model_from_json
from scipy.stats import pearsonr
from preprocessing import datagen, loadexpt

def load_model(model_path, weight_filename):
	''' Loads a Keras model using:
			- an architecture.json file
			- an h5 weight file, for instance 'epoch018_iter01300_weights.h5'
			
		INPUT:
			model_path		the full path to the saved weight and architecture files, ending in '/'
			weight_filename	an h5 file with the weights
        OUTPUT:
            returns keras model
	'''
	architecture_filename = 'architecture.json'
	architecture_data = open(model_path + architecture_filename, 'r')
	architecture_string = architecture_data.read()
	model = model_from_json(architecture_string)
	model.load_weights(model_path + weight_filename)
	
	return model


def load_partial_model(model, layer_id):
    '''
    Returns the model up to a specified layer.

    INPUT:
        model       a keras model
        layer_id    an integer designating which layer is the new final layer

    OUTPUT:
        a theano function representing the partial model
    '''

    # create theano function to generate activations of desired layer
    return theano.function([model.layers[0].input], model.layers[layer_id].get_output(train=False))


def list_layers(model_path, weight_filename):
    '''
    Lists the layers in the model with their children.
    
    This provides an easy way to see how many "layers" in the model there are, and which ones
    have weights attached to them.

    Layers without weights and biases are relu, pool, or flatten layers.

    INPUT:
			model_path		the full path to the saved weight and architecture files, ending in '/'
			weight_filename	an h5 file with the weights
    OUTPUT:
            an ASCII table using tableprint
    '''
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
            print(tableprint.row([l.encode('ascii','ignore'), params[-1][0].encode('ascii','ignore'),
                params[-1][1].encode('ascii','ignore')]))
        else:
            print(tableprint.row([l.encode('ascii','ignore'), '', '']))

    print(tableprint.hr(3))

def get_performance(model, stim_type='natural', cells=[0]):
    '''
    Get correlation coefficient on held-out data for deep-retina.
    '''
    if stim_type is 'natural':
        test_data = loadexpt(cells, 'naturalscene', 'test', 40)
    elif stim_type is 'white':
        test_data = loadexpt(cells, 'whitenoise', 'test', 40)

    truth = []
    predictions = []
    for X, y in datagen(50, *test_data):
        truth.extend(y)
        predictions.extend(model.predict(X))

    truth = np.array(truth)
    predictions = np.array(predictions)

    test_cc = []
    for c in cells:
        test_cc.append(pearsonr(truth[:,c], predictions[:,c])[0])

    return test_cc

def get_weights(path_to_weights, layer_name='layer_0'):
    '''
    A simple function to return the weights from a saved .h5 file.
    '''
    
    weight_file = h5py.File(path_to_weights, 'r')

    # param_0 stores the weights, param_1 stores biases
    weights = weight_file[layer_name]['param_0']
    return weights

