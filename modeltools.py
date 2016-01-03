import theano
import h5py
import tableprint
from keras.models import model_from_json

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
    Lists the layers in the model.
    '''
    weights = h5py.File(model_path + weight_filename, 'r')
    layer_names = list(weights)
    params = []
    for l in layer_names:
        params.append(list(weights[l]))

    print([layer_names, params])
    #tableprint.table([layer_names, params], ['layers', 'params'])

