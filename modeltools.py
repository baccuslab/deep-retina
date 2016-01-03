from keras.models import model_from_json

def load_model(model_path, weight_filename):
	''' Loads a Keras model using:
			- an architecture.json file
			- an h5 weight file, for instance 'epoch018_iter01300_weights.h5'
			
		INPUT:
			model_path		the full path to the saved weight and architecture files
			weight_filename	an h5 file with the weights
	'''
	architecture_filename = 'architecture.json'
	architecture_data = open(model_path + architecture_filename, 'r')
	architecture_string = architecture_data.read()
	model = model_from_json(architecture_string)
	model.load_weights(model_path + weight_filename)
	
	return model
