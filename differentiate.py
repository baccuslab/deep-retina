import numpy as np
from os.path import expanduser
from keras.models import model_from_json
import h5py
from scipy.optimize import minimize
from numpy.linalg import norm
from utils import mksavedir

## FILENAMES AND DIRECTORIES ##
architecture_filename = 'architecture.json'
naturalscenes_data_dir = expanduser('~/Dropbox/deep-retina/saved/lenna.salamander/2015-11-07 16.52.44 convnet/')
naturalscenes_weight_filename = 'epoch038_iter02700_weights.h5' # .53 cc on held-out
ln_data_dir = expanduser('~/Dropbox/deep-retina/saved/lenna.nirum/2015-11-08 04.41.18 LN/')
ln_weight_filename = 'epoch010_iter00750_weights.h5' # .468 cc on held-out

## LOAD CONVNET TRAINED ON NATURAL SCENES ##
naturalscenes_architecture_data = open(naturalscenes_data_dir + architecture_filename, 'r')
naturalscenes_architecture_string = naturalscenes_architecture_data.read()
naturalscenes_model = model_from_json(naturalscenes_architecture_string)
naturalscenes_model.load_weights(naturalscenes_data_dir + naturalscenes_weight_filename)

## LOAD LN MODEL ##
ln_architecture_data = open(ln_data_dir + architecture_filename, 'r')
ln_architecture_string = ln_architecture_data.read()
ln_model = model_from_json(ln_architecture_string)
ln_model.load_weights(ln_data_dir + ln_weight_filename)

## DEFINE WHEN THE TWO MODELS ARE DIFFERENTIATED ##
def model_separation(stimulus):
    # reshape stimulus
    temporal_kernel = stimulus[:40] # length of 40
    spatial_profile = stimulus[40:] # length of 50*50
    low_rank_stimulus = np.outer(temporal_kernel, spatial_profile)
    stimulus = low_rank_stimulus.reshape((1,40,50,50))
    
    # get responses
    ln_response = ln_model.predict(stimulus)[0][0]
    convnet_response = naturalscenes_model.predict(stimulus)[0][0]
    return -(convnet_response - ln_response)**2

## DEFINE CONSTRAINT ##
constraint = {}
def unit_norm_constraint(stimulus):
    return (300. - norm(stimulus))**2
constraint['fun'] = unit_norm_constraint
constraint['type'] = 'eq'

## RUN MINIMIZATION ##
initial_guess = np.random.randn(40 + 50*50)
res_constrained = minimize(model_separation, x0=initial_guess, constraints=constraint)

## SAVE RESULT ##
save_dir = mksavedir(prefix='Maximal Differentiated Stimuli')
f = h5py.File('differentiated_stimuli.h5', 'w')
f.create_dataset('stimulus', data=res_constrained)
f.close()


