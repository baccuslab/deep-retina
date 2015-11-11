import numpy as np
from os.path import expanduser, join
from keras.models import model_from_json
import h5py
from scipy.optimize import minimize
from numpy.linalg import norm
from utils import mksavedir
from scipy.stats import zscore

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
    full_stimulus = np.outer(stimulus[:40], stimulus[40:])
    return norm(full_stimulus) - 1.0/np.sqrt(np.prod(full_stimulus.shape[:]))
constraint['fun'] = unit_norm_constraint
constraint['type'] = 'ineq'

## RUN MINIMIZATION ##
initial_guess = zscore(np.random.randint(0, 2, 40 + 50*50).astype('float32'))
res_constrained = minimize(model_separation, x0=initial_guess, constraints=constraint, method='COBYLA')

optimal_stimulus = res_constrained.x
constraint_violation = unit_norm_constraint(optimal_stimulus)

temporal_kernel = optimal_stimulus[:40]
spatial_profile = optimal_stimulus[40:]
low_rank_optimal_stimulus = np.outer(temporal_kernel, spatial_profile)
optimal_stimulus = low_rank_optimal_stimulus.reshape((1,40,50,50))

ln_response = ln_model.predict(optimal_stimulus)[0][0]
convnet_response = naturalscenes_model.predict(optimal_stimulus)[0][0]
responses = np.array([ln_response, convnet_response])

## SAVE RESULT ##
save_dir = mksavedir(prefix='Maximal Differentiated Stimuli')
f = h5py.File(join(save_dir, 'differentiated_stimuli.h5'), 'w')
f.create_dataset('stimulus', data=optimal_stimulus)
f.create_dataset('responses', data=responses)
f.create_dataset('constraint', data=constraint_violation)
f.close()


