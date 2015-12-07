import numpy as np
from os.path import expanduser, join
import os
import json
import theano
import pyret.filtertools as ft
from preprocessing import datagen, loadexpt
from utils import rolling_window, mksavedir
import h5py
from scipy.stats import pearsonr
import preprocessing


# make save directory
save_dir = mksavedir(prefix='Convnet STC')

# Load stimulus
whitenoise_train_unrolled = loadexpt(0, 'whitenoise', 'train', 40, roll=False)


# Load CNN model spikes
path_to_spikes = expanduser('~/Dropbox/deep-retina/saved/lenna.salamander/2015-11-08 15.42.33 convnet/')
spikes_filename = 'model_predictions.h5'
g = h5py.File(os.path.join(path_to_spikes, spikes_filename), 'r')
model_predictions = g['predictions']

                            
# Load real times of stimulus and get time stamps of model spikes
f = h5py.File(os.path.join(preprocessing.datadirs['lenna'], '15-10-07/whitenoise.h5'), 'r')
time = np.array(f['train/time'][40:])

# Get model spike times
model_spikes = np.where(model_predictions > 2.39*np.std(model_predictions), 1, 0)
model_spike_times = time[model_spikes.flatten().astype('bool')]


# STA
sta, tax = ft.getsta(time, whitenoise_train_unrolled.X[40:], model_spike_times, 35)


# For dimensionality reduction, cut out window
Xcut = ft.cutout(whitenoise_train_unrolled.X[40:], idx=np.flipud(ft.filterpeak(sta)[1]), width=5)
sta_cut = ft.cutout(sta, idx=np.flipud(ft.filterpeak(sta)[1]), width=5)


# Do STC
stc = np.zeros((35*11*11, 35*11*11))
for idx, s in enumerate(ft.getste(time, Xcut, model_spike_times, 35)):
    sr = s.astype('float').ravel()
    if sr.size == (35*11*11):
        stc += np.outer(sr, sr)
        
    if idx % 500 == 0:
        print('{}'.format(100.*idx/len(spk)))


stc_normalized = stc/len(spk)
stc_normalized -= np.outer(sta_cut.ravel(), sta_cut.ravel())


## SAVE RESULT ##
h = h5py.File(join(save_dir, 'full_stc_convnet_15_10_07.h5'), 'w')
h.create_dataset('stc', data=stc_normalized)
h.close()




