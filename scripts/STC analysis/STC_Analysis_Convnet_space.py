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
from keras.models import model_from_json


# make save directory
save_dir = mksavedir(prefix='Convnet STC Long')

# Load stimulus
whitenoise_train_unrolled = loadexpt(0, 'whitenoise', 'train', history=0)


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
sta_cut_space, sta_cut_time = ft.decompose(sta_cut)


# load whitenoise model
architecture_filename = 'architecture.json'
whitenoise_data_dir = expanduser('~/Dropbox/deep-retina/saved/lenna.salamander/2015-11-08 15.42.33 convnet/')
whitenoise_weight_filename = 'epoch018_iter01300_weights.h5' # .63 cc on held-out
whitenoise_architecture_data = open(whitenoise_data_dir + architecture_filename, 'r')
whitenoise_architecture_string = whitenoise_architecture_data.read()
whitenoise_model = model_from_json(whitenoise_architecture_string)

# Do long STC
stc = np.zeros((11*11, 11*11))
n_samples = 1000000
batch_size = 100
for idx in range(n_samples/batch_size):
    stim = np.random.randint(0,2, size=(batch_size,40,50,50)).astype('float32')
    preds = whitenoise_model.predict(stim)
    for ids, st in enumerate(stim):
        s = ft.cutout(st[5:], idx=np.flipud(ft.filterpeak(sta)[1]), width=5)
        # project stimulus ensemble onto temporal sta kernel
        sr = np.inner(s.reshape((35,11*11)).T, sta_cut_time)
        if sr.size == (35*11*11):
            stc += preds[ids] * np.outer(sr, sr)

    if idx*batch_size % 500 == 0:
        print('{}'.format(100.*idx/n_samples))

stc_normalized = stc/n_samples
stc_normalized -= np.outer(sta_cut_space.ravel(), sta_cut_space.ravel())


## SAVE RESULT ##
h = h5py.File(join(save_dir, 'full_stc_convnet_long_15_10_07.h5'), 'w')
h.create_dataset('stc', data=stc_normalized)
h.close()




