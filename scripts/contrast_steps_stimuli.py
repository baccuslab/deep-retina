import numpy as np
from os.path import expanduser, join
import os
import json
import theano
import pyret.filtertools as ft
import pyret.visualizations as pyviz
from preprocessing import datagen, loadexpt
from utils import rolling_window
from keras.models import model_from_json
import h5py
import matplotlib.pyplot as plt
from utils import mksavedir

save_dir = mksavedir(prefix='Contrast Steps')

# GET LN AND CNN RESPONSES TO CONTRAST STEPS
architecture_filename = 'architecture.json'
naturalscenes_data_dir = expanduser('~/Dropbox/deep-retina/saved/lenna.salamander/2015-11-07 16.52.44 convnet/')
naturalscenes_weight_filename = 'epoch038_iter02700_weights.h5' # .53 cc on held-out
ln_data_dir = expanduser('~/Dropbox/deep-retina/saved/lenna.nirum/2015-11-08 04.41.18 LN/')
ln_weight_filename = 'epoch010_iter00750_weights.h5' # .468 cc on held-out

# LOAD NATURAL SCENES MODEL
naturalscenes_architecture_data = open(naturalscenes_data_dir + architecture_filename, 'r')
naturalscenes_architecture_string = naturalscenes_architecture_data.read()
naturalscenes_model = model_from_json(naturalscenes_architecture_string)
naturalscenes_model.load_weights(naturalscenes_data_dir + naturalscenes_weight_filename)

# LOAD LN MODEL
ln_architecture_data = open(ln_data_dir + architecture_filename, 'r')
ln_architecture_string = ln_architecture_data.read()
ln_model = model_from_json(ln_architecture_string)
ln_model.load_weights(ln_data_dir + ln_weight_filename)

def get_full_field_flicker(period=1, low_contrast=0.1, high_contrast=1.0):
    sample_rate = 100
    flicker_sequence = np.hstack([low_contrast*np.random.randn(period*sample_rate),
                               high_contrast*np.random.randn(period*sample_rate),
                              low_contrast*np.random.randn(period*sample_rate)])

    # Convert flicker sequence into full field movie
    full_field_flicker = np.outer(flicker_sequence, np.ones((1,50,50)))
    full_field_flicker = full_field_flicker.reshape((flicker_sequence.shape[0], 50, 50))

    # Convert movie to 400ms long samples in the correct format for our model
    full_field_movies = rolling_window(full_field_flicker, 40)
    full_field_movies = np.rollaxis(full_field_movies, 2)
    full_field_movies = np.rollaxis(full_field_movies, 3, 1)
    return full_field_movies

ntrials = 10000
period = 1
sample_rate = 100
batch_size = 26
nsamples = 3*period*sample_rate - 40
assert nsamples % batch_size == 0, 'nsamples must be divisible by batch_size'
cnn_responses = np.zeros((ntrials, nsamples))
ln_responses = np.zeros((ntrials, nsamples))
for n in range(ntrials):
    if n % 50 == 0:
        print 'Starting %d trial out of %d.' %(n,ntrials)
    stimulus = get_full_field_flicker()
    for batch in range(nsamples/batch_size):
        cnn_responses[n, batch*batch_size:(batch+1)*batch_size] = \
            naturalscenes_model.predict(stimulus[batch*batch_size:(batch+1)*batch_size])[:,0]
        ln_responses[n, batch*batch_size:(batch+1)*batch_size] = \
            ln_model.predict(stimulus[batch*batch_size:(batch+1)*batch_size])[:,0]

f = h5py.File(join(save_dir, 'responses_to_contrast_steps.h5'), 'w')
f.create_dataset('ln_responses', data=ln_responses)
f.create_dataset('cnn_responses', data=cnn_responses)
f.close()

fig = plt.gcf()
fig.set_size_inches((10,8))

# Plot multiple flicker sequences
for flicker in flicker_sequences:
    plt.plot(np.linspace(0,2.6,260), 14+flicker[40:], 'k')

# Plot average CNN and LN responses
plt.plot(np.linspace(0.0,2.6,260), average_cnn_response, 'b', linewidth=3)
plt.plot(np.linspace(0.0,2.6,260), average_ln_response, 'r', linewidth=3)
plt.xlabel('Time (seconds)', fontsize=20)
#plt.title('RED: LN Responses, BLUE: CNN Responses', fontsize=20)
plt.savefig(join(save_dir, 'Contrast Adaptation - white noise LN model - top.png'), dpi=300)


fig = plt.gcf()
fig.set_size_inches((5,8))

# Plot multiple flicker sequences
for flicker in flicker_sequences:
    plt.plot(np.linspace(0,2,200), 14+flicker[50+50:50+250], 'k')

# Plot average CNN and LN responses
plt.plot(np.linspace(0.0,2,200), average_cnn_response[10+50:10+250], 'b', linewidth=3)
plt.plot(np.linspace(0.0,2,200), average_ln_response[10+50:10+250], 'r', linewidth=3)
plt.xlabel('Time (seconds)', fontsize=20)
#plt.title('RED: LN Responses, BLUE: CNN Responses', fontsize=20)
plt.savefig(join(save_dir, 'Contrast Adaptation - white noise LN model - cropped - top.png'), dpi=300)

