# script to center weights
import numpy as np
from os.path import expanduser
import os
import pyret.filtertools as ft
from keras.models import model_from_json
from deepretina.toolbox import load_model
import h5py

mdl_dir = os.path.expanduser('~/deep-retina-results/database/3520cd convnet/')
weight_name = 'best_weights.h5'
model = load_model(mdl_dir, weight_name)
weights = model.get_weights()

new_weights = np.copy(weights[0])

# for each subunit filter type
for idw, w in enumerate(new_weights):
    space, time = ft.decompose(w)
    peak, widths, theta = ft.get_ellipse(np.arange(space.shape[0]), np.arange(space.shape[1]), space)
    peak = np.round(np.array(peak)).astype(int)[::-1]
    peak2 = np.unravel_index(np.argmax(abs(space)), space.shape)
    center = np.round([s/2 for s in space.shape]).astype('int')
    centered_w = np.copy(w)
    for ax, shift in enumerate(peak):
        # roll array elements according to (array, shift, axis)
        centered_w = np.roll(centered_w, center[ax]-shift-1, ax+1)
    
    new_weights[idw] = centered_w

# save new weights in an h5 file
new_weight_name = 'centered_weights.h5'
copy_command = 'cp "%s" "%s"' %(mdl_dir + weight_name, mdl_dir + new_weight_name)
os.system(copy_command)


h = h5py.File(mdl_dir + new_weight_name, 'r+')
data = h['layer_0/param_0']
data[...] = new_weights
h.close()
