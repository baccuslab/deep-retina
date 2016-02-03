import numpy as np
import h5py
from os.path import expanduser
import pyret.spiketools as spktools
import pyret.filtertools as ft
import pyret.visualizations as pyviz
import matplotlib.pyplot as plt
from scipy.stats import zscore

expt = raw_input("Please type in the experiment date (e.g. 15-10-07) 'Enter': ")
data_dir = expanduser('~/experiments/data/')
test_stim = 'whitenoise.h5'

full_path = data_dir + expt + '/' + test_stim
f = h5py.File(full_path, 'r')
bspk = np.array(f['train/response/binned'])
tax = np.array(f['train/time'])
ncells = bspk.shape[0]

# store stimulus
stimulus = np.array(f['train/stimulus'])
stimulus = zscore(stimulus)

# filter length for STA
filter_length = 40 # frames


for c in range(ncells):
    cell_label = 'cell%02i' %(c+1)
    print('Computing STA for cell %i of %i...' %(c+1, ncells))

    # compute sta
    sta = np.zeros((filter_length,stimulus.shape[1],stimulus.shape[2]))
    for idx, response in enumerate(bspk[c,:]):
        if idx >= filter_length:
            sta += response * stimulus[idx-filter_length:idx]

    # decompose STA
    space, time = ft.decompose(sta)

    # plot STA
    fig = plt.figure(figsize=(6,10))
    ax = pyviz.plotsta(np.linspace(0.0,filter_length*10.0, filter_length), sta, fig=fig)
    plt.savefig(data_dir + expt + '/' + 'cell%02i.jpg' %(c+1))
    plt.close()
