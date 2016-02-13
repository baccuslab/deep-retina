import numpy as np
import h5py
from os.path import expanduser
import pyret.spiketools as spktools
import pyret.filtertools as ft
import pyret.visualizations as pyviz
import matplotlib.pyplot as plt
from scipy.stats import zscore
from matplotlib import gridspec

# Experiments to draw from
experiments = ['15-10-07', '15-11-21a', '15-11-21b', '16-01-07', '16-01-08']
data_dir = expanduser('~/experiments/data/')
test_stim = 'whitenoise.h5'

# STA parameters
# filter length for STA
filter_length = 40 # frames
all_temporal_filters = []

for expt in experiments:
    print('Computing STAs for experiment {}.'.format(expt))

    full_path = data_dir + expt + '/' + test_stim
    f = h5py.File(full_path, 'r')
    bspk = np.array(f['train/response/firing_rate_10ms'])
    tax = np.array(f['train/time'])
    ncells = bspk.shape[0]

    # store stimulus
    stimulus = np.array(f['train/stimulus'])
    stimulus = zscore(stimulus)


    for c in range(ncells):
        cell_label = 'cell%02i' %(c+1)
        #print('Computing STA for cell %i of %i...' %(c+1, ncells))

        # compute sta
        sta = np.zeros((filter_length,stimulus.shape[1],stimulus.shape[2]))
        for idx, response in enumerate(bspk[c,:]):
            if idx >= filter_length:
                sta += response * stimulus[idx-filter_length:idx]

        # decompose STA
        spatial_profile, temporal_filter = ft.decompose(sta)
        all_temporal_filters.append(temporal_filter)

    f.close()

# save all temporal kernels
h = h5py.File('all_temporal_kernels.h5', 'w')
h.create_dataset('kernels', data=np.vstack(all_temporal_filters))
h.close()

# plot all temporal kernels
fig = plt.figure(figsize=(6,10))
time = np.linspace(0.0, filter_length*10.0, filter_length)
#fig, ax = pyviz.plotsta(np.linspace(0.0,filter_length*10.0, filter_length), sta, fig=fig)

plt.plot(time, np.vstack(all_temporal_filters), alpha=0.4, linewidth=2, color='LightCoral', linestyle='-')
plt.xlabel('Time (ms)', fontsize=20)

plt.savefig('all temporal kernels.png', dpi=150)
plt.close()
