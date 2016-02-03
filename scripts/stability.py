import numpy as np
import h5py
from os.path import expanduser
import pyret.spiketools as spktools
from deepretina.metrics import cc
from deepretina.visualizations import adjust_spines
import matplotlib.pyplot as plt

expt = raw_input("Please type in the experiment date (e.g. 15-10-07) 'Enter': ")
num_stim_types = raw_input("Please indicate how many stimulus types were in this experiment (e.g. 2 for whitenoise and naturalscene) 'Enter': ")
stim_idx = np.array([i in range(int(num_stim_types)) for i in range(4)]).astype('bool')
#experiments = ['15-10-07', '15-11-21a', '15-11-21b', '16-01-07', '16-01-08']
data_dir = expanduser('~/experiments/data/')
stim_types = np.array(['whitenoise.h5', 'naturalscene.h5', 'structured.h5', 'naturalmovie.h5'])
stim_types = stim_types[stim_idx]
#test_stim = 'whitenoise.h5'

#for expt in experiments:
all_ccs = []
all_frs = []
all_nrepeats = []
for test_stim in stim_types:
    full_path = data_dir + expt + '/'
    f = h5py.File(full_path + test_stim, 'r')
    bspk = np.array(f['train/response/binned'])
    tax = np.array(f['train/time'])
    ncells = bspk.shape[0]

    # estimate firing rate with 1 second Gaussian filter
    frs = np.zeros_like(bspk)
    for c in range(ncells):
        frs[c,:] = spktools.estfr(tax, bspk[c,:], 2.0)

    # estimate the cross-correlogram between trials
    nrepeats = np.array(f['test/repeats/cell01']).shape[0]
    correlations = np.zeros((ncells, nrepeats, nrepeats))
    for c in range(ncells):
        # need to rewrite new h5 files so cells are cell01 through cellnn
        if expt in ['15-10-07', '15-11-21a', '15-11-21b']:
            cell_label = 'cell%02i' %(c+1)
        else:
            cell_label = 'cell%02i' %(c)
        repeats = np.array(f['test/repeats/' + cell_label])
        for i in range(nrepeats):
            for j in range(i+1): #(i, nrepeats):
                correlations[c,i,j] = cc(repeats[i,:], repeats[j,:])

    all_ccs.append(correlations)
    all_frs.append(frs)
    all_nrepeats.append(nrepeats)


# generate plots
for c in range(ncells):
    # need to rewrite new h5 files so cells are cell01 through cellnn
    if expt in ['15-10-07', '15-11-21a', '15-11-21b']:
        cell_label = 'cell%02i' %(c+1)
    else:
        cell_label = 'cell%02i' %(c)

    cols = len(stim_types)
    rows = 2
    fig = plt.figure()
    #fig.set_size_inches((15,10))
    for i in range(rows):
        for j in range(cols):
            plt_idx = i*cols + j + 1
            ax = plt.subplot(rows, cols, plt_idx)

            # each row should either be correlogram or firing rate plot
            if i % 2 == 0:
                if i == 0:
                    plt.title(stim_types[j], fontsize=25)
                plt.imshow(all_ccs[j][c], cmap='Reds', clim=[0.0, 1.0], interpolation='nearest')
                plt.ylabel('Correlation across time', fontsize=20)
                ax.grid('off')
                #ax.axis('off')

                # plot actual values on white portion
                flipped_ccs = all_ccs[j][c].T
                for y_val in range(all_nrepeats[j]):
                    for x_val in range(y_val):
                        if x_val != y_val:
                            label = '%0.2f' %(flipped_ccs[x_val, y_val])
                            ax.text(y_val, x_val, label, va='center', ha='center') 

            else:
                plt.plot(tax/60.0, all_frs[j][c,:], 'k', linewidth=3)
                plt.xlabel('Time (min)', fontsize=20)
                plt.ylabel('Firing Rate (Hz)', fontsize=20)
                plt.ylim([0.0,10.0])
                plt.xlim([0.0,60.0])
                adjust_spines(ax, ['left', 'bottom'])
                #ax.spines['right'].set_visible(False)
                #ax.spines['top'].set_visible(False)
                #ax.yaxis.set_ticks_position('left')
                #ax.xaxis.set_ticks_position('bottom')
    plt.savefig(full_path + 'stability_' + cell_label + '.jpg')


f.close()
