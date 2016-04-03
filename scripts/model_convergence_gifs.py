import numpy as np
import os
import csv
import json
from deepretina.visualizations import *

f = input("Please type in the path of the model you want to visualize converging and press 'Enter': ")

directory = os.path.expanduser(f)
files = os.listdir(directory)
weights = sorted([h for h in files if np.bitwise_and(h[:5] == 'epoch', h[-3:] == '.h5')])
count = 0
snapshots = []
for w in weights:
    weight_name = directory + '/' + w 

    # save rank 1 visualization of weights
    # this will be saved as plt_title + '_spatiotemporal_profiles.png'
    plt_title = '%04i_snapshot' %(count)
    fig_dir = directory + '/'
    visualize_convnet_weights(weight_name, title=plt_title, layer_name='layer_0',
            fig_dir=fig_dir, fig_size=(8,10), dpi=100, space=True, time=True, display=False,
            save=True, cmap='seismic', normalize=True)
    snapshots.append(directory + '/' + plt_title + '_spatiotemporal_profiles.png')

    count += 1
    print('%i / %i Done.' %(count, len(weights)))

brackets = ' {} '
all_brackets = len(snapshots) * brackets
gif_name = fig_dir + '/' + 'convergence' + '.gif'
gif_name = gif_name.replace(" ", "")
system_command = 'convert' + all_brackets + gif_name
system_command = system_command .format(*snapshots)
os.system(system_command)
for f in snapshots:
    remove_command = 'rm {}' .format(f)
    os.system(remove_command)


