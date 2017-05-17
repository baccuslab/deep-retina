import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import csv
import json
from deepretina.visualizations import *

f = input("Please type in the path you want to start from and press 'Enter': ")

print('Starting to walk down directories')
# e.g. first run from os.path.expanduser('~/Dropbox/deep-retina/saved')
walker = os.walk(os.path.expanduser(f), topdown=True)
weight_name = 'best_weights.h5'
architecture_name = 'architecture.json'

#full_paths = []
#for dirs, subdirs, files in walker:
#    full_paths.append([dirs, subdirs, files])

models_parsed = 0
for dirs, subdirs, files in walker:
#for path in full_paths:
    #files = path[-1]
    #subdirs = subpaths[-2]
    #dirs = subpaths[-3]
    if (architecture_name in files) and (weight_name in files) and (('bn' in dirs) or ('dsrnn' in dirs)):
        with open(dirs + '/' + architecture_name, 'r') as f:
            arch = json.load(f)

        # save layers of model
        try:
            layers = arch['config']['layers']
        except:
            layers = []

        j = 1
        k = 1
        for idl, l in enumerate(layers):
            if l['class_name'] == 'Conv2D':
                num_filters = l['config']['filters']
                filter_shape = l['config']['kernel_size']
                plt_title = '%02i conv layer' %(idl)
                weights = dirs + '/' + weight_name
                layer_name = 'conv2d_%d/conv2d_%d' %(j,j)
                fig_dir = dirs + ' ' #+ '/'

                if j == 1:
                    try:
                        # rank 1 visualizations
                        visualize_convnet_weights(weights, title=plt_title, layer_name=layer_name,
                            fig_dir=fig_dir, fig_size=(8,10), dpi=100, space=True, time=True, display=False,
                            save=True, cmap='seismic', normalize=True)

                        # movies of weights
                        animate_convnet_weights(weights, title=plt_title, layer_name=layer_name,
                            fig_dir=fig_dir, fig_size=(6,6), dpi=100, display=False,
                            save=True, cmap='seismic', normalize=True)
                    except:
                        print('Failed to retrieve conv weights on %s' %dirs)
                else:
                    try:
                        # rank 1 visualizations
                        visualize_convnet_weights(weights, title=plt_title, layer_name=layer_name,
                            fig_dir=fig_dir, fig_size=(8,10), dpi=100, space=True, time=True, display=False,
                            save=True, cmap='seismic', normalize=True, lowrank=False)
                    except:
                        print('Failed to retrieve conv weights on %s' %dirs)
                j += 1


            elif l['class_name'] == 'Dense':
                output_dim = l['config']['units']
                plt_title = '%02i affine layer' %(idl)
                weights = dirs + '/' + weight_name
                layer_name = 'layer_%i' %(idl)
                fig_dir = dirs + ' ' #+ '/'

                try:
                    visualize_affine_weights(weights, num_filters, layer_name=layer_name, title=plt_title,
                        fig_dir=fig_dir, fig_size=(8,10), dpi=100, display=False, save=True, cmap='seismic')
                except:
                    print('Failed for {} model, {} filters, layer {}.'.format(dirs, num_filters, idl))

                num_filters = output_dim

            elif l['class_name'] == 'TimeDistributed':
                if l['config']['layer']['class_name'] == 'Conv2D':
                    l = l['config']['layer']
                    num_filters = l['config']['filters']
                    filter_shape = l['config']['kernel_size']
                    plt_title = '%02i conv layer' %(idl)
                    weights = dirs + '/' + weight_name
                    layer_name = 'time_distributed_%d/time_distributed_%d' %(j,j)
                    fig_dir = dirs + ' ' #+ '/'

                    if k == 1:
                        # rank 1 visualizations
                        visualize_convnet_weights(weights, title=plt_title, layer_name=layer_name,
                            fig_dir=fig_dir, fig_size=(8,10), dpi=100, space=True, time=True, display=False,
                            save=True, cmap='seismic', normalize=True)

                        # movies of weights
                        animate_convnet_weights(weights, title=plt_title, layer_name=layer_name,
                            fig_dir=fig_dir, fig_size=(6,6), dpi=100, display=False,
                            save=True, cmap='seismic', normalize=True)
                    else:
                        # rank 1 visualizations
                        visualize_convnet_weights(weights, title=plt_title, layer_name=layer_name,
                            fig_dir=fig_dir, fig_size=(8,10), dpi=100, space=True, time=True, display=False,
                            save=True, cmap='seismic', normalize=True, lowrank=False)
                    k += 1
                j += 1
        
        models_parsed += 1
        print('Visualized model %i' %(models_parsed))
