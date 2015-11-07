import numpy as np
import matplotlib.pyplot as plt
import pyret.filtertools as ft
import pyret.visualizations as viz
import json
import os
from keras.models import model_from_json

pwd = os.getcwd()

def visualize_convnet_filters(weights, title, fig_dir=pwd, fig_size=(8,10), 
        dpi=500, space=True, time=True, display=False, save=True):
    '''
    Visualize convolutional spatiotemporal filters in a convolutional neural 
    network.

    Computes the spatial and temporal profiles by SVD.

    INPUTS:
    weights         weight array of shape (num_filters, history, space, space)
    title           title of plots; also the saved plot file base name
    fig_dir         where to save figures
    fig_size        figure size in inches
    dpi             resolution in dots per inch
    space           bool; if True plots the spatial profiles of weights
    time            bool; if True plots the temporal profiles of weights
                    NOTE: if space and time are both False, function returns
                    spatial and temporal profiles instead of plotting
    display         bool; display figure?
    save            bool; save figure?
    
    OUTPUT:
    When space or time are true, ouputs are plots saved to fig_dir.
    When neither space nor time are true, output is:
        spatial_profiles        list of spatial profiles of filters
        temporal_profiles       list of temporal profiles of filters
    '''
    
    num_filters = weights.shape[0]

    # plot space and time profiles together
    if space and time:
        fig = plt.gcf()
        fig.set_size_inches(fig_size)
        plt.title(title, fontsize=20)
        num_cols = int(np.sqrt(num_filters))
        num_rows = int(np.ceil(num_filters/num_cols))
        idxs = range(num_cols)
        for x in range(num_cols):
            for y in range(num_rows):
                plt_idx = y * num_cols + x + 1
                spatial,temporal = ft.decompose(weights[plt_idx-1])
                #plt.subplot(num_rows, num_cols, plt_idx)
                ax = plt.subplot2grid((num_rows*4, num_cols), (4*y, x), rowspan=3)
                ax.imshow(spatial) #, clim=[np.min(W0), np.max(W0)])
                plt.grid('off')
                plt.axis('off')
                
                ax = plt.subplot2grid((num_rows*4, num_cols), (4*y+3, x), rowspan=1)
                ax.plot(np.linspace(0,400,40), temporal)
                plt.grid('off')
                plt.axis('off')
        if display:
            plt.show()
        if save:
            plt.savefig(fig_dir + title + '_spatiotemporal_profiles.png', dpi=dpi)

    # plot just spatial profile
    elif space and not time:
        fig = plt.gcf
        fig.set_size_inches(fig_size)
        plt.title(title, fontsize=20)
        num_cols = int(np.sqrt(num_filters))
        num_rows = int(np.ceil(num_filters/num_cols))
        for x in range(num_cols):
            for y in range(num_rows):
                plt_idx = y * num_cols + x + 1
                spatial, temporal = ft.decompose(weights[plt_idx-1])
                plt.subplot(num_rows, num_cols, plt_idx)
                plt.imshow(spatial, interpolation='nearest', colormap='grey')
                plt.colorbar()
                plt.grid('off')
                plt.axis('off')
        if display:
            plt.show()
        if save:
            plt.savefig(fig_dir + title + '_spatiotemporal_profiles.png', dpi=dpi)

    # plot just temporal profile
    elif time and not space:
        fig = plt.gcf
        fig.set_size_inches(fig_size)
        plt.title(title, fontsize=20)
        num_cols = int(np.sqrt(num_filters))
        num_rows = int(np.ceil(num_filters/num_cols))
        for x in range(num_cols):
            for y in range(num_rows):
                plt_idx = y * num_cols + x + 1
                spatial, temporal = ft.decompose(weights[plt_idx-1])
                plt.subplot(num_rows, num_cols, plt_idx)
                plt.plot(np.linspace(0,400,40), temporal)
                plt.grid('off')
                plt.axis('off')
        if display:
            plt.show()
        if save:
            plt.savefig(fig_dir + title + '_spatiotemporal_profiles.png', dpi=dpi)

    # don't plot anything, just return spatial and temporal profiles
    else:
        spatial_profiles = []
        temporal_profiles = []
        for f in weights:
            spatial, temporal = ft.decompose(f)
            spatial_profiles.append(spatial)
            temporal_profiles.append(temporal)
        return spatial, temporal
            
        







