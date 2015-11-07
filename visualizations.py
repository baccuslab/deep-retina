import numpy as np
import matplotlib.pyplot as plt
import pyret.filtertools as ft
import pyret.visualizations as viz
import json
import os
from keras.models import model_from_json

pwd = os.getcwd()

def visualize_convnet_filters(weights, title, fig_dir=pwd, fig_size=(16,10), 
        dpi=500, space=True, time=True):
    '''
    Visualize convolutional spatiotemporal filters in a convolutional neural 
    network.

    Computes the spatial and temporal profiles by SVD.

    Ouput are plots saved to fig_dir.
    '''
    
    num_filters = weights.shape[0]

    if space:
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


