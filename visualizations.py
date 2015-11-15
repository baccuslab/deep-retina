import numpy as np
import matplotlib.pyplot as plt
import pyret.filtertools as ft
import pyret.visualizations as viz
import theano
import json
import os
from keras.models import model_from_json
from deepretina.preprocessing import datagen, loadexpt

pwd = os.getcwd()

def visualize_convnet_weights(weights, title='convnet', fig_dir=pwd, 
        fig_size=(8,10), dpi=500, space=True, time=True, display=False,
        save=True):
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
                ax.imshow(spatial, interpolation='nearest', cmap='gray') #, clim=[np.min(W0), np.max(W0)])
                plt.grid('off')
                plt.axis('off')
                
                ax = plt.subplot2grid((num_rows*4, num_cols), (4*y+3, x), rowspan=1)
                ax.plot(np.linspace(0,400,40), temporal, 'k', linewidth=2)
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
                plt.imshow(spatial, interpolation='nearest', cmap='gray')
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
                plt.plot(np.linspace(0,400,40), temporal, 'k', linewidth=2)
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
            
        


def visualize_affine_weights(weights, num_conv_filters, title='affine', fig_dir=pwd, 
        fig_size=(8,10), dpi=500, display=False, save=True):
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
    display         bool; display figure?
    save            bool; save figure?
    
    OUTPUT:
    saved figure or displayed figure (or both).
    '''
    
    num_affine_units = weights.shape[1]
    spatial_size = np.sqrt(weights.shape[0]/num_conv_filters)
    assert weights.shape[0] % num_conv_filters == 0, 'Incorrect number of convolutional filters'

    # plot space and time profiles together
    fig = plt.gcf()
    fig.set_size_inches(fig_size)
    plt.title(title, fontsize=20)
    num_cols = int(num_conv_filters)
    num_rows = int(num_affine_units)
    idxs = range(num_cols)
    for y in range(num_rows):
        one_unit = weights[:,y].reshape((num_conv_filters, spatial_size, spatial_size))
        colorlimit = [np.min(one_unit), np.max(one_unit)]
        for x in range(num_cols):
            plt_idx = y * num_cols + x + 1
            plt.subplot(num_rows, num_cols, plt_idx)
            ax = plt.imshow(one_unit[x], clim=colorlimit, interpolation='nearest', cmap='gray')
            plt.grid('off')
            plt.xticks([])
            plt.yticks([])

            if x == 0:
                if y == int(num_rows/2):
                    plt.ylabel('%d Units in Affine Layer' %(num_affine_units), fontsize=20)
            if y == num_rows-1:
                if x == 0:
                    plt.xlabel('Weights per Convolutional Filter Type', fontsize=20)

    if display:
        plt.show()
    if save:
        plt.savefig(fig_dir + title + '_weights.png', dpi=dpi)


# TO-DO:
# - function that checks if filters are low-rank
def singular_values(weights):
    '''Returns singular values of 3D filters.
    Filters should be (time, space, space)
    '''
    fk, u, s, v = ft.lowranksta(weights)
    return s

# - function that plots distribution of linear projections on threshold
def activations(model, layer_id, stimulus):
    '''
    Returns the activations of a specified layer.
    '''
    # create theano function to generate activations of desired layer
    get_activations = theano.function([model.layers[0].input], model.layers[layer_id].get_output(train=False))
    
    # get intermediate unit response to stimulus
    response = get_activations(stimulus)
    return response

def response_before_threshold(weights, model, layer_id, stimulus):
    '''
    Get the activity of a layer before thresholding. For instance
    could be useful to see where the effective threshold is for each
    unit.

    INPUT:
    weights should be dict containing filter weights and biases
    model is keras model
    layer_id is integer referring to the particular layer you want the response from
    stimulus is of size (samples, history, size, size)

    OUTPUT:
    list of responses; length of list is equal to number of units
    '''
    filters = weights['param_0']
    biases = weights['param_1']
    if layer_id == 0:
        flat_stim = stimulus.reshape((stimulus.shape[0], -1))
        flat_filters = [np.reshape(filt, -1) for filt in filters]
        responses = [np.dot(flat_stim, flat_filt) + biases[idx] for idx, flat_filt in enumerate(flat_filters)]
        return responses
    else:
        prequel_response = activations(model, layer_id-1, stimulus)
        flat_stim = prequel_response.reshape((prequel_response.shape[0], -1))
        flat_filters = [np.reshape(filt, -1) for filt in filters]
        responses = [np.dot(flat_stim, flat_filt) + biases[idx] for idx, flat_filt in enumerate(flat_filters)]
        return responses




# function that plots the receptive field of the interneurons (i.e. conv or affine layer activations)
def get_sta(model, layer_id, samples=50000, batch_size=50):
    '''
    White noise STA of an intermeidate unit.
    '''
    # Get function for generating responses of intermediate unit.
    get_activations = theano.function([model.layers[0].input], model.layers[layer_id].get_output(train=False))

    impulse = np.random.randn(2, 40, 50, 50).astype('uint8')
    impulse_response = get_activations(impulse)
    impulse_response_flat = impulse_response.reshape(2, -1).T
    impulse_flat = impulse.reshape(2, -1)
    #num_filter_types = impulse_response.shape[1]
    sta = np.zeros_like(np.dot(impulse_response_flat, impulse_flat))

    # Initialize STA
    #stas = [np.zeros((40, 50, 50), dtype='float') for _ in range(num_stas)]
    stas = {}

    # Generate white noise and map STA
    for batch in range(int(np.ceil(samples/batch_size))):
        whitenoise = np.random.randn(batch_size, 40, 50, 50)
        response = get_activations(whitenoise)
        true_response_shape = response.shape[1:]

        response_flat = response.reshape(batch_size, -1).T
        whitenoise_flat = whitenoise.reshape(batch_size, -1)
        # sta will be matrix of units x sta dims
        sta += np.dot(response_flat, whitenoise_flat)
        #sta = sta.reshape((*true_response_shape, -1))

        #for dim in true_response_shape:


        #for filt_type in range(num_stas):
        #    nonzero_inds = np.where(response

        #nonzero_inds = np.where(response > 0)[0]
        #for idx in nonzero_inds:
        #    sta += response[idx] * whitenoise[idx]
    
    sta /= samples
    sta = sta.reshape((*true_response_shape, -1))
    return sta
    


# a useful visualization of intermediate units may be its STC
def get_stc(stimulus, response):
    """
    Compute the spike-triggered covariance

    Returns
    -------
    stc : ndarray
        The spike-triggered covariance (STC) matrix

    """
    sta = get_sta(stimulus, response)
    flat_sta = sta.ravel()

    nonzero_inds = np.where(response > 0)[0]
    
    # initialize stc
    sta = np.empty(stimulus[0].shape, dtype='float')
    
    # loop over nonzero responses
    for idx in nonzero_inds:
        sta += response[idx] * sample[idx]
    sta /= len(nonzero_inds)
    return sta


    
    # get the blas function for computing the outer product
    assert stimulus.dtype == 'float64', 'Stimulus must be double precision'
    outer = get_blas_funcs('syr', dtype='d')

    # get the iterator
    ste = getste(time, stimulus, spikes, filter_length)

    # reduce, note that this only contains the upper triangular portion
    try:
        first_slice = next(ste)
        stc_init = np.triu(np.outer(first_slice.ravel(), first_slice.ravel()))
        stc_ut = reduce(lambda C, x: outer(1, x.ravel(), a=C),
                        ste, stc_init) / float(len(spikes))
    except StopIteration:
        ndims = np.prod(stimulus.shape[1:]) * filter_length
        return np.nan * np.ones((ndims, ndims))

    # make the full STC matrix (copy the upper triangular portion to the lower
    # triangle)
    stc = np.triu(stc_ut, 1).T + stc_ut

    # compute the STA (to subtract it)
    sta = getsta(time, stimulus, spikes, filter_length)[0].ravel()

    return stc - np.outer(sta, sta)

