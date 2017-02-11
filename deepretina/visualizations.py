"""
Model visualization tools
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import pyret.filtertools as ft
import theano
import os
import h5py
from matplotlib import animation, gridspec
from scipy.interpolate import interp1d


def roc_curve(fpr, tpr, name='', auc=None, fmt='-', color='navy', ax=None):
    """Plots an ROC curve"""
    labelstr = '{} (AUC={:0.3f})'.format(name, auc) if auc is not None else name

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.plot([0., 1.], [0., 1.], '--', color='lightgrey', lw=4)

    if fmt == '-':
        x = np.linspace(0, 1, 1e3)
        f = interp1d(fpr, tpr, kind='nearest', fill_value='extrapolate')
        ax.plot(x, f(x), '-', color=color, label=labelstr)

    elif fmt == '.':
        ax.plot(fpr, tpr, '.', color=color, label=labelstr)

    plt.legend(loc=4, fancybox=True, frameon=True)
    ax.set_xlabel('False positive rate', fontsize=20)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylabel('True positive rate', fontsize=20)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    adjust_spines(ax)

    return ax


def gif(filename, array, fps=10, scale=1.0):
    """Creates a gif given a stack of images using moviepy

    Notes
    -----
    works with the bleeding edge version of moviepy (not the pip version)

    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)

    Parameters
    ----------
    filename : string
        The filename of the gif to write to

    array : array_like
        A numpy array that contains a sequence of images

    fps : int
        frames per second (default: 10)

    scale : float
        how much to rescale each image by (default: 1.0)
    """
    from moviepy.editor import ImageSequenceClip

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # broadcast along the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip


def response1D(x, r, dt=0.01, us_factor=50, figsize=(16, 10), name='Cell'):
    """Plots a response given a 1D (temporal) representation of the stimulus

    Parameters
    ----------
    x : array_like
        A 1-D representation of the stimulus

    x : array_like
        A (t, n) array of the response of n cells to the t time points in the stimulus

    dt : float
        The temporal sampling period, in seconds (default: 0.01)

    us_factor : int
        How much to upsample the stimulus by before plotting it (used to get a cleaner picture)

    figsize : tuple
        The figure dimensions. (default: (16, 10))
    """
    assert x.size == r.shape[0], "Dimensions do not agree"
    time = np.arange(x.size) * dt
    nrows = 8
    nspan = 6

    # upsample the stimulus
    time_us = np.linspace(0, time[-1], us_factor * time.size)
    x = interp1d(time, x, kind='nearest')(time_us)
    maxval = abs(x).max()
    maxrate = r.max()

    # helper function
    def mkplot(rate, title=''):
        fig = plt.figure(figsize=figsize)
        ax0 = plt.subplot2grid((nrows, 1), (0, 0))
        ax1 = plt.subplot2grid((nrows, 1), (nrows - nspan, 0), rowspan=nspan)

        # 1D stimulus trace
        ax0.fill_between(time_us, np.zeros_like(x), x, color='lightgrey', interpolate=True)
        ax0.plot(time_us, x, '-', color='gray')
        ax0.set_xlim(0, time_us[-1] + dt)
        ax0.set_yticks([-maxval, maxval])
        ax0.set_yticklabels([-maxval, maxval], fontsize=22)
        ax0.set_title(title)
        adjust_spines(ax0, spines=('left'))

        # neural responses
        ax1.plot(time, rate, '-', color='firebrick')
        ax1.set_xlim(0, time[-1] + dt)
        ax1.set_ylim(0, maxrate)
        ax1.set_ylabel('Firing rate (Hz)')
        ax1.set_xlabel('Time (s)')
        adjust_spines(ax1)

        return fig

    figures = list()
    figures.append(mkplot(r.mean(axis=1), title='{} population response'.format(name)))

    # loop over cells
    for ci in range(r.shape[1]):
        figures.append(mkplot(r[:, ci], title='{} {}'.format(name, ci + 1)))

    return figures


def plot_traces_grid(weights, tax=None, color='k', lw=3):
    """Plots the given array of 1D traces on a grid

    Parameters
    ----------
    weights : array_like
        Must have shape (num_rows, num_cols, num_dimensions)

    Returns
    -------
    fig : a matplotlib figure handle
    """

    # create the figure
    fig = plt.figure(figsize=(12, 8))

    # number of convolutional filters and number of affine filters
    nrows = weights.shape[0]
    ncols = weights.shape[1]
    ix = 1

    # keep everything on the same y-scale
    ylim = (weights.min(), weights.max())

    # time axis
    if tax is None:
        tax = np.arange(weights.shape[2])

    for row in range(nrows):
        for col in range(ncols):

            # add the subplot
            ax = fig.add_subplot(nrows, ncols, ix)
            ix += 1

            # plot the filter
            ax.plot(tax, weights[row, col], color=color, lw=lw)
            ax.set_ylim(ylim)
            ax.set_xlim((tax[0], tax[-1]))

            if row == col == 0:
                ax.set_xlabel('Time (s)')
            else:
                plt.grid('off')
                plt.axis('off')

    plt.show()
    plt.draw()
    return fig


def plot_filters(weights, cmap='seismic', normalize=True):
    """Plots an array of spatiotemporal filters

    Parameters
    ----------
    weights : array_like
        Must have shape (num_conv_filters, num_temporal, num_spatial, num_spatial)

    cmap : str, optional
        A matplotlib colormap (Default: 'seismic')

    normalize : boolean, optional
        Whether or not to scale the color limit according to the minimum and maximum
        values in the weights array

    Returns
    -------
    fig : a matplotlib figure handle
    """

    # create the figure
    fig = plt.figure(figsize=(12, 8))

    # number of convolutional filters
    num_filters = weights.shape[0]

    # get the number of rows and columns in the grid
    nrows, ncols = gridshape(num_filters, tol=2.0)

    # build the grid for all of the filters
    outer_grid = gridspec.GridSpec(nrows, ncols)

    # normalize to the maximum weight in the array
    if normalize:
        max_val = np.max(abs(weights.ravel()))
        vmin, vmax = -max_val, max_val
    else:
        vmin = np.min(weights.ravel())
        vmax = np.max(weights.ravel())

    # loop over each convolutional filter
    for w, og in zip(weights, outer_grid):

        # get the spatial and temporal frame
        spatial, temporal = ft.decompose(w)

        # build the gridspec (spatial and temporal subplots) for this filter
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=og, height_ratios=(4, 1), hspace=0.0)

        # plot the spatial frame
        ax = plt.Subplot(fig, inner_grid[0])
        ax.imshow(spatial, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.add_subplot(ax)
        plt.grid('off')
        plt.axis('off')

        ax = plt.Subplot(fig, inner_grid[1])
        ax.plot(temporal, 'k', lw=2)
        fig.add_subplot(ax)
        plt.grid('off')
        plt.axis('off')

    plt.show()
    plt.draw()
    return fig


def reshape_affine(weights, num_conv_filters):
    """Reshapes a layer of affine weights (loaded from a keras h5 file)

    Takes a weights array (from a keras affine layer) that has dimenions (S*S*C) x (A), and reshape
    it to be C x A x S x S, where C is the number of conv filters (given), A is the number of affine
    filters, and S is the number of spatial dimensions in each filter
    """
    num_spatial = np.sqrt(weights.shape[0] / num_conv_filters)
    assert np.mod(num_spatial, 1) == 0, 'Spatial dimensions are not square, check the num_conv_filters'

    newshape = (num_conv_filters, int(num_spatial), int(num_spatial), -1)
    return np.rollaxis(weights.reshape(*newshape), -1, 1)


def plot_spatial_grid(weights, cmap='seismic', normalize=True):
    """Plots the given array of spatial weights on a grid

    Parameters
    ----------
    weights : array_like
        Must have shape (num_conv_filters, num_affine_filters, num_spatial, num_spatial)

    cmap : str, optional
        A matplotlib colormap (Default: 'seismic')

    normalize : boolean, optional
        Whether or not to scale the color limit according to the minimum and maximum
        values in the weights array

    Returns
    -------
    fig : a matplotlib figure handle
    """

    # create the figure
    fig = plt.figure(figsize=(16, 12))

    # number of convolutional filters and number of affine filters
    nrows = weights.shape[0]
    ncols = weights.shape[1]
    ix = 1

    # normalize to the maximum weight in the array
    if normalize:
        max_val = np.max(abs(weights.ravel()))
        vmin, vmax = -max_val, max_val
    else:
        vmin = np.min(weights.ravel())
        vmax = np.max(weights.ravel())

    for row in range(nrows):
        for col in range(ncols):

            # add the subplot
            ax = fig.add_subplot(nrows, ncols, ix)
            ix += 1

            # plot the filter
            ax.imshow(weights[row, col], interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.grid('off')
            plt.axis('off')

    plt.show()
    plt.draw()
    return fig


def gridshape(n, tol=2.0):
    """Generates the dimensions of a grid containing n elements

    Parameters
    ----------
    n : int
        The number of elements that need to fit inside the grid

    tol : float
        The maximum allowable aspect ratio in the grid (e.g. a tolerance
        of 2 allows for a grid where one dimension is twice as long as
        the other). (Default: 2.0)

    Examples
    --------
    >>> gridshape(13)
    (3, 5)
    >>> gridshape(8, tol=2.0)
    (2, 4)
    >>> gridshape(8, tol=1.5)
    (3, 3)
    """

    # shortcut if n is small (fits on a single row)
    if n <= 5:
        return (1, n)

    def _largest_fact(n):
        k = np.floor(np.sqrt(n))
        for j in range(int(k), 0, -1):
            if np.mod(n, j) == 0:
                return j, int(n / j)

    for j in np.arange(n, np.ceil(np.sqrt(n)) ** 2 + 1):
        a, b = _largest_fact(j)
        if (b / a <= tol):
            return (a, b)


def visualize_convnet(h5file, layers, dt=1e-2):
    """Visualize the weights for a convnet given an hdf5 file handle"""

    # visualize each layer
    figures = list()
    for key, layer in zip(h5file.keys(), layers):

        if layer['name'] == 'Convolution2D':
            num_conv_filters = layer['nb_filter']
            fig = plot_filters(np.array(h5file[key]['param_0']))
            figures.append(fig)

        elif layer['name'] == 'Dense':
            weights = np.array(h5file[key]['param_0'])

            # affine layer after a convlayer
            try:
                W = reshape_affine(weights, num_conv_filters)
                fig = plot_spatial_grid(W)
                figures.append(fig)

            # arbitrary affine matrix
            except AssertionError:
                maxval = np.max(np.abs(weights.ravel()))
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(weights, cmap='seismic', vmin=-maxval, vmax=maxval)
                figures.append(fig)

    return figures


def visualize_glm(h5file):
    """Visualize the parameters of a GLM"""

    # plot the filters
    filters = np.array(h5file['filter'])
    fig1 = plot_filters(np.rollaxis(filters, 3))

    # plot the history / coupling traces
    history = np.array(h5file['history'])
    fig2 = plot_traces_grid(np.rollaxis(history, 0, 3))

    return [fig1, fig2]


def visualize_ln(h5file):
    """Visualize the parameters of an LN model"""

    # roll the number of cells to be the first dimension
    W = np.rollaxis(np.array(h5file['layer_1/param_0']), 1)

    # reshape the filters
    filtersize = np.sqrt(W.shape[1] / 40)
    filters = np.stack([w.reshape(40, filtersize, filtersize) for w in W])

    # make the plot
    fig = plot_filters(filters)
    return [fig]


def visualize_convnet_weights(weights, title='convnet', layer_name='layer_0',
        fig_dir=None, fig_size=(8,10), dpi=300, space=True, time=True, display=True,
        save=False, cmap='seismic', normalize=True):
    '''
    Visualize convolutional spatiotemporal filters in a convolutional neural
    network.

    Computes the spatial and temporal profiles by SVD.

    INPUTS:
    weights         weight array of shape (num_filters, history, space, space)
                        or full path to weight file
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

    if fig_dir is None:
        fig_dir = os.getcwd()

    # if user supplied path instead of array of weights
    if type(weights) is str:
        weight_file = h5py.File(weights, 'r')
        weights = weight_file[layer_name]['param_0']

    num_filters = weights.shape[0]

    if normalize:
        max_val = np.max(abs(weights[:]))
        colorlimit = [-max_val, max_val]

    # plot space and time profiles together
    if space and time:
        fig = plt.gcf()
        fig.set_size_inches(fig_size)
        plt.title(title, fontsize=20)
        num_rows = int(np.sqrt(num_filters))
        num_cols = int(np.ceil(num_filters/num_rows))
        idxs = range(num_cols)
        for x in range(num_cols):
            for y in range(num_rows):
                plt_idx = y * num_cols + x + 1
                # in case fewer weights than fit neatly in rows and cols
                if plt_idx <= len(weights):
                    spatial,temporal = ft.decompose(weights[plt_idx-1])
                    #plt.subplot(num_rows, num_cols, plt_idx)
                    ax = plt.subplot2grid((num_rows*4, num_cols), (4*y, x), rowspan=3)
                    if normalize:
                        ax.imshow(spatial, interpolation='nearest', cmap=cmap, clim=colorlimit)
                    else:
                        ax.imshow(spatial, interpolation='nearest', cmap=cmap)
                    plt.title('Subunit %i' %plt_idx)
                    plt.grid('off')
                    plt.axis('off')

                    ax = plt.subplot2grid((num_rows*4, num_cols), (4*y+3, x), rowspan=1)
                    ax.plot(np.linspace(0,len(temporal)*10,len(temporal)), temporal, 'k', linewidth=2)
                    plt.grid('off')
                    plt.axis('off')
        if save:
            plt.savefig(fig_dir + title + '_spatiotemporal_profiles.png', dpi=dpi)
            plt.close()
        if display:
            plt.show()

    # plot just spatial profile
    elif space and not time:
        fig = plt.gcf()
        fig.set_size_inches(fig_size)
        plt.title(title, fontsize=20)
        num_cols = int(np.sqrt(num_filters))
        num_rows = int(np.ceil(num_filters/num_cols))
        for x in range(num_cols):
            for y in range(num_rows):
                plt_idx = y * num_cols + x + 1
                # in case fewer weights than fit neatly in rows and cols
                if plt_idx <= len(weights):
                    spatial, temporal = ft.decompose(weights[plt_idx-1])
                    plt.subplot(num_rows, num_cols, plt_idx)
                    plt.imshow(spatial, interpolation='nearest', cmap=cmap, clim=colorlimit)
                    plt.grid('off')
                    plt.axis('off')
        if save:
            plt.savefig(fig_dir + title + '_spatiotemporal_profiles.png', dpi=dpi)
            plt.close()
        if display:
            plt.show()

    # plot just temporal profile
    elif time and not space:
        fig = plt.gcf()
        fig.set_size_inches(fig_size)
        plt.title(title, fontsize=20)
        num_cols = int(np.sqrt(num_filters))
        num_rows = int(np.ceil(num_filters/num_cols))
        for x in range(num_cols):
            for y in range(num_rows):
                plt_idx = y * num_cols + x + 1
                # in case fewer weights than fit neatly in rows and cols
                if plt_idx <= len(weights):
                    spatial, temporal = ft.decompose(weights[plt_idx-1])
                    plt.subplot(num_rows, num_cols, plt_idx)
                    plt.plot(np.linspace(0,len(temporal)*10,len(temporal)), temporal, 'k', linewidth=2)
                    plt.grid('off')
                    plt.axis('off')
        if save:
            plt.savefig(fig_dir + title + '_spatiotemporal_profiles.png', dpi=dpi)
            plt.close()
        if display:
            plt.show()

    # don't plot anything, just return spatial and temporal profiles
    else:
        spatial_profiles = []
        temporal_profiles = []
        for f in weights:
            spatial, temporal = ft.decompose(f)
            spatial_profiles.append(spatial)
            temporal_profiles.append(temporal)
        return spatial, temporal


def visualize_affine_weights(weights, num_conv_filters, layer_name='layer_4', title='affine units',
        fig_dir=None, fig_size=(8,10), dpi=300, display=True, save=False, cmap='seismic'):
    '''
    Visualize convolutional spatiotemporal filters in a convolutional neural
    network.

    Computes the spatial and temporal profiles by SVD.

    INPUTS:
    weights         weight array of shape (num_filters, history, space, space)
                        or full path to weight file
    title           title of plots; also the saved plot file base name
    fig_dir         where to save figures
    fig_size        figure size in inches
    dpi             resolution in dots per inch
    display         bool; display figure?
    save            bool; save figure?

    OUTPUT:
    saved figure or displayed figure (or both).
    '''

    if fig_dir is None:
        fig_dir = os.getcwd()

    # if user supplied path instead of array of weights
    if type(weights) is str:
        weight_file = h5py.File(weights, 'r')
        weights = weight_file[layer_name]['param_0']

    num_affine_units = weights.shape[1]
    spatial_size = np.sqrt(weights.shape[0]/num_conv_filters)
    assert weights.shape[0] % num_conv_filters == 0, 'Incorrect number of convolutional filters'

    # plot all filters on same color axis
    colorlimit = [-np.max(abs(weights[:])), np.max(abs(weights[:]))]

    # plot space and time profiles together
    fig = plt.gcf()
    fig.set_size_inches(fig_size)
    plt.title(title, fontsize=20)
    num_cols = int(num_conv_filters)
    num_rows = int(num_affine_units)

    # create one grid that we'll plot at the end
    G = np.zeros((num_affine_units * (1 + spatial_size), num_conv_filters * (1 + spatial_size)))

    idxs = range(num_cols)
    for y in range(num_rows):
        one_unit = weights[:,y].reshape((num_conv_filters, spatial_size, spatial_size))
        for x in range(num_cols):
            plt_idx = y * num_cols + x + 1
            G[y*(spatial_size)+y:(y+1)*spatial_size+y,x*(spatial_size)+x:(x+1)*spatial_size+x] = one_unit[x]

    plt.imshow(G, interpolation='nearest', cmap=cmap, clim=colorlimit)
    plt.grid('off')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('%d Units in Affine Layer' %(num_affine_units), fontsize=20)
    plt.xlabel('Weights per Convolutional Filter Type', fontsize=20)


    if save:
        plt.savefig(fig_dir + title + '_weights.png', dpi=dpi)
        plt.close()
    if display:
        plt.show()


# TO-DO:
# - function that checks if filters are low-rank
def singular_values(weights):
    """Returns singular values of 3D filters.
    Filters should be (time, space, space)
    """
    return ft.lowranksta(weights)[2]


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
def get_sta(model, layer_id, samples=50000, batch_size=50, print_every=None, subunit_id=None):
    '''
    White noise STA of an intermedate unit.
    If subunit_id is specified, it's a tuple of (x,y) locations for picking out one subunit in a conv layer
    '''
    # Get function for generating responses of intermediate unit.
    if subunit_id is not None:
        activations = theano.function([model.layers[0].input], model.layers[layer_id].get_output(train=False))
        def get_activations(stim):
            activity = activations(stim)
            # first dimension is batch size
            return activity[:, :, subunit_id[0], subunit_id[1]]
    else:
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
        whitenoise = np.random.randn(batch_size, 40, 50, 50).astype('float32')
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

        if print_every:
            if batch % print_every == 0:
                print('On batch %i of %i...' %(batch, samples/batch_size))

    sta /= samples
    #sta = sta.reshape((*(list(true_response_shape) + [-1])))
    #sta = sta.reshape((*true_response_shape, -1))

    # when the sta is of a conv layer
    if len(true_response_shape) == 3:
        sta = sta.reshape(true_response_shape[0], true_response_shape[1], true_response_shape[2], -1)
        return sta
    # when the sta is of an affine layer
    elif len(true_response_shape) == 1:
        sta = sta.reshape(true_response_shape[0], 40, 50, 50)
        return sta
    else:
        print('STA shape not recognized. Returning [sta, shape of response].')
        return [sta, true_response_shape]


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


def visualize_sta(sta, fig_size=(8, 10), display=True, save=False, normalize=True):
    '''
    Visualize one or many STAs of deep-retina interunits.

    Computes the spatial and temporal profiles by SVD.

    INPUTS:
    sta             weight array of shape (time, space, space)
                        or (num_units, time, space, space)
    fig_size        figure size in inches
    display         bool; display figure?
    save            bool; save figure?
    '''

    if len(sta) == 3:
        num_units = 1
    else:
        num_units = sta.shape[0]

    if normalize:
        colorlimit = [-np.max(abs(sta[:])), np.max(abs(sta[:]))]

    # plot space and time profiles together
    fig = plt.gcf()
    fig.set_size_inches(fig_size)
    plt.title('STA', fontsize=20)
    num_cols = int(np.sqrt(num_units))
    num_rows = int(np.ceil(num_units/num_cols))
    idxs = range(num_cols)
    for x in range(num_cols):
        for y in range(num_rows):
            plt_idx = y * num_cols + x + 1
            if num_units > 1:
                spatial,temporal = ft.decompose(sta[plt_idx-1])
            else:
                spatial,temporal = ft.decompose(sta)
            #plt.subplot(num_rows, num_cols, plt_idx)
            ax = plt.subplot2grid((num_rows*4, num_cols), (4*y, x), rowspan=3)
            if not normalize:
                ax.imshow(spatial, interpolation='nearest', cmap='seismic')
            else:
                ax.imshow(spatial, interpolation='nearest', cmap='seismic', clim=colorlimit)
            plt.grid('off')
            plt.axis('off')

            ax = plt.subplot2grid((num_rows*4, num_cols), (4*y+3, x), rowspan=1)
            ax.plot(np.linspace(0,len(temporal)*10,len(temporal)), temporal, 'k', linewidth=2)
            plt.grid('off')
            plt.axis('off')
    if save:
        plt.savefig(fig_dir + title + '.png', dpi=300)
        plt.close()
    if display:
        plt.show()


def adjust_spines(ax, spines=('left', 'bottom')):
    """Modify the spines of a matplotlib axes handle

    Usage
    -----
    >>> adjust_spines(plt.gca(), spines=['left', 'bottom'])
    """
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def animate_convnet_weights(weights, title='convnet', layer_name='layer_0',
        fig_dir=None, fig_size=(6,6), dpi=300, display=True,
        save=False, cmap='seismic', normalize=True):
    '''
    Visualize convolutional spatiotemporal filters in a convolutional neural
    network.

    Computes the spatial and temporal profiles by SVD.

    INPUTS:
    weights         weight array of shape (num_filters, history, space, space)
                        or full path to weight file
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

    if fig_dir is None:
        fig_dir = os.getcwd()

    # if user supplied path instead of array of weights
    if type(weights) is str:
        weight_file = h5py.File(weights, 'r')
        weights = weight_file[layer_name]['param_0']

    num_filters = weights.shape[0]

    if normalize:
        max_val = np.max(abs(weights[:]))
        colorlimit = [-max_val, max_val]

    # set up the figure
    fig = plt.gcf()
    fig.set_size_inches(fig_size)
    fig.suptitle(title + ' frame 0', fontsize=20)
    num_cols = int(np.sqrt(num_filters))
    num_rows = int(np.ceil(num_filters/num_cols))
    imgs = []
    filenames = []
    for x in range(num_cols):
        for y in range(num_rows):
            plt_idx = y * num_cols + x + 1
            # in case fewer weights than fit neatly in rows and cols
            if plt_idx <= len(weights):
                plt.subplot(num_rows, num_cols, plt_idx)
                img_i = plt.imshow(weights[plt_idx-1][0], interpolation='nearest', cmap=cmap, clim=colorlimit)
                imgs.append(img_i)
                plt.grid('off')
                plt.axis('off')


    def animate(i):
        fig.suptitle(title + ' frame %i' %i, fontsize=20)
        for x in range(num_cols):
            for y in range(num_rows):
                plt_idx = y * num_cols + x + 1
                # in case fewer weights than fit neatly in rows and cols
                if plt_idx <= len(weights):
                    plt.subplot(num_rows, num_cols, plt_idx)
                    imgs[plt_idx-1].set_data(weights[plt_idx-1][i])
                    #plt.imshow(weights[plt_idx-1][i], interpolation='nearest', cmap=cmap, clim=colorlimit)
                    plt.grid('off')
                    plt.axis('off')

        if save:
            name = fig_dir + title + ' frame %02i.png' %i
            name = name.replace(" ", "")
            plt.savefig(name)
            filenames.append(name)
            #plt.close()

        return imgs

    anim = animation.FuncAnimation(fig, animate, np.arange(weights.shape[1]),
            interval=100, repeat=True)


    if save:
        # Set up formatting for the movie files
        #Writer = animation.writers['ffmpeg'](fps=10)
        #anim.save(fig_dir + title + '_spatiotemporal_profiles.mp4', writer=Writer, dpi=dpi)
        #return anim

        # save each frame of animation as an image;
        # we're going to convert it to a gif later
        #filenames = []
        for i in range(weights.shape[1]):
            imgs = animate(i)
            #name = fig_dir + title + ' frame %02i.png' %i
            #name = name.replace(" ", "")
            #plt.savefig(name)
            #filenames.append(name)
            #plt.close()

        brackets = ' {} '
        all_brackets = weights.shape[1] * brackets
        gif_name = fig_dir + title + '.gif'
        gif_name = gif_name.replace(" ", "")
        system_command = 'convert' + all_brackets + gif_name
        system_command = system_command .format(*filenames)
        os.system(system_command)
        for f in filenames:
            remove_command = 'rm {}' .format(f)
            os.system(remove_command)

        plt.close()


    if display:
        plt.show()
        plt.draw()
