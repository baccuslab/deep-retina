"""
Generate commonly used visual stimuli

Functions in this module either generate a numpy array
that encodes a particular stimulus (e.g. the `flash` function
generates a full-field flash, or the `contrast_steps` function
generates a sequence of contrast step changes), or they are used
for composing multiple stimulus sequences (concatenating them)
and converting them into a spatiotemporal stimulus (using rolling_window)
that can be fed to a Keras model (for example).
"""

from __future__ import absolute_import, division, print_function
from itertools import repeat
import numpy as np
from .experiments import rolling_window
from .utils import tuplify
from numbers import Number
from skimage.transform import downscale_local_mean
from skimage.filters import gaussian

__all__ = ['concat', 'white', 'contrast_steps', 'flash', 'spatialize', 'bar',
           'driftingbar', 'cmask', 'paired_flashes']


def concat(*args, nx=50, nh=40):
    """Returns a spatiotemporal stimulus that has been transformed using
    rolling_window given a list of stimuli to concatenate

    Parameters
    ----------
    stimuli : iterable
        A list or iterable of stimuli (numpy arrays). The first dimension
        is the sample, which can be different, but the rest of the
        dimensions (spatial dimensions) must be the same

    nh : int, optional
        Number of time steps in the rolling window history (default: 40)

    nx : int, optional
        Number of spatial dimensions (default: 50)
    """
    concatenated = np.vstack(map(lambda s: spatialize(s, nx), args)).astype('float32')
    return rolling_window(concatenated, nh)


def white(nt, nx=1, contrast=1.0):
    """Gaussian white noise with the given contrast

    Parameters
    ----------
    nt : int
        number of temporal samples

    nx : int
        number of spatial dimensions (default: 1)

    contrast : float
        Scalar multiplied by the whole stimulus (default: 1.0)
    """
    return contrast * np.random.randn(nt, nx, nx)


def contrast_steps(contrasts, lengths, nx=1):
    """Returns a random sequence with contrast step changes

    Parameters
    ----------
    contrasts : array_like
        List of the contrasts in the sequence

    lengths : int or array_like
        If an integer is given, each sequence has the same length.
        Otherwise, the given list is used as the lengths for each contrast

    nx : int
        Number of spatial dimensions (default: 1)
    """
    if isinstance(lengths, int):
        lengths = repeat(lengths)

    return np.vstack([white(nt, nx=nx, contrast=sigma)
                      for sigma, nt in zip(contrasts, lengths)])


def spatialize(array, nx):
    """Returns a spatiotemporal version of a full field stimulus

    Given an input array of shape (t, 1, 1), returns a new array with
    shape (t, nx, nx) where each temporal value is copied at each
    spatial location

    Parameters
    ----------
    array : array_like
        The full-field stimulus to spatialize

    nx : int
        The number of desired spatial dimensions (along one edge)
    """
    return np.broadcast_to(array, (array.shape[0], nx, nx))


def flash(duration, delay, nsamples, intensity=-1.):
    """Generates a 1D flash

    Parameters
    ----------
    duration : int
        The duration (in samples) of the flash

    delay : int
        The delay (in samples) before the flash starts

    nsamples : int
        The total number of samples in the array

    intensity : float or array_like, optional
        The flash intensity. If a number is given, the flash is a full-field
        flash. Otherwise, if it is a 2D array, then that image is flashed. (default: 1.0)
    """
    # generate the image to flash
    if isinstance(intensity, Number):
        img = intensity * np.ones((1, 1, 1))
    else:
        img = intensity.reshape(1, *intensity.shape)

    assert nsamples > (delay + duration), \
        "The total number samples must be greater than the delay + duration"
    sequence = np.zeros((nsamples,))
    sequence[delay:(delay + duration)] = 1.0
    return sequence.reshape(-1, 1, 1) * img


def bar(center, width, height, nx=50, intensity=-1., us_factor=1, blur=0.):
    """Generates a single frame of a bar"""

    # upscale factor (for interpolation between discrete bar locations)
    c0 = center[0] * us_factor
    c1 = center[1] * us_factor
    width *= us_factor
    height *= us_factor
    nx *= us_factor

    # center of the bar
    cx, cy = int(c0 + nx // 2), int(c1 + nx // 2)

    # x- and y- indices of the bar
    bx = slice(max(0, cx - width // 2), max(0, min(nx, cx + width // 2)))
    by = slice(max(0, cy - height // 2), max(0, min(nx, cy + height // 2)))

    # set the bar intensity values
    frame = np.zeros((nx, nx))
    frame[by, bx] = intensity

    # downsample the blurred image back to the original size
    return downsample(frame, us_factor, blur)


def downsample(img, factor, blur):
    """Smooth and downsample the image by the given factor"""
    return downscale_local_mean(gaussian(img, blur), (factor, factor))


def driftingbar(velocity, width, intensity=-1., x=(-30, 30)):
    """Drifting bar

    Usage
    -----
    >>> centers, stim = driftingbar(0.08, 2)

    Parameters
    ----------
    velocity : float
        bar velocity in pixels / frame (if negative, the bar reverses direction)

    width : int
        bar width in pixels

    Returns
    -------
    centers : array_like
        The center positions of the bar at each frame in the stimulus

    stim : array_like
        The spatiotemporal drifting bar movie
    """
    npts = 1 + int(x[1] - x[0] / np.abs(velocity))
    centers = np.sign(velocity) * np.linspace(x[0], x[1], npts)
    return centers, concat(np.stack(map(lambda x: bar((x, 0), width, np.Inf, us_factor=5, blur=2.), centers)))


def cmask(center, radius, array):
    """Generates a mask covering a central circular region"""
    a, b = center
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx-a, -b:ny-b]
    return x ** 2 + y ** 2 <= radius ** 2


def paired_flashes(ifi, duration, intensity, padding):
    """Example of a paired flash stimulus

    Parameters
    ----------
    ifi : int
        Inter-flash interval, in samples (number of samples between the end of the first
        flash and the beginning of the second)

    duration : int or tuple
        The duration (in samples) of each flash. If an int is given, then each flash has
        the same duration, otherwise, you can specify two different durations in a tuple

    intensity : float or tuple
        The intensity (luminance) of each flash. If a number is given, then each flash has
        the same intensity, otherwise, you can specify two different values in a tuple

    padding : int or tuple
        Padding (in samples) before the first and last flashes. If an int is given,
        then both the first and last pads have the same number of samples, otherwise,
        you can specify two different padding lengths in a tuple
    """
    # Convert numbers to tuples
    duration = tuplify(duration, 2)
    intensity = tuplify(intensity, 2)
    padding = tuplify(padding, 2)

    # generate the flashes
    f0 = flash(duration[0], padding[0], padding[0] + duration[0] + ifi, intensity[0])
    f1 = flash(duration[1], 0, duration[1] + padding[1], intensity[1])

    # return the concatenated pair
    return concat(f0, f1)


def square(halfperiod, nsamples, phase=0., intensity=1.0):
    """Generates a 1-D square wave"""
    assert 0 <= phase <= 1, "Phase must be a fraction between 0 and 1"

    # if halfperiod is zero, return all ones
    if halfperiod == 0:
        return np.ones(nsamples)

    # discretize the offset in terms of the period
    offset = int(2 * phase * halfperiod)

    # generate one period of the waveform
    waveform = np.stack(repeat(np.array([intensity, -intensity]), halfperiod)).T.ravel()

    # generate the repeated sequence
    repeats = int(np.ceil(nsamples / (2 * halfperiod)) + 1)
    sequence = np.hstack(repeat(waveform, repeats))

    # use the offset to specify the phase
    return sequence[offset:(nsamples + offset)]


def grating(barsize=(5, 0), phase=(0., 0.), nx=50, intensity=(1., 1.), us_factor=1, blur=0.):
    """Returns a grating as a spatial frame

    Parameters
    ----------
    barsize : (int, int), optional
        Size of the bar in the x- and y- dimensions. A size of 0 indicates no spatial
        variation along that dimension. Default: (5, 0)

    phase : (float, float), optional
        The phase of the grating in the x- and y- dimensions (as a fraction of the period).
        Must be between 0 and 1. Default: (0., 0.)

    intensity=(1., 1.)
        The contrast of the grating for the x- and y- dimensions

    nx : int, optional
        The number of pixels along each dimension of the stimulus (default: 50)

    us_factor : int
        Amount to upsample the image by (before downsampling back to 50x50), (default: 1)

    blur : float
        Amount of blur to applied to the upsampled image (before downsampling), (default: 0.)
    """
    # generate a square wave along each axis
    x = square(barsize[0], nx * us_factor, phase[0] % 1., intensity[0])
    y = square(barsize[1], nx * us_factor, phase[1] % 1, intensity[1])

    # generate the grating frame and downsample
    return downsample(np.outer(y, x), us_factor, blur)


def jittered_grating(nsamples, sigma=0.1, size=3):
    """Creates a grating that jitters over time according to a random walk"""
    phases = np.cumsum(sigma * np.random.randn(nsamples)) % 1.0
    frames = np.stack([grating(barsize=(size, 0), phase=(p, 0.)) for p in phases])
    return frames


def drifting_grating(nsamples, dt, barsize, us_factor=1, blur=0.):
    """Generates a drifting vertical grating

    Parameters
    ----------
    nsamples : int
        The total number of temporal samples

    dt : float
        The timestep of each sample. A smaller value of dt will generate a slower drift

    barsize : int
        The width of the bar in samples

    us_factor : int, optional
        Amount to upsample the image by (before downsampling back to 50x50), (default: 1)

    blur : float, optional
        Amount of blur to applied to the upsampled image (before downsampling), (default: 0.)
    """
    phases = np.mod(np.arange(nsamples) * dt, 1)
    return np.stack([grating(barsize=(barsize, 0),
                             phase=(phi, 0.),
                             us_factor=us_factor,
                             blur=blur) for phi in phases])


def reverse(img, halfperiod, nsamples):
    """Generates a temporally reversing stimulus using the given image

    Parameters
    ----------
    img : array_like
        A spatial image to reverse (e.g. a grating)

    halfperiod : int
        The number of frames each half period of the reversing image is shown for

    nsamples : int
        The total length of the stimulus in samples
    """
    return np.stack([t * img for t in square(halfperiod, nsamples)])


def get_grating_movie(grating_width=1, switch_every=10, movie_duration=100, mask=False,
                      intensity=1, phase=0, roll=True):
    '''
        Returns a reversing gratings stimulus.

        INPUT:
            grating_width   the width (in checkers) of each horizontal bar
            switch_every    how often (in frames) you want to reverse grating polarity
            movie_duration  number of frames in stimulus
            mask            either False or a np.array of shape (50,50); masks the gratings (i.e., over the receptive field)
            intensity       what is the contrast of the gratings?
            phase           what is the phase of the gratings?

        OUTPUT:
            full_movies     an np.array of shape (movie_duration, 40, 50, 50)
    '''

    # make grating
    grating_frame = -1 * np.ones((50, 50))
    for i in range(grating_width):
        grating_frame[:, (i + phase)::2 * grating_width] = 1
    if mask:
        grating_frame = grating_frame * mask * intensity
    else:
        grating_frame = grating_frame * intensity

    # make movie
    grating_movie = np.zeros((movie_duration, 50, 50))
    polarity_count = 0
    for frame in range(movie_duration):
        polarity_count += 1
        if int(polarity_count/switch_every) % 2 == 0:
            grating_movie[frame] = grating_frame
        else:
            grating_movie[frame] = -1 * grating_frame

    if roll:
        # roll movie axes to get the right shape
        full_movies = rolling_window(grating_movie, 40)
        return full_movies
    else:
        return grating_movie
