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
import numpy as np
from .experiments import rolling_window
from itertools import repeat
from .utils import tuplify
from numbers import Number

__all__ = ['concat', 'white', 'contrast_steps', 'flash', 'spatialize', 'bar',
           'driftingbar', 'cmask', 'paired_flashes',
           'oms', 'osr', 'motion_anticipation']


def concat(*stimuli, nx=50, nh=40):
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
    concatenated = np.vstack(map(lambda s: spatialize(s, nx), stimuli))
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


def bar(center, width, height, nx=50, intensity=-1.):
    """Generates a single frame of a bar"""
    frame = np.zeros((nx, nx))

    # get the bar indices
    cx, cy = int(center[0] + 25), int(center[1] + 25)
    bx = slice(max(0, cx - width // 2), max(0, min(nx, cx + width // 2)))
    by = slice(max(0, cy - height // 2), max(0, min(nx, cy + height // 2)))

    # set the bar intensity values
    frame[by, bx] = intensity
    return frame


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
    xs = (np.sign(velocity) * np.linspace(x[0], x[1], npts)).astype('int')
    return xs, concat(np.stack(map(lambda x: bar((x, 0), width, np.Inf), xs)))


def cmask(center, radius, array):
    """Generates a mask covering a central circular region"""
    a, b = center
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx-a, -b:ny-b]
    return x ** 2 + y ** 2 <= radius ** 2


def paired_flashes(ifi=20, duration=5, intensity=-1., padding=50):
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
        full_movies = np.rollaxis(full_movies, 2)
        full_movies = np.rollaxis(full_movies, 3, 1)
        return full_movies
    else:
        return grating_movie


def oms(duration=4, sample_rate=0.01, transition_duration=0.07, silent_duration=0.93,
        magnitude=5, space=(50, 50), center=(25, 25), object_radius=5, coherent=False, roll=False):
    """
    Object motion sensitivity stimulus, where an object moves differentially
    from the background.

    INPUT:
    duration        movie duration in seconds
    sample_rate     sample rate of movie in Hz
    coherent        are object and background moving coherently?
    space           spatial dimensions
    center          location of object center
    object_width    width in pixels of object
    speed           speed of random drift
    motion_type     'periodic' or 'drift'
    roll            whether to roll_axis for model prediction

    OUTPUT:
    movie           a numpy array of the stimulus
    """
    # fixed params
    contrast = 1
    grating_width = 3

    transition_frames = int(transition_duration/sample_rate)
    silent_frames = int(silent_duration/sample_rate)
    total_frames = int(duration/sample_rate)

    # silence, one direction, silence, opposite direction
    obj_position = np.hstack([np.zeros((silent_frames,)), np.linspace(0, magnitude, transition_frames),
                            magnitude*np.ones((silent_frames,)), np.linspace(magnitude, 0, transition_frames)])

    half_silent = silent_frames/2
    back_position = np.hstack([obj_position[half_silent:], obj_position[:-half_silent]])

    # make position sequence last total_frames
    if len(back_position) > total_frames:
        print("Warning: movie won't be {} shorter than a full period.".format(np.float(2*transition_frames + 2*silent_frames)/total_frames))
        back_position[:total_frames]
        obj_position[:total_frames]
    else:
        reps = np.ceil(np.float(total_frames)/len(back_position))
        back_position = np.tile(back_position, reps)[:total_frames]
        obj_position = np.tile(obj_position, reps)[:total_frames]

    # create a larger fixed world of bars that we'll just crop from later
    padding = 2*grating_width + magnitude
    fixed_world = -1*np.ones((space[0], space[1]+padding))
    for i in range(grating_width):
        fixed_world[:, i::2 * grating_width] = 1

    # make movie
    movie = np.zeros((total_frames, space[0], space[1]))
    for frame in range(total_frames):
        # make background grating
        background_frame = np.copy(fixed_world[:,back_position[frame]:back_position[frame]+space[0]])

        if not coherent:
            # make object frame
            object_frame = np.copy(fixed_world[:,obj_position[frame]:obj_position[frame]+space[0]])

            # set center of background frame to object
            object_mask = cmask(center, object_radius, object_frame)
            background_frame[object_mask] = object_frame[object_mask]

        # adjust contrast
        background_frame *= contrast
        movie[frame] = background_frame

    if roll:
        # roll movie axes to get the right shape
        roll_movies = rolling_window(movie, 40)
        return roll_movies
    else:
        return movie


def osr(duration, interval, nflashes, intensity=-1.):
    """Omitted stimulus response

    Usage
    -----
    >>> stim = osr(2, 20, 5)

    Parameters
    ----------
    duration : float
        The duration of a flash, in samples

    frequency : int
        The inter-flash interval, in samples

    nflashes : int
        The number of flashes to repeat before the omitted flash
    """
    single_flash = flash(duration, interval, interval * 2, intensity=intensity)
    omitted_flash = flash(duration, interval, interval * 2, intensity=0.0)
    flash_group = list(repeat(single_flash, nflashes))
    zero_pad = np.zeros((interval, 1, 1))
    return concat(zero_pad, *flash_group, omitted_flash, *flash_group, nx=50, nh=40)


def motion_anticipation():
    """Generates the Berry motion anticipation stimulus

    Stimulus from the paper:
    Anticipation of moving stimuli by the retina,
    M. Berry, I. Brivanlou, T. Jordan and M. Meister, Nature 1999

    Returns
    -------
    motion : array_like
    flashes : array_like
    """
    # moving bar stimulus
    centers, stim = driftingbar(0.08, 2)

    # flashed bar stimulus
    images = (bar((x, 0), 2, 50) for x in np.unique(centers))
    flashes = [flash(2, 48, 100, intensity=im) for im in images]

    return centers[40:], stim, flashes
