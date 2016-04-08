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

__all__ = ['concat', 'white', 'contrast_steps', 'flash', 'spatialize', 'osr']


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

    intensity : float, optional
        The flash intensity (default: 1.0)
    """
    assert nsamples > (delay + duration), \
        "The total number samples must be greater than the delay + duration"
    sequence = np.zeros((nsamples,))
    sequence[delay:(delay + duration)] = intensity
    return sequence.reshape(-1, 1, 1)


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


def bar(center, width, height, nx=50, intensity=-1.):
    """Generates a single frame of a bar"""
    frame = np.zeros((nx, nx))

    # get the bar indices
    cx, cy = center[0] + 25, center[1] + 25
    bx = slice(max(0, cx - width // 2), max(0, min(nx, cx + width // 2)))
    by = slice(max(0, cy - height // 2), max(0, min(nx, cy + height // 2)))

    # set the bar intensity values
    frame[by, bx] = intensity
    return frame


def driftingbar(speed, width, intensity=-1.):
    """Drifting bar
    
    Usage
    -----
    >>> stim = driftingbar(0.1, 5)

    Parameters
    ----------
    speed : float
        bar speed in pixels / frame

    width : int
        bar width in pixels
    """
    xs = np.linspace(-50, 50, int(100 / speed)).astype('int')
    return xs, concat(np.stack(map(lambda x: bar((x, 0), width, 50), xs)))


def get_flash_sequence(initial_flash=45, latency=10, nsamples=100, intensity=1, flash_length=1):
    """Returns a 1 dimensional flash sequence."""
    flash_sequence = np.zeros((nsamples,))

    # Make two flashes
    for i in range(flash_length):
        flash_sequence[initial_flash+i] = intensity
    if latency < (nsamples - (initial_flash+flash_length)):
        for i in range(flash_length):
            flash_sequence[initial_flash+latency+i] = intensity
    return flash_sequence


def get_full_field_flashes(mask=np.ones((50, 50)), initial_flash=60, latency=10,
                           nsamples=100, intensity=1, flash_length=1):
    '''
        Returns full field flash stimulus.

        INPUT:
            mask            np.array of shape (50,50) that masks the flashes. Default is no mask.
            initial_flash   how many frames after stimulus start does the first flash occur?
            latency         how many frames after the initial flash does the second flash occur?
                            if greater than the length of stimulus, stimulus will have just the one
                            initial flash.
            nsamples        number of frames in the stimulus.
            intensity       how bright is the flash?
            flash_length    what is the duration, in samples, of the flash(es)?

        OUTPUT:
            full_field_movies   a np.array of shape (nsamples, 40, 50, 50)
    '''

    flash_sequence = get_flash_sequence(initial_flash=initial_flash, latency=latency,
                                        nsamples=nsamples, intensity=intensity,
                                        flash_length=flash_length)

    # Convert flash sequence into full field movie
    full_field_flash = np.outer(flash_sequence, mask)
    full_field_flash = full_field_flash.reshape((flash_sequence.shape[0], 50, 50))

    # Convert movie to 400ms long samples in the correct format for our model
    full_field_movies = rolling_window(full_field_flash, 40)
    full_field_movies = np.rollaxis(full_field_movies, 2)
    full_field_movies = np.rollaxis(full_field_movies, 3, 1)
    return full_field_movies


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
    grating_frame = -1*np.ones((50,50))
    for i in range(grating_width):
        grating_frame[:,(i+phase)::2*grating_width] = 1
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


def cmask(center, radius, array):
    a,b = center
    nx,ny = array.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= radius*radius
    return mask


def oms(duration=4, sample_rate=0.01, transition_duration=0.07, silent_duration=0.93,
        magnitude=5, space=(50,50), center=(25,25), object_radius=5, coherent=False, roll=False):
    '''
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
    '''
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
        fixed_world[:,i::2*grating_width] = 1

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


def motion_anticipation(duration=4, sample_rate=0.01, bar_speed=8, bar_width=3,
        space=(50,50), flash_pos=25, flash_dur=1, flash_time=1, mode='moving', roll=False):
    '''
        Object motion sensitivity stimulus, where an object moves differentially
        from the background.

        INPUT:
        duration        movie duration in seconds
        sample_rate     sample rate of movie in Hz
        mode            'moving' or 'flash'
        space           spatial dimensions
        bar_speed       speed of bar in pixels/sec; paper has 0.44mm/s -> 440microns/s -> 8.8pixels/s with 50microns pixels
        bar_width       width in pixels of bar
        flash_pos       spatial position of bar center when flashed
        flash_dur       duration in seconds of flash; will be centered in movie
        flash_time      time of flash start in seconds
        roll            whether to roll_axis for model prediction

        OUTPUT:
        movie           a numpy array of the stimulus
    '''
    # fixed params
    contrast = 1
    total_frames = int(duration/sample_rate)

    # determine bar position across time
    if mode == 'flash':
        # force left edge to always be >= 0
        leftedge = np.max(flash_pos - bar_width/2, 0)
        bar_pos = np.zeros((total_frames,)) #flash_pos * np.ones((total_frames,))

        flash_start = int(flash_time/sample_rate)
        flash_dur = int(flash_start/sample_rate)
        if flash_start > total_frames:
            print('Flash time later than duration!!!')

    elif mode == 'moving':
        bar_speed_frames = bar_speed * sample_rate
        bar_pos = np.linspace(0, bar_speed_frames*total_frames, total_frames)

        # start from leftmost edge
        leftedge = 0


    # make movie
    movie = np.zeros((total_frames, space[0], space[1]))
    for frame in range(total_frames):
        ## Draw Frame
        # draw background
        background = -1 * np.ones((space[0], space[1]))
        # start of leftedge on this frame
        bar_start = bar_pos[frame] + leftedge
        background[:,bar_start:bar_start+bar_width] = 1

        # adjust contrast
        background *= contrast
        if mode == 'flash':
            if frame > flash_start and frame < flash_start + flash_dur:
                movie[frame] = background
        else:
            movie[frame] = background

    if roll:
        # roll movie axes to get the right shape
        roll_movies = rolling_window(movie, 40)
        return roll_movies
    else:
        return movie
