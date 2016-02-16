"""
Generate commonly used visual stimuli

"""

from __future__ import absolute_import, division, print_function
import numpy as np
from .experiments import rolling_window


def get_contrast_changes(period=5, low_contrast=0.1, high_contrast=1.0, sample_rate=30, roll=True):
    """Probe contrast adaptation
        Returns full field flicker stimulus with low-high-low contrast.

        INPUT:
            period          number of seconds for each period of contrast
            low_contrast    0-1; how low is the low contrast step?
            high_contrast   0-1; how high is the high contrast step?
            sample_rate     int; how many frames per second is this stimulus?

        OUTPUT:
            full_field_movie    np array of shape (nframes, 40, 50, 50)
    """

    flicker_sequence = np.hstack([low_contrast*np.random.randn(period*sample_rate),
                                  high_contrast*np.random.randn(period*sample_rate),
                                  low_contrast*np.random.randn(period*sample_rate)])

    # Convert flicker sequence into full field movie
    full_field_flicker = np.outer(flicker_sequence, np.ones((1,50,50)))
    full_field_flicker = full_field_flicker.reshape((flicker_sequence.shape[0], 50, 50))

    if roll:
        # Convert movie to 400ms long samples in the correct format for our model
        full_field_movies = rolling_window(full_field_flicker, 40)
        full_field_movies = np.rollaxis(full_field_movies, 2)
        full_field_movies = np.rollaxis(full_field_movies, 3, 1)
        return full_field_movies
    else:
        return full_field_flicker

# Probe flash response
def get_flash_sequence(initial_flash=45, latency=10, nsamples=100, intensity=1, flash_length=1):
    '''
        Returns a 1 dimensional flash sequence.
    '''
    flash_sequence = np.zeros((nsamples,))

    # Make two flashes
    for i in range(flash_length):
        flash_sequence[initial_flash+i] = intensity
    if latency < (nsamples - (initial_flash+flash_length)):
        for i in range(flash_length):
            flash_sequence[initial_flash+latency+i] = intensity
    return flash_sequence

def get_full_field_flashes(mask=np.ones((50,50)), initial_flash=60, latency=10, nsamples=100, intensity=1,
                          flash_length=1):
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

    flash_sequence = get_flash_sequence(initial_flash=initial_flash, latency=latency, nsamples=nsamples,
                                       intensity=intensity, flash_length=flash_length)

    # Convert flash sequence into full field movie
    full_field_flash = np.outer(flash_sequence, mask)
    full_field_flash = full_field_flash.reshape((flash_sequence.shape[0], 50, 50))

    # Convert movie to 400ms long samples in the correct format for our model
    full_field_movies = rolling_window(full_field_flash, 40)
    full_field_movies = np.rollaxis(full_field_movies, 2)
    full_field_movies = np.rollaxis(full_field_movies, 3, 1)
    return full_field_movies

# Probe nonlinear subunits by reversing gratings
def get_grating_movie(grating_width=1, switch_every=10, movie_duration=100, mask=False, intensity=1, phase=0, roll=True):
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

def oms(duration=100, coherent=False, space=(50,50), center=(25,25), object_radius=5, speed=2, roll=True):
    '''
        Object motion sensitivity stimulus, where an object moves differentially
        from the background.

        INPUT:
        duration        movie duration in frames
        coherent        are object and background moving coherently?
        space           spatial dimensions
        center          location of object center
        object_width    width in pixels of object
        speed           speed of random drift
        roll            whether to roll_axis for model prediction

        OUTPUT:
        movie           a numpy array of the stimulus
    '''
    grating_width = 3
    contrast = 1

    # make movie
    movie = np.zeros((duration, 50, 50))
    
    # make random drift
    background_drift = speed*np.random.randn(duration)
    background_drift = background_drift.astype('int')
    if not coherent:
        object_drift = speed*np.random.randn(duration)
        object_drift = object_drift.astype('int')

    for frame in range(duration):
        # translate random walk into phase of bars on this frame
        background_phase = sum(background_drift[:frame+1]) % (2*grating_width)

        # make background grating
        background_frame = -1*np.ones(space)
        for i in range(grating_width):
            background_frame[:,(i+background_phase)::2*grating_width] = 1

        if not coherent:
            # object motion to phase
            object_phase = sum(object_drift[:frame+1]) % (2*grating_width)

            # make object frame
            object_frame = -1*np.ones(space)
            for i in range(grating_width):
                object_frame[:,(i+object_phase)::2*grating_width] = 1

            # set center of background frame to object
            object_mask = cmask(center, object_radius, object_frame)
            background_frame[object_mask] = object_frame[object_mask]
            
        # adjust contrast
        background_frame *= contrast
        movie[frame] = background_frame

    if roll:
        # roll movie axes to get the right shape
        roll_movies = rolling_window(movie, 40)
        roll_movies = np.rollaxis(roll_movies, 2)
        roll_movies = np.rollaxis(roll_movies, 3, 1)
        return roll_movies
    else:
        return movie 

def cmask(center,radius,array):
    a,b = center
    nx,ny = array.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= radius*radius
    return mask

