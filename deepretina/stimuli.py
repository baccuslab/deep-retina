"""
Generate commonly used visual stimuli

"""

from __future__ import absolute_import, division, print_function
import numpy as np
from .experiments import rolling_window
from scipy.signal import convolve2d


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


def cmask(center,radius,array):
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
        duration        movie duration in frames
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

    #import pdb
    #import matplotlib.pyplot as plt
    #pdb.set_trace()

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

        #import pdb
        #import matplotlib.pyplot as plt
        #pdb.set_trace()

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

        
    
    


        



