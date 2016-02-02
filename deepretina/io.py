"""
Helper utilities for saving models and model outputs

"""

from __future__ import absolute_import, division, print_function
from os import mkdir, uname, getenv
from os.path import join, expanduser
from time import strftime

__all__ = ['mksavedir']


directories = {
    'weights': '~/deep-retina-saved-weights/',
    'dropbox': '~/Dropbox/deep-retina/saved/',
    'database': '~/deep-retina-results/',
}

# def initialize()


def mksavedir(basedir='~/Dropbox/deep-retina/saved', prefix=''):
    """
    Makes a new directory for saving models

    Parameters
    ----------
    basedir : string, optional
        Base directory to store model results in

    prefix : string, optional
        Prefix to add to the folder (name of the model or some other identifier)

    """

    assert type(prefix) is str, "prefix must be a string"

    # get the current date and time
    now = strftime("%Y-%m-%d %H.%M.%S") + " " + prefix

    # the save directory is the given base directory plus the current date/time
    userdir = uname()[1] + '.' + getenv('USER')
    savedir = join(expanduser(basedir), userdir, now)

    # create the directory
    mkdir(savedir)

    return savedir


def tomarkdown(filename, lines):
    """
    Write the given lines to a markdown file

    """

    # add .md to the filename if necessary
    if not filename.endswith('.md'):
        filename += '.md'

    with open(filename, 'a') as f:
        f.write('\n'.join(lines))
