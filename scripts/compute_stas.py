"""
Computes STAs for all cells and stores them back in the h5 files

"""

from pyret.filtertools import getsta
from tqdm import tqdm
from deepretina.experiments import _loadexpt_h5
from deepretina.utils import notify
from scipy.stats import zscore 
import h5py
import numpy as np
import os


def get_filename(expt, name):
    return os.path.expanduser(os.path.join('~/experiments/data', expt, name + '.h5'))


def compute_stas(expt='all-cells'):
    """Computes STAs using the whitenoise.h5 file"""
    with h5py.File(get_filename(expt, 'whitenoise'), 'r') as f:
        stas = {}
        with notify('z-scoring the stimulus'):
            stim = zscore(np.array(f['train/stimulus']))
        tarr = np.array(f['train/time'])
        for key in tqdm(f['spikes']):
            spk = np.array(f['spikes'][key])
            sta, tax = getsta(tarr, stim, spk, 40)
            stas[key] = sta
    return stas


def save_stas(stas, filename, expt='all-cells'):
    """Saves the STAs to the h5 file"""
    with h5py.File(get_filename(expt, filename), 'r+') as f:
        for key in tqdm(f['spikes']):
            f['stas/{}'.format(key)] = stas[key] 


if __name__ == "__main__":
    # stasA = compute_stas(expt='15-11-21a')
    # stasB = compute_stas(expt='15-11-21b')
    # stasC = compute_stas(expt='15-10-07')
    stas = compute_stas(expt='all-cells')
    # save_stas(stas, 'whitenoise')
    # save_stas(stas, 'naturalscene')
