from fuel.streams import AbstractDataStream
from fuel.iterator import DataIterator
from nems.utilities import rolling_window
import pyret.spiketools as spkt
import os
import h5py
import numpy as np

class RetinaStream(AbstractDataStream):
    """
    A fuel DataStream for retina data

    from fuel:
    A data stream is an iterable stream of examples/minibatches. It shares
    similarities with Python file handles return by the ``open`` method.
    Data streams can be closed using the :meth:`close` method and reset
    using :meth:`reset` (similar to ``f.seek(0)``).


    """

    def __init__(self, h5filename='retina_012314b.hdf5', datadir='~/experiments/data/012314b', cellidx=1, history=40, partition_label='train', fraction=1.0, seed=0, **kwargs):

        # ignore axis labels if not given
        kwargs.setdefault('axis_labels', '')

        # call __init__ of the AbstractDataStream
        super(self.__class__, self).__init__(**kwargs)

        # load the h5py file
        self.datafile = h5py.File(os.path.join(os.path.expanduser(datadir), h5filename))
        self.cellidx = cellidx
        self.sources = ('data', 'rates')

        # pre-process
        stim = self.datafile['stimulus/sequence']
        timestamps = self.datafile['stimulus/timestamps']
        spk = self.datafile['spikes']

        # options / constants for spike binning & smoothing
        FRAMETIME = 0.01
        SPATIAL_SMOOTHING = 0.01

        # bin spikes
        bspk, tbins = spkt.binspikes(spk[('cell%i' % self.cellidx)],
                                     time=np.append(timestamps, timestamps[-1] + FRAMETIME))

        # compute firing rates
        rates = spkt.estfr(tbins, bspk, sigma=SPATIAL_SMOOTHING)

        # slice the stimulus
        stim_sliced = stim[34:-34, 34:-34, :]

        # roll the stimulus
        X = rolling_window(stim_sliced, history)
        X = np.rollaxis(X, 2)
        X = np.rollaxis(X, 3, 1)
        Y = rates[history:].astype('float32')

        #TODO: smooth rates 

        # store references
        self.X = X
        self.Y = np.expand_dims(Y, axis=1)

        # shuffle indices, subselect a fraction
        np.random.seed(seed=seed)
        inds = np.arange(self.Y.size)
        np.random.shuffle(inds)
        num_train = np.round(fraction * np.float32(inds.size))
        num_val   = np.round((np.float32(inds.size) - num_train)/2)
        num_test  = num_val

        if partition_label == 'train':
            self.inds = inds[:num_train]
        elif partition_label == 'val':
            self.inds = inds[num_train:num_train+num_val]
        elif partition_label == 'test':
            self.inds = inds[num_train+num_val:]
        self.current_index = 0

    def get_data(self, request=None):
        """Get a new sample of data"""

        if request is None:
            # get this sample
            sample = (self.X[self.current_index,:,:,:], self.Y[self.current_index])

            # increment
            self.current_index += 1
        else:
            sample = (self.X[request,:,:,:], self.Y[request])

        return sample

    def close(self):
        """Close the hdf5 file"""
        self.datafile.close()

    def reset(self):
        """Reset the current data index"""
        self.current_index = 0

    def get_epoch_iterator(self, **kwargs):
        return super(self.__class__, self).get_epoch_iterator(**kwargs)
    #    return None

    # TODO: implement iterator
    def next_epoch(self):
        return None
