from fuel.streams import AbstractDataStream
from fuel.iterator import DataIterator
from nems.utilities import rolling_window
import pyret.spiketools as spkt
import os

class RetinaStream(AbstractDataStream):
    """
    A fuel DataStream for retina data

    from fuel:
    A data stream is an iterable stream of examples/minibatches. It shares
    similarities with Python file handles return by the ``open`` method.
    Data streams can be closed using the :meth:`close` method and reset
    using :meth:`reset` (similar to ``f.seek(0)``).


    """

    def __init__(self, h5filename='retina_012314b.hdf5', datadir='~/experiments/data/012314b', cellidx=1, history=40, **kwargs):

        # ignore axis labels if not given
        kwargs.setdefault('axis_labels', '')

        # default to all of the data
        kwargs.setdefault('fraction', 1.0)

        # call __init__ of the AbstractDataStream
        super(RetinaStream, self).__init__(**kwargs)

        # load the h5py file
        self.datafile = h5py.File(os.path.join(os.path.expanduser(datadir), h5filename))
        self.cellidx = cellidx

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
        Y = rates[history:]

        # store references
        self.X = X
        self.Y = Y

        # shuffle indices, subselect a fraction
        inds = np.arange(self.Y.size)
        np.random.shuffle(inds)            # TODO: make this random selection reproducible
        self.inds = inds[:np.round(kwargs['fraction'] * float(inds.size))]
        self.current_index = 0

    def get_data():
        """Get a new sample of data"""

        # get this sample
        sample = (self.X[self.current_index,:,:,:], self.Y[self.current_index])

        # increment
        self.current_index += 1

        return sample

    def close():
        """Close the hdf5 file"""
        self.datafile.close()

    def reset():
        """Reset the current data index"""
        self.current_index = 0
