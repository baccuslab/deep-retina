"""
Helper utilities for saving models and model outputs

"""

from __future__ import absolute_import, division, print_function
from os import mkdir, uname, getenv
from os.path import join, expanduser
from json import dumps
import shutil
import time
import theano
import keras
import deepretina
import hashlib

__all__ = ['Monitor']

directories = {
    'dropbox': expanduser('~/Dropbox/deep-retina/saved/'),
    'database': expanduser('~/deep-retina-results/database'),
}


class Monitor:

    def __init__(self, name, model, data):

        # pointer to Keras model and experimental data
        self.name = name
        self.model = model
        self.data = data
        self.rate = data.test.y

        # get the start time and date and some system information
        self.start = {
            'machine': uname()[1],
            'user': getenv('USER'),
            'timestamp': time.time(),
            'date': time.strftime("%Y-%m-%d"),
            'time': time.strftime("%H.%M.%S"),
        }

        # store version info
        self.versions = {
            'keras': keras.__version__,
            'deep-retina': deepretina.__version__,
            'theano': theano.__version__,
        }

        # build a hash for this model architecture
        tmp = hashlib.md5()
        tmp.update('\n'.join((self.start['date'],
                              self.start['time'],
                              self.model.to_yaml()).encode('ascii')))
        self.hashkey = tmp.hexdigest()[:6]

        # make the necessary folders
        self.datadir = ' '.join((self.hashkey, self.name))
        for _, directory in directories.items():
            mkdir(join(directory, self.datadir))

        # write files to disk
        self.write('architecture.json', self.model.to_json())
        self.write('architecture.yaml', self.model.to_yaml())
        self.write('version_info.json', dumps(self.versions))
        self.write('experiment.json', dumps(data.info))
        self.write('system.json', dumps(self.start))

    def save(self, epoch, iteration):
        pass

    def write(self, filename, text, copy=True):
        """Writes the given text to a file"""
        fullpath = join(directories['database'], self.datadir, filename)

        with open(fullpath) as f:
            f.write(text)

        # copy to dropbox
        if copy:
            shutil.copy(fullpath, directories['dropbox'])

    @property
    def rhat(self):
        return self.model.predict(self.data.test.X)

    # def test(self, epoch, iteration):

        # # performance on the entire holdout set
        # rhat_test = self.predict(self.holdout.X)
        # r_test = self.holdout.y

        # # performance on a subset of the training data
        # training_sample_size = rhat_test.shape[0]
        # inds = choice(self.training.y.shape[0], training_sample_size, replace=False)
        # rhat_train = self.predict(self.training.X[inds, ...])
        # r_train = self.training.y[inds]

        # # evalue using the given metrics  (computes an average over the different cells)
        # # ASSUMES TRAINING ON MULTIPLE CELLS
        # functions = map(multicell, (cc, lli, rmse))
        # results = [epoch, iteration]

        # for f in functions:
            # results.append(f(r_train, rhat_train)[0])
            # results.append(f(r_test, rhat_test)[0])

        # # save the results to a CSV file
        # self.save_csv(results)

        # # TODO: plot the train / test firing rates and save in a figure

        # return results

    # def save(self, epoch, iteration):
        # """
        # Save weights and optional test performance to directory

        # """

        # filename = join(self.weightsdir,
                        # "epoch{:03d}_iter{:05d}_weights.h5"
                        # .format(epoch, iteration))

        # # store the weights
        # self.model.save_weights(filename)
