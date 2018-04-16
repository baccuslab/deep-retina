import os
import tableprint as tp
import argparse
from deepretina import experiments
from collections import defaultdict
from deepretina.metrics import np_wrap, cc, rmse, fev

import tensorflow as tf
import sonnet as snt

import deepdish as dd

import numpy as np

from tqdm import tqdm

from collections import namedtuple


Graph = namedtuple('graph', ('stim', 'hist', 'rate'))


class GLM(snt.AbstractModule):
    def __init__(self, nln=tf.nn.softplus, l2_filter=0., l2_hist=0., name='glm'):
        self.nln = nln
        self.l2_filter = l2_filter
        self.l2_hist = l2_hist
        super().__init__(name=name)

    def _build(self, stim, hist):

        init = {'w': tf.zeros_initializer()}
        reg = {'w': tf.contrib.layers.l2_regularizer(self.l2_filter)}
        stim_ravel = snt.BatchFlatten()(stim)
        u_stim = snt.Linear(1, initializers=init, regularizers=reg,
                            use_bias=False, name='filter')(stim_ravel)

        init = {
            'w': tf.zeros_initializer(),
            'b': tf.zeros_initializer()
        }
        reg = {
            'w': tf.contrib.layers.l2_regularizer(self.l2_hist),
            'b': tf.contrib.layers.l2_regularizer(self.l2_hist)
        }
        hist_ravel = snt.BatchFlatten()(hist)
        u_hist = snt.Linear(1, initializers=init, regularizers=reg,
                            name='spike_history')(hist_ravel)

        vm = u_stim + u_hist
        return self.nln(vm)


def datafeed(expt, keys, batchsize=5000, niter=1000):

    # number of data points
    T = expt.X.shape[0]
    inds = np.arange(T)

    for _ in range(niter):
        if batchsize is None:
            b = inds
        else:
            np.random.shuffle(inds)
            b = inds[:batchsize]

        yield {
            keys.stim: expt.X[b],
            keys.hist: expt.spkhist[b],
            keys.rate: expt.y[b].reshape(-1, 1)
        }


def fit_glm(ci, exptdate, stimulus, cutout=7, history=40):
    tp.banner(f'Training GLM, expt {exptdate}, {stimulus}, cell {ci+1:02d}')

    # load experiment
    expt = experiments.loadexpt(exptdate, (ci,), stimulus, 'train', history, 5000, cutout_width=cutout)

    tf.reset_default_graph()

    stim = tf.placeholder(tf.float32, shape=(None, *expt.X.shape[1:]), name='stimulus')
    hist = tf.placeholder(tf.float32, shape=(None, *expt.spkhist.shape[1:]), name='spike_history')
    rate = tf.placeholder(tf.float32, (None, 1), name='firing_rate')
    graph = Graph(stim, hist, rate)

    # define model
    model = GLM(l2_filter=1e-4, l2_hist=1e-2)
    pred = model(stim, hist)

    # neg. log-likelihood loss
    epsilon = 1e-6
    dt = 1e-2
    loss = dt * tf.reduce_mean(pred - rate * tf.log(pred + epsilon))

    # training
    opt = tf.train.AdamOptimizer(learning_rate=1e-3)
    regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_total = tf.reduce_sum(regs)
    train_op = opt.minimize(loss + reg_total)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    metrics = {
        'loss': loss,
        'filter_norm': tf.nn.l2_loss(model.get_variables()[0]),
        'hist_norm': tf.nn.l2_loss(model.get_variables()[1]),
        'reg_total': reg_total,
        'cc': tf.squeeze(cc(pred, rate)),
        'fev': tf.squeeze(fev(pred, rate)),
        'rmse': tf.squeeze(rmse(pred, rate)),
    }
    for reg in regs:
        if 'spike_history' in reg.name:
            if 'w' in reg.name:
                metrics['l2_hist'] = reg
            else:
                metrics['l2_offset'] = reg
        if 'filter' in reg.name:
            metrics['l2_filter'] = reg

    store = defaultdict(list)

    feed = datafeed(expt, graph, niter=1000)
    for fd in tqdm(feed):
        res = sess.run([train_op, metrics], feed_dict=fd)[1]
        for k, v in res.items():
            store[k].append(v)

    # get parameters
    weights, whist, offset = sess.run(model.get_variables())
    sta = weights.reshape(history, cutout * 2, cutout * 2)
    whist = whist.ravel()

    testdata = experiments.loadexpt(exptdate, (ci,), stimulus, 'test', history, 5000, cutout_width=cutout)
    test_feed = next(datafeed(testdata, graph, batchsize=None))
    test_pred = sess.run(pred, feed_dict=test_feed)[:, 0]
    test_rate = testdata.y

    # test metrics
    test = {
        'cc': np_wrap(cc)(test_rate, test_pred),
        'rmse': np_wrap(rmse)(test_rate, test_pred),
        'fev': np_wrap(fev)(test_rate, test_pred),
    }

    # save
    results = {
        'stimulus_filter': sta,
        'history_filter': whist,
        'bias': offset,
        'test_scores': test,
        'train_store': dict(store),
        'test_pred': test_pred,
        'test_rate': test_rate,
    }

    basedir = os.path.expanduser('~/research/deep-retina/results/glms/')
    folder = os.path.join(basedir, f'{exptdate}_{stimulus}')
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    dd.io.save(os.path.join(folder, f'cell{ci:02d}.h5'), results)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # expts = ('15-10-07', '15-11-21a', '15-11-21b')
    # stims = ('whitenoise', 'naturalscene')

    parser = argparse.ArgumentParser(description='Train a GLM model')
    parser.add_argument('--expt', help='Experiment date (e.g. 15-10-07)')
    parser.add_argument('--stim', help='Stimulus class (e.g. naturalscene)')
    parser.add_argument('--cell', help='Cell index')
    args = parser.parse_args()

    fit_glm(int(args.cell), args.expt, args.stim)
