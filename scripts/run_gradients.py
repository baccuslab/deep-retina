import matplotlib
matplotlib.use('Agg')
import os
import keras.backend as K
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from deepretina.toolbox import Model
from deepretina.experiments import Experiment
import tensorflow as tf


def inspect(model, X):
    """Get the response from all layers of a model"""
    all_layers = K.function([model.layers[0].input, K.learning_phase()],
                            [layer.output for layer in model.layers])
    outputs = all_layers([X, 0])

    results = OrderedDict()
    for layer, output in zip(model.layers, outputs):
        results[layer.name] = output

    return results


def gradient(model, X, ci, sess):
    grads = K.gradients(model.output[:, ci], model.input)[0]
    return sess.run(grads, feed_dict={model.input: X, K.learning_phase(): False})


def savemov(movies, subplots, filename, cmaps, T=None, clim=None, fps=15, figsize=None):

    # Set up the figure
    fig, axs = plt.subplots(*subplots, figsize=figsize)

    # total length
    if T is None:
        T = movies[0].shape[0]

    # mean subtract
    xs = []
    imgs = []
    for movie, ax, cmap in zip(movies, axs, cmaps):
        X = movie.copy()
        # X -= X.mean()
        xs.append(X)

        img = ax.imshow(X[0])
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_xlim(0, X.shape[1])
        ax.set_ylim(0, X.shape[2])
        ax.set_xticks([])
        ax.set_yticks([])

        # Set up the colors
        img.set_cmap(cmap)
        img.set_interpolation('nearest')
        if clim is not None:
            img.set_clim(clim)
        else:
            maxval = np.max(np.abs(X))
            img.set_clim([-maxval, maxval])
        imgs.append(img)

    plt.show()
    plt.draw()

    dt = 1 / fps

    def animate(t):
        for X, img, ax in zip(xs, imgs, axs):
            i = np.mod(int(t / dt), T)
            ax.set_title('frame %2.2f' %(float(i)/dt))
            img.set_data(X[i])
        return mplfig_to_npimage(fig)

    animation = VideoClip(animate, duration=T * dt)
    # animation.write_gif(filename + '.gif', fps=fps)
    animation.write_videofile(filename + '.mp4', fps=fps)


if __name__ == '__main__':
    # parameters
    timesteps = 100
    height = 50
    width = 50
    filter_length = 40
    cells = [0,1,2,3,4]
    nout = len(cells)
    input_shape = (timesteps, filter_length, height, width)

    # load model
    mdl = Model(os.path.expanduser('~/deep-retina-results/database'), '7fc87c bn_cnn_requ')
    #keras_mdl = mdl.keras()
    
    # load data
    # load experiment data
    data = Experiment('15-10-07', cells, [], ['naturalscene_4_6_2017'], 40, timesteps, nskip=6000)
    X = data._test_data['naturalscene_4_6_2017'][0][:timesteps]

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        keras_mdl = mdl.keras()
        g = gradient(keras_mdl, X, 0, sess)

    plt.imshow(g[50,0], cmap='seismic')
    plt.savefig('temp.png')
