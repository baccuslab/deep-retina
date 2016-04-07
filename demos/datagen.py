import numpy as np
from scipy.stats import zscore


def generate_sine_wave(num_samples, num_steps, snr=10, dt=1e-2):
    """
    Generates noisy sequence prediction data for a sine wave
    """

    for t in range(num_samples):

        phase = np.random.rand() * 2 * np.pi
        noise = np.random.randn(num_steps + 1)
        signal = np.sin(2 * np.pi * np.arange(num_steps + 1) * dt + phase) / 0.7
        xobs = zscore(signal * snr + noise)

        yield xobs[:-1], xobs[-1]


def generate_batch(*args, **kwargs):

    X = []
    Y = []

    for x, y in generate_sine_wave(*args, **kwargs):
        X.append(x)
        Y.append(y)

    return np.stack(X), np.stack(Y)
