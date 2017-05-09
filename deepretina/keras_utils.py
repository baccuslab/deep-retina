import keras.backend as K
from collections import OrderedDict


def inspect(model, X):
    """get a stimulus response from all layers of a model"""
    all_layers = K.function([model.layers[0].input, K.learning_phase()],
                            [layer.output for layer in model.layers])
    outputs = all_layers([X, 0])

    results = OrderedDict()
    for layer, output in zip(model.layers, outputs):
        results[layer.name] = output

    return results
