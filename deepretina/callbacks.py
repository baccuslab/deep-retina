"""
Custom deepretina callbacks
"""
import keras
import tableprint as tp


class TPLogger(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        tp.banner(f"Epoch {epoch}")
        print(tp.header(['iter', 'loss']))

    def on_batch_end(self, batch, logs={}):
        print(tp.row([batch, logs['loss']]))

    def on_epoch_end(self, epoch, logs={}):
        print(tp.bottom(2))
        print(logs)
