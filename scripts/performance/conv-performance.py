import numpy as np
from os.path import expanduser
import os
import json
import theano
# import pyret.filtertools as ft
# import pyret.visualizations as pyviz
# import deepretina.visualizations as viz
from preprocessing import datagen, loadexpt
from keras.models import model_from_json
import h5py
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#LoadData for cell 

architecture_filename = 'architecture.json'
whitenoise_data_dir = expanduser('~/Dropbox/deep-retina/saved/lenna.salamander/2015-11-13 15.25.14 convnet/')
whitenoise_weight_filename = 'epoch006_iter00450_weights.h5' # .63 cc on held-out
#naturalscenes_data_dir = expanduser('~/lenna.nirum/2015-11-13 02.26.28 convnet/')
naturalscenes_data_dir = expanduser('~/Dropbox/deep-retina/saved/lenna.salamander/2015-11-13 00.14.26 convnet/')
naturalscenes_weight_filename = 'epoch025_iter01800_weights.h5' # .53 cc on held-out

#LoadWhiteNoise
whitenoise_architecture_data = open(whitenoise_data_dir + architecture_filename, 'r')
whitenoise_architecture_string = whitenoise_architecture_data.read()
whitenoise_model = model_from_json(whitenoise_architecture_string)
whitenoise_model.load_weights(whitenoise_data_dir + whitenoise_weight_filename)

#Load nat scenes
naturalscenes_architecture_data = open(naturalscenes_data_dir + architecture_filename, 'r')
naturalscenes_architecture_string = naturalscenes_architecture_data.read()
naturalscenes_model = model_from_json(naturalscenes_architecture_string)
naturalscenes_model.load_weights(naturalscenes_data_dir + naturalscenes_weight_filename)

naturalscenes_test = loadexpt(1, 'naturalscene', 'test', 40)
whitenoise_test = loadexpt(1, 'whitenoise', 'test', 40)

whitenoise_truth = []
naturalscenes_truth = []
whitenoise_on_whitenoise = []
whitenoise_on_naturalscenes = []
naturalscenes_on_whitenoise = []
naturalscenes_on_naturalscenes = []

#White noise test
for X, y in datagen(50, *whitenoise_test):
    whitenoise_truth.extend(y)
    whitenoise_on_whitenoise.extend(whitenoise_model.predict(X))
    naturalscenes_on_whitenoise.extend(naturalscenes_model.predict(X))

whitewhite_performance = pearsonr(np.array(whitenoise_truth), np.array(whitenoise_on_whitenoise).squeeze())[0]
print("Train White, Test White: " + str(whitewhite_performance))

naturalwhite_performance = pearsonr(np.array(whitenoise_truth), np.array(naturalscenes_on_whitenoise).squeeze())[0]
print("Train Nat, Test White: " + str(naturalwhite_performance))

#Nat test
for X, y in datagen(50, *naturalscenes_test):
    naturalscenes_truth.extend(y)
    whitenoise_on_naturalscenes.extend(whitenoise_model.predict(X))
    naturalscenes_on_naturalscenes.extend(naturalscenes_model.predict(X))

naturalnatural_performance = pearsonr(np.array(naturalscenes_truth), np.array(naturalscenes_on_naturalscenes).squeeze())[0]
print("Train Natural, Test Natural: " + str(naturalnatural_performance))

whitenatural_performance = pearsonr(np.array(naturalscenes_truth), np.array(whitenoise_on_naturalscenes).squeeze())[0]
print("Train White, Test Natural: " + str(whitenatural_performance))
