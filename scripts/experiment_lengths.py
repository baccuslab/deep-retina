from main_lane import *

stim_types = ['whitenoise_1_min', 'whitenoise_2_min', 'whitenoise_4_min', 'whitenoise_8_min', 'whitenoise_16_min', 'whitenoise_32_min']
cells = [0,1,2,3,4]
num_epochs = 60

for stim in stim_types:
    mdl = fit_convnet(cells, stim, num_epochs=num_epochs, exptdate='15-10-07')
