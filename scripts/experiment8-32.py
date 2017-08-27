from main_lane import *

stim_types = ['whitenoise_8_min', 'whitenoise_16_min']
expt_length = [8.0, 16.0]
cells = [0,1,2,3,4]

for i,stim in enumerate(stim_types):
    num_epochs = int(2065/(expt_length[i]))
    mdl = fit_convnet(cells, stim, num_epochs=num_epochs, exptdate='15-10-07', l2=0.03)
