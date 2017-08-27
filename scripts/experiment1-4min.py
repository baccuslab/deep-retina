from main_lane import *

stim_types = ['whitenoise_1_min', 'whitenoise_2_min', 'whitenoise_4_min']
expt_length = [1.0, 2.0, 4.0]
cells = [0,1,2,3,4]

for i,stim in enumerate(stim_types):
    num_epochs = int(2065/(expt_length[i]+1))
    mdl = fit_convnet(cells, stim, num_epochs=num_epochs, exptdate='15-10-07', l2=0.03)
