from main_lane import *

stim_types = ['structured', 'naturalmovie', 'whitenoise', 'naturalscene']
cells = [0,1,2,3,5,6,8,9,10,11,12,13]
num_epochs = 60

for stim in stim_types:
    mdl = fit_convnet(cells, stim, num_epochs=num_epochs, exptdate='16-01-08')
