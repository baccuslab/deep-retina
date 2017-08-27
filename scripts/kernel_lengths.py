from main_lane import *

kernel_lengths = [2,5,10,20,30,40,60,80,100]
stimulus_type = 'whitenoise'
cells = [0,1,2,3,4]
num_epochs = 30

for k in kernel_lengths:
    fit_convnet(cells, stimulus_type, history=k, num_epochs=num_epochs)
