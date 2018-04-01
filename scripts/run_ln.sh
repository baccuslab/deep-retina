#!/bin/bash

# activations = "softplus sigmoid relu requ exp rbf"

# associative array of cells for each experiment
declare -A EXPTS
EXPTS['15-10-07']="0 1 2 3 4"
EXPTS['15-11-21a']="6 10 12 13"
EXPTS['15-11-21b']="0 1 3 5 8 9 13 14 16 17 18 20 21 22 23 24 25"

for expt in ${!EXPTS[@]}
do
  for ci in ${EXPTS[$expt]}
  do
    for stim in 'whitenoise' 'naturalscene'
    do
      python fit_models.py --expt $expt --stim $stim --model LN_requ --cell $ci
    done
  done
done
