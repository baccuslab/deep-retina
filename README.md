## Deep Learning Models of the Retinal Response to Natural Scenes
Deep retina is a project to test to what degree artificial neural networks can predict retinal ganglion cell responses to natural stimuli.

Please see our [NIPS paper](https://arxiv.org/abs/1702.01825) for more details.

Note that deepretina requires python 3.5 or higher.

### Usage
To install the dependencies, run `pip install -r requirements.txt`. If you run the `runme.py` script, it will print out a brief overview of the different modules in deepretina (assuming it is able to import everything correctly).

The following is a high level description of the different modules:
- `core.py`: contains a function for training a deepretina model
- `models.py`: contains functions for building different kinds of deepretina models (convnets, RNNs, etc.)
- `experiments.py`: class structure for loading experimental data
- `io.py`: contains tools for saving model training progress and parameters to disk

A more comprehensive tutorial is in the works.

### Contact
Lane McIntosh (lmcintosh@stanford.edu) and Niru Maheswaranathan (nirum@stanford.edu)
