"""
Generates model responses to structured stimuli

..author: Niru Maheswaranathan
..created: 4/21/16
"""
import os
from deepretina.toolbox import scandb, select
import deepretina.retinal_phenomena as rp
import matplotlib.pyplot as plt
from tqdm import tqdm
__version__ = 0.1

suite = {
    'omitted-stimulus-response': ('osr', [], {}),
    'reversing-grating': ('reversing_grating', [], {'size': 3}),
    'paired-flash-20ms': ('paired_flash', [], {'ifi': 2}),
    'off-step': ('step_response', [], {'intensity': -2}),
    'on-step': ('step_response', [], {'intensity': 2}),
}


def run_battery(key, model, filetype='pdf'):
    """Runs a suite of stimuli through the given model"""

    # create a folder for this model to hold figures
    os.mkdir(key)

    for figure_name, (func, args, kwargs) in tqdm(suite.items()):

        # run the stimuli through the model, generate the visualization
        figures, *_ = getattr(rp, func)(model, *args, **kwargs)

        for idx, fig in enumerate(figures):

            # make the figure filename
            cell = 'population' if idx == 0 else 'cell{}'.format(idx)
            filename = '/'.join([key, '{}_{}.{}'.format(figure_name, cell, filetype)])

            # save
            fig.savefig(filename, bbox_inches='tight')

        plt.close('all')


if __name__ == '__main__':

    # load the  models
    db = scandb('/Volumes/sni/deep-retina/database')

    # keys to analyze
    # keys = ['0b7e61', '4ef88d', '689d81', '3520cd', 'b06a60', '3585ed'] # 'd66598', '']
    keys = ['4cac52']

    # run the suite of structured stimuli through the models
    [run_battery(mdl.hashkey, mdl.keras()) for mdl in select(db, keys)]
