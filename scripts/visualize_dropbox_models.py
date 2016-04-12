import os
from deepretina.toolbox import scandb
import matplotlib.pyplot as plt


def main(database_directory):
    """Stores parameters in a directory in Dropbox"""
    dropbox = os.path.expanduser('~/Dropbox/deep-retina/saved')
    dropbox_files = os.listdir(dropbox)
    models = scandb(database_directory)

    # loop over all models
    for mdl in models:

        # first check to make sure this model exists in Dropbox
        if mdl.key in dropbox_files:

            # check to see if the parameters directory already exists
            paramdir = os.path.join(dropbox, mdl.key, 'parameters')
            if not os.path.exists(paramdir):
                print('Generating plots for {}'.format(mdl.key))

                # try making the plot
                try:

                    # generate the figures
                    figs = mdl.plot()

                    # save the figures
                    os.mkdir(paramdir)
                    for ix, fig in enumerate(figs):
                        fname = os.path.join(paramdir, 'fig{}.jpg'.format(ix))
                        fig.savefig(fname, bbox_inches='tight', dpi=100)

                    plt.close('all')

                # if the plotting fails for some reason
                except (ValueError, AttributeError):
                    continue

if __name__ == "__main__":
    main('/Volumes/sni/deep-retina/database/')
