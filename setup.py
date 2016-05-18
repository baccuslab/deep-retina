from setuptools import setup, find_packages
import deepretina

setup(name = 'deepretina',
        version = deepretina.__version__,
        description = 'Neural network models of the retina',
        author = 'Niru Maheshwaranathan, Lane McIntosh, Aran Nayebi',
        author_email = 'lmcintosh@stanford.edu',
        url = 'https://github.com/baccuslab/deep-retina.git',
        install_requires = [i.strip() for i in open("requirements.txt").readlines()],
        long_description = '''
            The deepretina package contains methods for learning convolutional
            and LSTM neural network models of the retina.
            ''',
        classifiers = [
            'Intended Audience :: Science/Research',
            'Operating System :: MacOS :: MacOS X',
            'Topic :: Scientific/Engineering :: Information Analysis'],
        packages = find_packages(),
        )

