from setuptools import setup

setup(name = 'deepretina', 
        version = '0.1',
        description = 'Deep learning model of the retina',
        author = 'Niru Maheshwaranathan, Lane McIntosh, Aran Nayebi',
        author_email = 'lmcintosh@stanford.edu',
        url = 'https://github.com/baccuslab/deep-retina.git',
        requires = [i.strip() for i in open("requirements.txt").readlines()],
        long_description = '''
            The deepretina package contains methods for learning convolutional
            and LSTM neural network models of the retina. 
            ''',
        classifiers = [
            'Intended Audience :: Science/Research', 
            'Operating System :: MacOS :: MacOS X',
            'Topic :: Scientific/Engineering :: Information Analysis'],
        packages = ['deepretina'],
        package_dir = {'deepretina': ''},
        py_modules = ['models', 'preprocessing', 'utils', 'visualizations']
        )

