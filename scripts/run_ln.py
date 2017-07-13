import subprocess
from tqdm import tqdm
from deepretina.experiments import CELLS
EXPTS = ('15-10-07', '15-11-21a', '15-11-21b')
STIMS = ('whitenoise', 'naturalscene')

for expt in EXPTS:
    for stim in STIMS:
        for ci in tqdm(CELLS[expt]):
            cmd = ['pig fit_models.py', f'--expt {expt}', f'--stim {stim}',
                   f'--model LN_softlpus', f'--cell {ci}']
            subprocess.run(cmd)
