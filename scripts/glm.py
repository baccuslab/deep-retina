import limo
from .preprocessing import loadexpt

# load data
data = loadexpt([0, 1, 2, 3, 4], 'whitenoise', 'train', history=40)
dt = 1e-2

# GLM features
cellidx = 0
f_stim = limo.Feature(data.X)
opt = limo.PoissonGLM([f_stim], data.y[:, cellidx], dt)

# fit
opt.fit()
