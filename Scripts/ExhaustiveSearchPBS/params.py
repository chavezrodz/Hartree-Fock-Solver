import os
import numpy as np
from Code.Utils import tuplelist

# Model Params
Model_Params = dict(
    N_shape=(100, 100),
    eps=0.5,
    Delta_CT=0.5
    )

# Diagram Ranges
i, j = 'U', 'J'
i_values = np.linspace(0, 1, 15)
j_values = np.linspace(0, 0.25, 15)

# Guess ranges
deltas = np.linspace(0, 1, 3)
sfm = np.linspace(0, 1, 3)
Deltas_FO = np.linspace(0, 1, 3)
safm = np.linspace(0, 1, 3)
Deltas_AFO = np.linspace(0, 1, 3)
params_list = tuplelist([deltas, sfm, Deltas_FO, safm, Deltas_AFO])

# Solver params
method = 'sigmoid'
beta = 1.5
Itteration_limit = 250
tolerance = 1e-3
bw_norm = True

# Sweeper params
verbose = True
save_guess_mfps = False

Run_ID = 'Itterated:'+str(i)+'_'+str(j)+'_'
Run_ID = Run_ID+'_'.join("{!s}={!r}".format(key, val) for (key, val) in Model_Params.items())

Results_Folder = os.path.join('Results', Run_ID)
