import numpy as np
import os
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Display.ResultsPlots import one_d_plots


Model_Params = dict(
    U=6,
    J=1.5
)

i = 'stress'
i_values = np.linspace(-2, 2, 10)
# i = 'N_shape'
# i_values = [
#     (10, 10),
#     (20, 20),
#     (30, 30),
#     (40, 40),
#     (50, 50),
#     (60, 60),
#     (70, 70),
#     (80, 80),
#     (90, 90),
#     (100, 100),
#     (110, 110),
#     (120, 120),
#     (130, 130),
#     (140, 140),
#     (150, 150),
#     (160, 160),
#     (170, 170),
#     (180, 180),
#     (190, 190),
#     (200, 200),
#     ]

params_list = [
    (1, 1, 0, 1, 0.15),
    (1, 0.5, 0, 1, 0.15),
    (0, 0.2, 0.5, 0, 0),
    (0.1, 0.5, 1, 0.5, 0.1),
    (0.5, 0.5, 0, 0.5, 0.1),
    (0.5, 0.5, 0.5, 0.5, 0.5)
    ]

method = 'sigmoid'
beta = 1.5
Itteration_limit = 250
tolerance = 1e-3
bw_norm = False
verbose = True

Model = Hamiltonian(Model_Params)

Batch_folder = 'one_d_cuts'
Run_ID = 'Itterated:'+str(i)+'_Model_params_'
Run_ID = Run_ID+'_'.join("{!s}={!r}".format(key, val) for (key, val) in Model_Params.items())
Results_Folder = os.path.join('Results', Batch_folder, Run_ID)

one_d_plots(i, i_values, Model.Dict, params_list, Results_Folder)   