import numpy as np
import os
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Display.ResultsPlots import E_Plots


Model_Params = dict(
    N_shape=(50, 50),
    Filling=0.25,
    Delta_CT=1,
    stress=-1,
    BZ_rot=1,
    eps=0,
    t_1=1,
    t_2=0.15,
    t_4=0.05,
    U=3,
    J=0.1)

i = 'U'
i_values = np.linspace(0, 4, 35)

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
bw_norm = True
verbose = True

Model = Hamiltonian(Model_Params)

run_folders = 'tent/1d'
for folder in os.listdir(os.path.join('Results', run_folders)):
    frf = os.path.join('Results', run_folders, folder)
    print(frf)
    E_Plots(i, i_values, Model.Dict, params_list, frf)