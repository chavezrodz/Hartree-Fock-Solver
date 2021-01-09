import numpy as np
import os
import argparse
import sys
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.one_d_sweeper import one_d_sweeper

# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
args = parser.parse_args()

Model_Params = dict(
    N_shape=(25, 25),
)

i = 'stress'
i_values = np.linspace(-5, 5, 5)
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

Batch_folder = 'one_d_cuts'
Run_ID = 'Itterated:'+str(i)+'_Model_params_'
Run_ID = Run_ID+'_'.join("{!s}={!r}".format(key, val) for (key, val) in Model_Params.items())
Results_Folder = os.path.join('Results', Batch_folder, Run_ID)
os.makedirs(Results_Folder, exist_ok=True)

print(f'Results_Folder: {Results_Folder}')

# sys.stdout = open(Results_Folder+'/logs.txt', 'w+')
for (key, val) in Model_Params.items():
    print("{!s}={!r}".format(key, val))
print("Itterated : {} Values: {}".format(i, i_values))

Model = Hamiltonian(Model_Params)
Solver = HFA_Solver(Model, method=method, beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)
sweeper = one_d_sweeper(Model, Solver, i, i_values, guesses=params_list, n_threads=8, Bandwidth_Normalization=bw_norm, verbose=verbose)

sweeper.Sweep()
sweeper.save_results(Results_Folder, Include_MFPs=True)

print('done')
