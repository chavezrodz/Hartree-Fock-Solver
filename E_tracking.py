import numpy as np
import os
import argparse
import sys
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.E_Tracker import E_Tracker

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

# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--run_ind', type=int, default=5)
args = parser.parse_args()

Run_ID = 'Itterated:'+str(i)+'_'
Run_ID = Run_ID+'_'.join("{!s}={!r}".format(key, val) for (key, val) in Model_Params.items())

outfolder = os.path.join('Results', 'E_Tracker', Run_ID)

if not os.path.exists(outfolder): os.mkdir(outfolder)

sys.stdout = open(outfolder+'/logs.txt','w+')
for (key, val) in Model_Params.items():
    print("{!s}={!r}".format(key, val))
print("Itterated : {} Values: {}".format(i,i_values))

Model = Hamiltonian(Model_Params)
Solver = HFA_Solver(Model, method=method,beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)
sweeper = E_Tracker(Model, Solver, i, i_values, guesses=params_list, n_threads=8, Bandwidth_Normalization=bw_norm, verbose=verbose)

sweeper.Sweep()
sweeper.save_results(outfolder, Include_MFPs=True)

print('done')
