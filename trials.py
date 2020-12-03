import numpy as np
import sys
import os
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Nickelates.Hamiltonian import Hamiltonian
from time import time
import params
import argparse

# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--run_ind', type=int, default=0)
args = parser.parse_args()
"""
# Local test
for n in range(len(params.params_list)):
    # Guesses Input
    args.trial_ind = n
"""

MF_params = np.array(params.params_list[args.trial_ind])
Guess_Name = 'Guess'+str(MF_params)
outfolder = os.path.join(params.Results_Folder, 'Guesses_Results', Guess_Name)
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

sys.stdout = open(outfolder+'/logs.txt', 'w+')
# Code
a = time()

Model = Hamiltonian(params.Model_Params, MF_params)
Solver = HFA_Solver(
    Model,
    method=params.method,
    beta=params.beta,
    Itteration_limit=params.Itteration_limit,
    tol=params.tolerance
    )

sweeper = Phase_Diagram_Sweeper(
    Model, Solver, MF_params,
    params.i, params.i_values, params.j, params.j_values,
    n_threads=args.n_threads,
    Bandwidth_Normalization=params.bw_norm,
    verbose=params.verbose
    )

sweeper.Sweep()
sweeper.save_results(outfolder, Include_MFPs=params.save_guess_mfps)
print(f'Diagram itteration:{args.trial_ind} time to complete (s): {round(time()-a,3)} Converged points:{round(sweeper.Convergence_pc,3)} ')
