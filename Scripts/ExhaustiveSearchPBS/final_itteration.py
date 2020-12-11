import numpy as np
import sys
import os
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Solver.Optimizer import Optimizer_exhaustive
from time import time
import argparse
import params

# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
args = parser.parse_args()

# Paths
Input_Folder = os.path.join(params.Results_Folder, 'Guesses_Results')
outfolder = os.path.join(params.Results_Folder, 'Final_Results')
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
sys.stdout = open(outfolder+'/logs.txt', 'w+')

# Code
a = time()
Model = Hamiltonian(params.Model_Params)

Solver = HFA_Solver(
    Model,
    method=params.method,
    beta=params.beta,
    Itteration_limit=params.Itteration_limit,
    tol=params.tolerance
    )

Optimal_guesses, Optimal_Energy = Optimizer_exhaustive(
    Input_Folder, params.params_list,
    input_MFP=params.save_guess_mfps,
    verbose=params.verbose)

sweeper = Phase_Diagram_Sweeper(
    Model, Solver, Optimal_guesses,
    params.i, params.i_values, params.j, params.j_values,
    n_threads=args.n_threads,
    Bandwidth_Normalization=params.bw_norm,
    verbose=params.verbose
    )

sweeper.Sweep()
sweeper.save_results(outfolder, Include_MFPs=True)

Final_Energies = sweeper.Es_trial
print(Final_Energies, '\n', Optimal_Energy)

print(f'Initial guess sweep and final calculations are consistent: {np.array_equal(Final_Energies, Optimal_Energy)}')
print(f'time to complete (s):{round(time()-a,3)} Converged points:{round(sweeper.Convergence_pc,3)} % \n')
