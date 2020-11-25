import sys
import Code.Utils as Utils
import numpy as np
import os
from time import time
import argparse
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Solver.Optimizer import Optimizer_exhaustive as Optimizer_exhaustive

Model_Params = dict(
    N_shape=(50, 50),
    Filling=0.25,
    BZ_rot=1,
    stress=-1,
    Delta_CT=0,
    eps=0,
    t_1=1,
    t_2=0.15,
    t_4=0.05,
    U=1,
    J=1)

i, j = 'U', 'J',
i_values = np.linspace(0, 3, 40)
j_values = np.linspace(0, 6, 40)

method = 'sigmoid'
beta = 1.5
Itteration_limit = 250
tolerance = 1e-3
bw_norm = True

verbose = True
save_guess_mfps = True

params_list = [
    (1, 1, 0, 1, 0.15),
    (1, 0.5, 0, 1, 0.15),
    (0, 0.2, 0.5, 0, 0),
    (0.1, 0.5, 1, 0.5, 0.1),
    (0.5, 0.5, 0, 0.5, 0.1),
    (0.5, 0.5, 0.5, 0.5, 0.5)
]

Batch_Folder = 'Meta'

# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--run_ind', type=int, default=0)
args = parser.parse_args()

epsilons = np.linspace(0, 1, 10)
delta_cts = np.linspace(-1.5, 1.5, 10)

model_params_lists = Utils.tuplelist([epsilons, delta_cts])
Model_Params['eps'], Model_Params['Delta_CT'] = model_params_lists[args.run_ind]

Run_ID = 'Itterated:'+str(i)+'_'+str(j)+'_'
Run_ID = Run_ID + '_'.join("{!s}={!r}".format(key, val) for (key, val) in Model_Params.items())

Results_Folder = os.path.join(Batch_Folder, Run_ID)
if not os.path.exists(Results_Folder):  os.makedirs(Results_Folder)

sys.stdout = open(Results_Folder+'/logs.txt', 'w+')
for (key, val) in Model_Params.items():
    print("{!s}={!r}".format(key, val))
print("Itterated : {} Values: {} \n ".format(i, i_values))
print("Itterated : {} Values: {}".format(j, j_values))

for n in range(len(params_list)):
    # Guesses Input

    MF_params = np.array(params_list[n])
    Guess_Name = 'Guess'+str(MF_params)
    outfolder = os.path.join(Results_Folder, 'Guesses_Results', Guess_Name)
    if not os.path.exists(outfolder):  os.makedirs(outfolder)

    # Code
    a = time()

    Model = Hamiltonian(Model_Params, MF_params)
    Solver = HFA_Solver(Model, method=method, beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)
    sweeper = Phase_Diagram_Sweeper(Model, Solver, MF_params, i, i_values, j, j_values, n_threads=args.n_threads, Bandwidth_Normalization=bw_norm, verbose=verbose)

    sweeper.Sweep()
    sweeper.save_results(outfolder, Include_MFPs=save_guess_mfps)

    print(f'Diagram itteration: {n} time to complete (s): {round(time()-a,3)} Converged points:{round(sweeper.Convergence_pc,3)} % \n')

a = time()
Input_Folder = os.path.join(Results_Folder, 'Guesses_Results')
Final_Results_Folder = os.path.join(Results_Folder, 'Final_Results')

if not os.path.exists(Final_Results_Folder): os.makedirs(Final_Results_Folder)

Model = Hamiltonian(Model_Params)
Solver = HFA_Solver(Model, method=method, beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)

Optimal_guesses, Optimal_Energy = Optimizer_exhaustive(Input_Folder, params_list, input_MFP=save_guess_mfps)

sweeper = Phase_Diagram_Sweeper(Model, Solver, Optimal_guesses, i, i_values, j, j_values, n_threads=args.n_threads,Bandwidth_Normalization=bw_norm, verbose=verbose)

sweeper.Sweep()
sweeper.save_results(Final_Results_Folder, Include_MFPs=True)

Final_Energies = sweeper.Es_trial
print(f'Initial guess sweep and final calculations are consistent:{np.array_equal(Final_Energies, Optimal_Energy)}')

print(f'time to complete (s):{round(time()-a,3)} Converged points:{round(sweeper.Convergence_pc,3)} % \n')
