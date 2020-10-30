import numpy as np
import itertools
import sys
import os
from time import time
import argparse
import logging
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Utils.tuplelist import tuplelist as tp
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Nickelates.Hamiltonian import Hamiltonian
import Code.Solver.Optimizer
from Code.Solver.Optimizer import Optimizer_exhaustive as Optimizer_exhaustive
from Code.Display.ResultsPlots import sweeper_plots
# from Code.Display.PhasePlots import PhasePlots as PhasePlots
from Code.Display.PhasePlotsStandard import PhasePlotsStandard as PhasePlots

Model_Params = dict(
N_shape = (3,3),
Filling = 0.25,
stress=0,
eps = 0,
t_1 = 1,
t_2 = 0.15,
t_4 = 0.05,
U = 1,
J = 1)

i,j = 'U','J',
i_values = np.linspace(0,6,3)
j_values = np.linspace(0,3,3)

params_list =[
(1,1,0,1,0.15),
(1,0.5,0,1,0.15),
# (0,0.2,0.5,0,0),
# (0.1,0.5,1,0.5,0.1),
# (0.5,0.5,0,0.5,0.1),
# (0.5,0.5,0.5,0.5,0.5)
]

method ='sigmoid'
beta = 1.5
Itteration_limit = 35
tolerance = 1e-3
bw_norm = True

verbose = False
save_guess_mfps = True

######### Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default = 8)
parser.add_argument('--run_ind',type=int, default=5)
args = parser.parse_args()

epsilons = [0,0.3,0.6,0.8]
strains = [-0.025,-1,0,1,2.5]
dopings = [0.2,0.25,0.3]

model_params_lists = tp([epsilons,strains,dopings])
Model_Params['eps'],Model_Params['stress'],Model_Params['Filling'] = model_params_lists[args.run_ind]


Run_ID = 'Itterated:'+str(i)+'_'+str(j)+'_'
Run_ID = Run_ID+'_'.join("{!s}={!r}".format(key,val) for (key,val) in Model_Params.items())

Results_Folder = os.path.join('Results',Run_ID)


for n in range(len(params_list)):
	############ Guesses Input

	MF_params = np.array(params_list[n])
	Guess_Name = 'Guess'+str(MF_params)
	outfolder = os.path.join(Results_Folder,'Guesses_Results',Guess_Name)
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)

	########## Code
	a = time()

	Model = Hamiltonian(Model_Params, MF_params)
	Solver = HFA_Solver(Model,method=method,beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)
	sweeper = Phase_Diagram_Sweeper(Model,Solver,MF_params,i,i_values,j,j_values, n_threads=args.n_threads, Bandwidth_Normalization=bw_norm, verbose=verbose)

	sweeper.Sweep()
	sweeper.save_results(outfolder,Include_MFPs=save_guess_mfps)

	sweeper_plots(i+'/W',i_values,j+'/W',j_values,Model.Dict,outfolder)

	print(f'Diagram itteration: {n} time to complete (s): {round(time()-a,3)} Converged points:{round(sweeper.Convergence_pc,3)}\n')

a = time()
Input_Folder = os.path.join(Results_Folder,'Guesses_Results')
Final_Results_Folder = os.path.join(Results_Folder,'Final_Results')

if not os.path.exists(Final_Results_Folder):
    os.makedirs(Final_Results_Folder)

Model = Hamiltonian(Model_Params)
Solver = HFA_Solver(Model,method=method, beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)

Optimal_guesses, Optimal_Energy = Optimizer_exhaustive(Input_Folder, params_list,input_MFP=save_guess_mfps)

sweeper = Phase_Diagram_Sweeper(Model,Solver,Optimal_guesses,i,i_values,j,j_values,n_threads=args.n_threads,Bandwidth_Normalization=bw_norm, verbose=verbose)

sweeper.Sweep()
sweeper.save_results(Final_Results_Folder,Include_MFPs=True)
sweeper_plots(i+'/W',i_values,j+'/W',j_values,Model.Dict,Final_Results_Folder)

Final_Energies = sweeper.Es_trial
print(f'Initial guess sweep and final calculations are consistent:{np.array_equal(Final_Energies, Optimal_Energy)}')

print(f'time to complete (s):{round(time()-a,3)} Converged points:{round(sweeper.Convergence_pc,3)} % \n')
