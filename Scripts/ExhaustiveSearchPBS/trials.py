import numpy as np
import itertools
import sys
import os
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Nickelates.Hamiltonian import Hamiltonian
from time import time
import params
import argparse

######### Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default = 1)
parser.add_argument('--trial_ind',type=int, default = 8)
args = parser.parse_args()

"""
"""
if not os.path.exists(os.path.join(params.Results_Folder,'logs')):
	os.makedirs(os.path.join(params.Results_Folder,'logs'))
# Local test
for i in range(len(params.params_list)):
	args.trial_ind = i

	############ Guesses Input
	params_list = params.params_list
	MF_params = np.array(params_list[args.trial_ind])

	Guess_Name = 'Guess'+str(MF_params)

	outfolder = os.path.join(params.Results_Folder,'Guesses_Results',Guess_Name)
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)

	########## Code
	a = time()

	Model = Hamiltonian(params.Model_Params, MF_params)
	Solver = HFA_Solver(Model,method=params.method,beta= params.beta, Itteration_limit=params.Itteration_limit, tol=params.tolerance)
	sweeper = Phase_Diagram_Sweeper(Model,Solver,MF_params,params.i,params.i_values,params.j,params.j_values, n_threads=args.n_threads, verbose=params.verbose)

	sweeper.Sweep()
	sweeper.save_results(outfolder,Include_MFPs=params.save_guess_mfps)
	print('\nDiagram itteration:',args.trial_ind, 'time to complete (s):',round(time()-a,3),'Converged points:',round(sweeper.Convergence_pc,3),'%' '\n')