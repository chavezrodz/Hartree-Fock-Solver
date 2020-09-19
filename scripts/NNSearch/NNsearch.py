from time import time
import numpy as np
import itertools
import sys
import os
import params
import Code.Nickelates.Hamiltonian as Ni
import Code.Solver.HFA_Solver as HFA
import Code.Solver.Optimizer_touchup as ot
import Code.Solver.PhaseDiagramSweeper as Sweeper
import Code.Display.DiagramPlots as Dp
import argparse
"""
Feed incomplete final results, itterates with nearest neighbours to try and fill the gaps
"""

########## Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
args = parser.parse_args()
n_threads = args.n_threads

######### Params file arguments
Model=Ni.Hamiltonian(params.Model_Params)
Solver = HFA.HFA_Solver(Model,beta=params.beta, Itteration_limit=params.Itteration_limit, tol=params.tol)
MFPs = params.Initial_mpfs
Convergence_Grid = params.Initial_Convergence_Grid

a = time()
old_convergence = np.mean(Convergence_Grid)*100
print('Initial convergence',old_convergence)
# 1 run optimer
optimal_guesses = ot.Optimizer_touchup(MFPs,Convergence_Grid)
# 2 feed into sweeper
sweeper = Sweeper.Phase_Diagram_Sweeper(Model,Solver,optimal_guesses,params.U_values,params.J_values,n_threads,verbose=params.verbose)
# 3 compute once
sweeper.Sweep()
new_convergence = sweeper.Convergence_pc
print('old_convergence:',old_convergence,'new_convergence:',new_convergence)
# 4 recompute, Do not save intermediary steps, only keep going as convergence increases
# while new_convergence > old_convergence:
for i in range(3):
	old_convergence = new_convergence
	optimal_guesses = ot.Optimizer_touchup(sweeper.Final_params,sweeper.Convergence_Grid)
	sweeper.Initial_params = optimal_guesses
	sweeper.Sweep()
	new_convergence = sweeper.Convergence_pc
	print('old_convergence:',old_convergence,'new_convergence:',new_convergence)

# 4 save final results
Final_Results_Folder = params.outfolder

if not os.path.exists(Final_Results_Folder):
    os.makedirs(Final_Results_Folder)
    os.makedirs(os.path.join(Final_Results_Folder,'MF_Solutions'))

sweeper.save_results(Final_Results_Folder, Include_MFPs=True)

Dp.DiagramPlots(Final_Results_Folder,Model.Dict)

print('time to complete (s):',round(time()-a,3),'\n')

"""
Initial convergence 97.88888888888889
old_convergence: 97.88888888888889 new_convergence: 97.55555555555556
old_convergence: 97.55555555555556 new_convergence: 97.55555555555556
"""
