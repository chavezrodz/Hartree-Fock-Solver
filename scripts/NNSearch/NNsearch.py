from time import time
import numpy as np
import itertools
import sys
import os
import params
import Code.Solver.Optimizer_touchup as ot
import Code.Solver.PhaseDiagramSweeper as Sweeper

"""
Feed incomplete final results, itterates with nearest neighbours to try and fill the gaps
"""


Model = params.Model
Solver = params.Solver
MFPs = params.Initial_mpfs
Convergence_Grid = params.Initial_Convergence_Grid


old_convergence = np.mean(Convergence_Grid)*100
print('Initial convergence',old_convergence)

# 1 run optimer
optimal_guesses = ot.Optimizer_touchup(MFPs,Convergence_Grid)
# 2 feed into sweeper
sweeper = Sweeper.Phase_Diagram_Sweeper(Model,Solver,optimal_guesses,params.U_values,params.J_values,params.n_threads,verbose=params.verbose)
sweeper.Sweep()
new_convergence = sweeper.Convergence_pc
# 3 recompute, Do not save intermediary steps, only keep going as convergence increases
while new_convergence > old_convergence:
	old_convergence = new_convergence
	optimal_guesses = ot.Optimizer_touchup(MFPs,Convergence_Grid)
	sweeper.Initial_params = optimal_guesses
	sweeper.Sweep()
	MFPs, Convergence_Grid, new_convergence = sweeper.Final_params, sweeper.Convergence_Grid, sweeper.Convergence_pc
	print('new_convergence:',new_convergence)

# 4 save final results
sweeper.save_results(Final_Results_Folder, Final_Run=True)

print('time to complete (s):',round(time()-a,3),'Converged points:',sweeper.Convergence_pc,'%' '\n')