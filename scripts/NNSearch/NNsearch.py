from time import time
import numpy as np
import itertools
import sys
import os
import params
import Code.Solver.Optimizer_touchup as ot
import Code.Solver.PhaseDiagramSweeper as Sweeper
import Code.Display.DiagramPlots as Dp
"""
Feed incomplete final results, itterates with nearest neighbours to try and fill the gaps
"""


Model = params.Model
Solver = params.Solver
MFPs = params.Initial_mpfs
Convergence_Grid = params.Initial_Convergence_Grid

a = time()
old_convergence = np.mean(Convergence_Grid)*100
print('Initial convergence',old_convergence)
# 1 run optimer
optimal_guesses = ot.Optimizer_touchup(MFPs,Convergence_Grid)
# 2 feed into sweeper
sweeper = Sweeper.Phase_Diagram_Sweeper(Model,Solver,optimal_guesses,params.U_values,params.J_values,params.n_threads,verbose=params.verbose)
# 3 compute once
sweeper.Sweep()
new_convergence = sweeper.Convergence_pc
print('old_convergence:',old_convergence,'new_convergence:',new_convergence)
# 4 recompute, Do not save intermediary steps, only keep going as convergence increases
while new_convergence > old_convergence:
# for i in range(2):
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

Dp.DiagramPlots(Final_Results_Folder,params.Dict)

print('time to complete (s):',round(time()-a,3),'\n')