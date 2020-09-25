from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate
import numpy as np
import glob
import os

import seaborn as sns
sns.set_theme()
sns.set_context("paper")

def Itteration_sequence(Solver):
	fig, axs = plt.subplots(2)
	fig.suptitle('Converged: '+str(Solver.converged))
	for i in range(Solver.N_params):
		axs[0].plot(np.arange(Solver.sol_seq.shape[0]),Solver.sol_seq[:,i],label=Solver.Hamiltonian.Dict[i])
	axs[0].set_title('Mean Field Parameters')
	axs[0].legend()
	
	axs[1].scatter(np.arange(len(Solver.beta_seq)),Solver.beta_seq)
	axs[1].set_title('Pullay Mixing Factor')
	plt.xlabel('Itteration')
	plt.show()
