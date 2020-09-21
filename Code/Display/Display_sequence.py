import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate
import glob
import os

# Input arrays of two parameters and phase


def Display_sequence(Solver):
	fig, axs = plt.subplots(2)
	fig.suptitle('Converged:'+str(Solver.converged))
	for i in range(Solver.N_params):
		axs[0].plot(np.arange(Solver.sol_seq.shape[0]),Solver.sol_seq[:,i],label=Solver.Hamiltonian.Dict[i])
	axs[0].set_title('Mean Field Parameters')
	axs[0].legend()
	
	axs[1].plot(np.arange(len(Solver.beta_seq)),Solver.beta_seq)
	axs[1].set_title('Pullay Mixing Factor')
	plt.xlabel('Itteration')
	plt.show()
