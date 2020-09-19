import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate
import glob
import os

# Input arrays of two parameters and phase


def Display_sequence(Solver):
	for i in range(Solver.N_params):
		plt.plot(np.arange(Solver.sol_seq.shape[0]),Solver.sol_seq[:,i],label=Solver.Hamiltonian.Dict[i])
	plt.title('Converged:'+str(Solver.converged))
	plt.xlabel('Itteration')
	plt.ylabel('Parameter Value$')
	plt.legend()
	plt.show()
