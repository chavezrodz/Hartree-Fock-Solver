import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate
import glob
import os

# Input arrays of two parameters and phase

def DiagramPlots(i_label,j_label,Dict,final_results_folder=None,show=False,transparent=False):
	Solutions_folder = os.path.join(final_results_folder,'MF_Solutions')
	if not os.path.exists(Solutions_folder):
		print('Solutions not found')
		sys.exit(2)

	Plots_folder = os.path.join(final_results_folder,'Plots') 
	if not os.path.exists(Plots_folder):
		os.mkdir(Plots_folder)


	for i in range(len(Dict)):
		MF = np.loadtxt(Solutions_folder+'/MF'+str(i)+'.csv',delimiter=",")
		arr = np.abs(MF[:,:].T)
		plt.pcolormesh(arr)
		plt.title(Dict[i])
		# plt.xlabel('$U/t_{1}$')
		# plt.ylabel('$J/t_{1}$')
		plt.xlabel(i_label)
		plt.ylabel(j_label)
		plt.xticks(np.linspace(0, arr.shape[0], 4),np.arange(0,8,2))
		plt.yticks(np.linspace(0, arr.shape[1], 4),np.arange(0,4,1))	
		plt.colorbar()
		if final_results_folder is not None:
			plt.savefig(Plots_folder+'/'+Dict[i]+'.png',transparent=transparent)
		if show:
			plt.show()
		plt.close()

	plt.title('Convergence Grid')
	plt.pcolormesh(np.loadtxt(final_results_folder+'/Convergence_Grid.csv',delimiter=',').T,cmap='bone')
	plt.xlabel(i_label)
	plt.ylabel(j_label)
	plt.xticks(np.linspace(0, arr.shape[0], 4),np.arange(0,8,2))
	plt.yticks(np.linspace(0, arr.shape[1], 4),np.arange(0,4,1))
	plt.colorbar()
	if final_results_folder is not None:
		plt.savefig(Plots_folder+'/Convergence_Grid.png',transparent=transparent)
	if show:
		plt.show()
	plt.close()

	plt.title('Conductance Grid')
	plt.pcolormesh(np.loadtxt(final_results_folder+'/Conductance_Grid.csv',delimiter=',').T,cmap='bone')
	plt.xlabel(i_label)
	plt.ylabel(j_label)
	plt.xticks(np.linspace(0, arr.shape[0], 4),np.arange(0,8,2))
	plt.yticks(np.linspace(0, arr.shape[1], 4),np.arange(0,4,1))
	plt.colorbar()
	if final_results_folder is not None:
		plt.savefig(Plots_folder+'/Conductance_Grid.png',transparent=transparent)
	if show:
		plt.show()
	plt.close()
