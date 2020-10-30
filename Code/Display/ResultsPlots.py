import earthpy.plot as ep
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from Code.Utils.Read_MFPs import Read_MFPs
import numpy as np
import scipy.interpolate
import glob
import os
import Code.Nickelates.Interpreter as In

def MFP_plots(MFPs, i_label, i_values, j_label, j_values, Dict, results_folder, show, transparent):
	for i in range(len(Dict)):
		arr = np.abs(MFPs[:,:,i].T)
		plt.pcolormesh(arr)
		plt.title(Dict[i])
		plt.xlabel(i_label)
		plt.ylabel(j_label)
		plt.xticks(np.linspace(0,len(i_values),4),np.linspace(0,max(i_values),4,dtype=int))
		plt.yticks(np.linspace(0,len(j_values),4),np.linspace(0,max(j_values),4,dtype=int))
		plt.colorbar()
		if results_folder is not None:
			MFPs_folder = os.path.join(results_folder,'Plots','Mean Field Parameters')
			if not os.path.exists(MFPs_folder): os.mkdir(MFPs_folder)
			plt.savefig(MFPs_folder+'/'+Dict[i]+'.png',transparent=transparent)
		if show:
			plt.show()
		plt.close()

def feature_plot(feature,i_label, i_values, j_label,j_values, results_folder, show, transparent):
	plt.title(feature)
	plt.pcolormesh(np.loadtxt(results_folder+'/'+feature+'.csv',delimiter=',').T,cmap='bone')
	plt.xlabel(i_label)
	plt.ylabel(j_label)
	plt.xticks(np.linspace(0,len(i_values),4),np.linspace(0,max(i_values),4,dtype=int))
	plt.yticks(np.linspace(0,len(j_values),4),np.linspace(0,max(j_values),4,dtype=int))
	plt.colorbar()

	if results_folder is not None:
		features_folder = os.path.join(results_folder,'Plots','Features')
		if not os.path.exists(features_folder): os.mkdir(features_folder)
		plt.savefig(features_folder+'/'+feature+'.png',transparent=transparent)
	if show:
		plt.show()
	plt.close()

def phases_plot(Phase,i_label, i_values, j_label,j_values, results_folder, show, transparent):
	CM = Phase[:,:,0]
	MF_Spin_orb = Phase[:,:,1:]
	spin_orb = In.arr_to_int(MF_Spin_orb)
	unique_states = np.unique(spin_orb)

	f, ax = plt.subplots(figsize=(8,5))
	ax.set_xlabel(i_label)
	ax.set_ylabel(j_label)
	ax.set(frame_on=False)
	plt.xticks(np.linspace(0,len(i_values),4),np.linspace(0,max(i_values),4,dtype=int))
	plt.yticks(np.linspace(0,len(j_values),4),np.linspace(0,max(j_values),4,dtype=int))	

	im = ax.pcolormesh(spin_orb.T,alpha=1)
	ep.draw_legend(im_ax=im,classes = unique_states, titles=[In.pos_to_label[state] for state in unique_states])

	plt.tight_layout()
	if results_folder is not None:
		plt.savefig(results_folder+'/Plots/PhaseDiagram.png',transparent=transparent)
	if show:
		plt.show()
	plt.close()

def E_Plots(i_label, i_values, Dict, guesses, final_results_folder=None, show=False, transparent=False):
	j_label = 'Guesses'
	j_values = np.arange(len(guesses))

	Solutions_folder = os.path.join(final_results_folder,'MF_Solutions')
	if not os.path.exists(Solutions_folder):
		print('Solutions not found')
		sys.exit(2)

	Plots_folder = os.path.join(final_results_folder,'Plots')
	if not os.path.exists(Plots_folder):
		os.mkdir(Plots_folder)

	MF = Read_MFPs(Solutions_folder)
	MFP_plots(MF, i_label, i_values, j_label, j_values, Dict, Plots_folder, show, transparent)

	Phase = In.array_interpreter(MF)
	phases_plot(Phase,i_label, i_values, j_label, j_values, guesses, final_results_folder, show, transparent)
	features = ['Energies', 'Distortion','Convergence','Conductance']
	for feature in features:
		feature_plot(feature,i_label, i_values, j_values, final_results_folder, show, transparent)

def sweeper_plots(i_label,i_values,j_label,j_values,Dict,final_results_folder=None,show=False,transparent=False):
	Solutions_folder = os.path.join(final_results_folder,'MF_Solutions')
	if not os.path.exists(Solutions_folder):
		print('Solutions not found')
		sys.exit(2)

	Plots_folder = os.path.join(final_results_folder,'Plots') 
	if not os.path.exists(Plots_folder):
		os.mkdir(Plots_folder)

	MF = Read_MFPs(Solutions_folder)
	MFP_plots(MF, i_label, i_values, j_label, j_values, Dict, final_results_folder, show, transparent)

	Phase = In.array_interpreter(MF)
	phases_plot(Phase, i_label, i_values, j_label, j_values, final_results_folder, show, transparent)

	features = ['Energies', 'Distortion','Convergence','Conductance']
	for feature in features:
		feature_plot(feature,i_label, i_values, j_label, j_values, final_results_folder, show, transparent)