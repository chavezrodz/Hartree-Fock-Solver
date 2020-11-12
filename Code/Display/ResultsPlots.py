import sys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import Code.Utils as Utils
import numpy as np
import scipy.interpolate
import glob
import os
import Code.Nickelates.Interpreter as In

def MFP_plots(MFPs, i_label, i_values, j_label, j_values, Dict, results_folder, show, transparent):
	for i in range(len(Dict)):
		arr = np.abs(MFPs[:,:,i].T)
		plt.pcolormesh(arr,vmin=0,vmax=1)
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
	plt.pcolormesh(np.loadtxt(results_folder+'/'+feature+'.csv',delimiter=',').T)
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

	cmap = plt.cm.get_cmap('prism',170)
	im = ax.pcolormesh(spin_orb.T,alpha=1,cmap=cmap,vmin=0,vmax=169)
	patches = [mpatches.Patch(color=cmap(state), label=In.pos_to_label[state]) for state in unique_states]
	ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, prop={"size": 13})

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
	if not os.path.exists(Solutions_folder): print('Solutions not found'); sys.exit(2)

	MF = Utils.Read_MFPs(Solutions_folder)

	Plots_folder = os.path.join(final_results_folder,'Plots') 
	if not os.path.exists(Plots_folder): os.mkdir(Plots_folder)

	MFP_plots(MF, i_label, i_values, j_label, j_values, Dict, final_results_folder, show, transparent)

	Phase = In.array_interpreter(MF)
	phases_plot(Phase,i_label, i_values, j_label, j_values, final_results_folder, show, transparent)
	features = ['Energies', 'Distortion','Convergence','Conductance']
	for feature in features:
		feature_plot(feature,i_label, i_values, j_label,j_values, final_results_folder, show, transparent)

def sweeper_plots(i_label,i_values,j_label,j_values,Dict,final_results_folder=None,show=False,transparent=False, BW_norm=False):
	Solutions_folder = os.path.join(final_results_folder,'MF_Solutions')
	if not os.path.exists(Solutions_folder): print('Solutions not found'); sys.exit(2)

	Plots_folder = os.path.join(final_results_folder,'Plots') 
	if not os.path.exists(Plots_folder): os.mkdir(Plots_folder)

	if BW_norm: i_label = i_label+'/W'; j_label = j_label+'/W'

	MF = Utils.Read_MFPs(Solutions_folder)
	MFP_plots(MF, i_label, i_values, j_label, j_values, Dict, final_results_folder, show, transparent)

	Phase = In.array_interpreter(MF)
	phases_plot(Phase, i_label, i_values, j_label, j_values, final_results_folder, show, transparent)

	features = ['Energies', 'Distortion','Convergence','Conductance']
	for feature in features:
		feature_plot(feature,i_label, i_values, j_label, j_values, final_results_folder, show, transparent)
