import sys
import os
import itertools
import earthpy.plot as ep
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import Code.Nickelates.Interpreter as In
from Code.Utils.Read_MFPs import Read_MFPs


def PhasePlotsStandard(i,i_values,j,j_values,final_results_folder=None,show=False,transparent=False):
	Solutions_folder = os.path.join(final_results_folder,'MF_Solutions')
	if not os.path.exists(Solutions_folder):
		print('Solutions not found')
		sys.exit(2)

	Plots_folder = os.path.join(final_results_folder,'Plots') 
	if not os.path.exists(Plots_folder):
		os.mkdir(Plots_folder)

	MFPs = Read_MFPs(Solutions_folder)
	Phase = In.array_interpreter(MFPs)

	CM = Phase[:,:,0]
	MF_Spin_orb = Phase[:,:,1:]
	spin_orb = In.arr_to_int(MF_Spin_orb)
	unique_states = np.unique(spin_orb)

	f, ax = plt.subplots(figsize=(8,5))
	ax.set_xlabel(i)
	ax.set_ylabel(j)
	ax.set(frame_on=False)
	plt.xticks(np.linspace(0,len(i_values),4),np.linspace(0,max(i_values),4,dtype=int))
	plt.yticks(np.linspace(0,len(j_values),4),np.linspace(0,max(j_values),4,dtype=int))	

	# Charge Modulation
	CS = ax.contour(CM.T,colors='red',levels=[0.1,0.3,0.5])
	ax.clabel(CS, inline=True, fontsize=10)

	im = ax.pcolormesh(spin_orb.T,alpha=1)
	ep.draw_legend(im_ax=im,classes = unique_states, titles=[In.pos_to_label[state] for state in unique_states])

	plt.tight_layout()
	if final_results_folder is not None:
		plt.savefig(Plots_folder+'/PhaseDiagram.png',transparent=transparent)
	if show:
		plt.show()
	plt.close()
