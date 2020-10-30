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


def PhasePlots(i,i_values,j,j_values,final_results_folder=None,show=False,transparent=False):
	Solutions_folder = os.path.join(final_results_folder,'MF_Solutions')
	if not os.path.exists(Solutions_folder):
		print('Solutions not found')
		sys.exit(2)

	Plots_folder = os.path.join(final_results_folder,'Plots') 
	if not os.path.exists(Plots_folder):
		os.mkdir(Plots_folder)

	MFPs = Read_MFPs(Solutions_folder)
	Phase = In.array_interpreter(MFPs)
	Spin_Dict = In.Spin_Dict
	Orbit_Dict = In.Orbit_Dict

	CM = Phase[:,:,0]
	MF_Spin_orb = Phase[:,:,1:]

	f, ax = plt.subplots(figsize=(8,5))
	ax.set_xlabel(i)
	ax.set_ylabel(j)
	ax.set(frame_on=False)
	plt.xticks(np.linspace(0,len(i_values),4),np.linspace(0,max(i_values),4,dtype=int))
	plt.yticks(np.linspace(0,len(j_values),4),np.linspace(0,max(j_values),4,dtype=int))	

	# Charge Modulation
	CS = ax.contour(CM.T,colors='red',levels=[0.1,0.3,0.5])
	ax.clabel(CS, inline=True, fontsize=10)

	#Spin Orbit combinations
	states = In.unique_states(MF_Spin_orb)
	labels = [r'$'+Spin_Dict[state[0]]+Spin_Dict[state[1]]+ ', ' +Orbit_Dict[state[2]]+Orbit_Dict[state[3]]+'$' for state in states]
	int_states = [In.vec_to_int(x) for x in states]
	State_Dict = {int_states[i]:i for i in range(len(int_states))}

	spin_orb = In.arr_to_int(MF_Spin_orb)
	spin_orb = np.vectorize(State_Dict.get)(spin_orb)
	print(len(int_states),len(labels))
	im = ax.pcolormesh(spin_orb.T,alpha=1,cmap='gnuplot')
	ep.draw_legend(im_ax=im, classes = int_states,titles=labels)

	plt.tight_layout()
	if final_results_folder is not None:
		plt.savefig(Plots_folder+'/PhaseDiagram.png',transparent=transparent)
	if show:
		plt.show()
	plt.close()
