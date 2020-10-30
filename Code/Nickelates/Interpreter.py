import os
import itertools
import earthpy.plot as ep
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate

def spin_interpreter(mfps,rounding=1):
	b1 = np.array([1,1])
	b2 = np.array([1,-1])

	spin = mfps[1]*b1 + mfps[3]*b2
	spin = np.rint(spin*rounding)

	#Symetries
	if np.product(np.sign(spin))==1:
		spin = np.abs(spin)
	if np.product(np.sign(spin))==-1 and np.sign(spin[0])==-1:
		spin = np.roll(spin,1)
	if spin[0] == 0:
		spin=np.roll(spin,1)
	if 0 in spin:
		spin=np.abs(np.sign(spin))

	return spin

def orbit_interpreter(mfps,rounding=1):
	b1 = np.array([1,1])
	b2 = np.array([1,-1])

	orbit = mfps[2]*b1 + mfps[4]*b2
	orbit = np.rint(orbit*rounding)
	#Symetries
	if orbit[0] == 0:
		orbit=np.roll(orbit,1)
	if np.product(np.sign(orbit))==-1 and np.sign(orbit[0])==-1:
		orbit = np.roll(orbit,1)
	# orbit = np.array([0,0])
	return orbit

def fullphase(mfp):
	phase = np.zeros(5)
	phase[0] = mfp[0]
	spin= spin_interpreter(mfp)
	orbit = orbit_interpreter(mfp)
	phase[1],phase[2],phase[3],phase[4] = spin[0],spin[1],orbit[0],orbit[1]
	return phase

def array_interpreter(MFPs):
	Full_phase = np.zeros((MFPs.shape[0],MFPs.shape[1],5))

	for v in itertools.product(np.arange(MFPs.shape[0]),np.arange(MFPs.shape[1])):
		Full_phase[v] = fullphase(MFPs[v])
	return Full_phase

def unique_states(state_array):
	states = state_array.reshape(-1,state_array.shape[-1])
	states = np.unique(states,axis=0)
	return list(states)

def vec_to_int(x):
	x = x + 3
	x = x.astype(int)
	ints = ''
	for i in range(len(x)):
		ints = ints + str(x[i])
	ints = int(ints)
	return ints

Spin_Dict = {-2: r' \Downarrow', -1: r' \downarrow', 0:r' 0', 1:r' \uparrow', 2:r' \Uparrow'}
Orbit_Dict = {-2:r' \bar{Z}', -1:r' \bar{z}', 0:r' 0', 1:r' z', 2:r' Z'}

All_states = [vec_to_int(np.array([i,j,k,l])) for i,j,k,l in itertools.product(np.arange(-2,3),repeat=4)]

state_to_pos = {state:int(np.where(np.isclose(All_states,state))[0]) for state in All_states}

state_to_label = {
vec_to_int(np.array([i,j,k,l])): r'$'+Spin_Dict[i]+Spin_Dict[j]+ ', ' +Orbit_Dict[k]+Orbit_Dict[l]+'$'
for i,j,k,l in itertools.product(np.arange(-2,3),repeat=4)
}

pos_to_label={
	state_to_pos[state]:state_to_label[state] for state in All_states
}

def arr_to_int(MFPs):
	Full_phase = np.zeros((MFPs.shape[0],MFPs.shape[1]))
	for v in itertools.product(np.arange(MFPs.shape[0]),np.arange(MFPs.shape[1])):
		state = vec_to_int(np.array(MFPs[v]))
		state = state_to_pos[state]
		Full_phase[v] = state
	return Full_phase.astype(int)
