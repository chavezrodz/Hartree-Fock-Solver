import os
import itertools
import earthpy.plot as ep
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate

def spin_interpreter(mfps,rounding=5):
	b1 = np.array([1,1])
	b2 = np.array([1,-1])

	spin = mfps[1]*b1 + mfps[3]*b2
	spin = np.rint(spin*rounding)
	spin = np.sign(spin)+1

	return spin

def orbit_interpreter(mfps):
	b1 = np.array([1,1])
	b2 = np.array([1,-1])

	orbit = mfps[2]*b1 + mfps[4]*b2
	orbit = np.rint(orbit*10)
	orbit = np.sign(orbit)

	return orbit

def fullphase(mfp):
	phase = np.zeros(5)
	phase[0] = mfp[0]
	spin= spin_interpreter(mfp)
	phase[1],phase[2] = spin[0],spin[1]
	orbit = orbit_interpreter(mfp)
	phase[3],phase[4] = orbit[0],spin[1]
	return phase

def array_interpreter(MFPs):
	Full_phase = np.zeros((MFPs.shape[0],MFPs.shape[1],5))

	for v in itertools.product(np.arange(MFPs.shape[0]),np.arange(MFPs.shape[1])):
		Full_phase[v] = fullphase(MFPs[v])
	return Full_phase

mfp = np.random.rand(3,3,5)
mfp = array_interpreter(mfp)

MF_Spin = mfp[:,:,1:3]
total_spin = np.zeros((3,3))
total_Spin = 10*MF_Spin[:,:,0]+MF_Spin[:,:,1]
print(total_Spin)
