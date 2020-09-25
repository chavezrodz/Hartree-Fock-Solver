import os
import itertools
import earthpy.plot as ep
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate

def spin_interpreter(mfps):
	b1 = np.array([1,1])
	b2 = np.array([1,-1])

	spin = mfps[1]*b1 + mfps[3]*b2
	spin = np.rint(spin*2)
	spin = np.sign(spin)

	if np.array_equal(spin,np.array([0,0])):
		spin = 0
	elif np.array_equal(spin,np.array([1,0])):
		spin = 1
	elif np.array_equal(spin,np.array([1,-1])):
		spin = 2
	elif np.array_equal(spin,np.array([1,1])):
		spin = 3
	else:
		spin = 4
	return spin

def orbit_interpreter(mfps):
	b1 = np.array([1,1])
	b2 = np.array([1,-1])

	orbit = mfps[2]*b1 + mfps[4]*b2
	orbit = np.rint(orbit*2)
	orbit = np.sign(orbit)

	if np.array_equal(orbit,np.array([0,0])):
		orbit = 0
	elif np.array_equal(orbit,np.array([1,0])):
		orbit = 1
	elif np.array_equal(orbit,np.array([1,-1])):
		orbit = 2
	elif np.array_equal(orbit,np.array([1,1])):
		orbit = 3
	else:
		orbit = 4
	return orbit

def fullphase(mfp):
	phase = np.zeros(3)
	phase[0] = mfp[0]
	phase[1] = spin_interpreter(mfp)
	phase[2] = orbit_interpreter(mfp)
	return phase

def array_interpreter(MFPs):
	Full_phase = np.zeros((MFPs.shape[0],MFPs.shape[1],3))

	for v in itertools.product(np.arange(MFPs.shape[0]),np.arange(MFPs.shape[1])):
		Full_phase[v] = fullphase(MFPs[v])
	return Full_phase

# mfp = np.random.rand(3,3,5)
# print(array_interpreter(mfp))