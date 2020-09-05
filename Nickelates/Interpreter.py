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
	"""
	Feed mfps,
	return discretized phase
	"""

	b1 = np.array([1,1])
	b2 = np.array([1,-1])

	spin = mfps[0]*b1 + mfps[1]*b2
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
	return spin

def spin_array_interpreter(MFPs):
	shape = MFPs.shape
	Full_phase = np.zeros((shape[0],shape[1]))

	vectorize =np.vectorize(spin_interpreter)
	for v in itertools.product(np.arange(shape[0]),np.arange(shape[1])):
		Full_phase[v] = spin_interpreter(MFPs[v])

	return Full_phase

def orbit_interpreter(mfps):
	"""
	Feed mfps,
	return discretized phase
	"""

	b1 = np.array([1,1])
	b2 = np.array([1,-1])

	spin = mfps[0]*b1 + mfps[1]*b2
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
	return orbit

def orbit_array_interpreter(MFPs):
	shape = MFPs.shape
	Full_phase = np.zeros((shape[0],shape[1]))

	vectorize =np.vectorize(spin_interpreter)
	for v in itertools.product(np.arange(shape[0]),np.arange(shape[1])):
		Full_phase[v] = spin_interpreter(MFPs[v])

	return Full_phase


# Loading
CM = MF[:,:,0]
OD = MF[:,:,2]
SFM = MF[:,:,1]
SAFM = MF[:,:,3]


f, ax = plt.subplots(figsize=(8,5))
ax.set_xlabel('$U/t_{1}$')
ax.set_ylabel('$J/t_{1}$')
plt.xticks(np.linspace(0, MF.shape[0], 4),np.arange(0,8,2))
plt.yticks(np.linspace(0, MF.shape[1], 4),np.arange(0,4,1))	

# Charge Modulation
CS = ax.contour(CM.T,colors='red',levels=[0.1,0.3,0.5])
ax.clabel(CS, inline=True, fontsize=10)

# Magnetization
MF_Spin = np.zeros((30,30,2))
MF_Spin[:,:,0]= np.abs(SFM)
MF_Spin[:,:,1]= np.abs(SAFM)

mag = ax.pcolormesh(spin_array_interpreter(MF_Spin).T)
ep.draw_legend(mag,
    titles=[r'$0 0$', r'$\uparrow 0$', r'$\uparrow \downarrow$', r'$\uparrow \uparrow$'],
    classes=[0, 1, 2, 3])

plt.tight_layout()
plt.show()
"""

f, ax = plt.subplots(figsize=(8,5))
ax.set_xlabel('$U/t_{1}$')
ax.set_ylabel('$J/t_{1}$')
plt.xticks(np.linspace(0, MF[0].shape[0], 4),np.arange(0,8,2))
plt.yticks(np.linspace(0, MF[0].shape[1], 4),np.arange(0,4,1))	


# Orbital Disproportionation
OD = np.rint(OD*2 + 2)
im = ax.pcolormesh(OD)
ep.draw_legend(im,
    titles=[r'$\bar{Z}$', r'$\bar{z}$', r'$z$', r'$Z$'],
    classes=[0, 1, 2, 3])

plt.tight_layout()
plt.show()
"""