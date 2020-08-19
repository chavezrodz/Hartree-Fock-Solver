import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate

# Input arrays of two parameters and phase

# MF = np.zeros((3,20,20))

# for i in range(3):
# 	MF[i,:,:] = np.loadtxt('../Results/Final_Results/MF_Solutions/MF'+str(i)+'.csv',delimiter=",")

MF = np.random.rand(3,20,20)

# print(MF)

MF = np.rint(MF)
for i in range(3):
	plt.imshow(MF[i])
	plt.colorbar()
	plt.show()
