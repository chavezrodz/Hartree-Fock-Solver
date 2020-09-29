import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.ndimage

def Optimizer_smoothing(mfps,sigma=[1,1]):
	y = np.zeros(mfps.shape)
	for i in range(mfps.shape[2]):
		y[:,:,i] = sp.ndimage.filters.gaussian_filter(mfps[:,:,i], sigma, mode='nearest')
	return y

"""
x = np.random.rand(20,20,2)
# Plot input array
for i in range(2):
	plt.imshow(x[:,:,i], cmap='Blues')
	plt.show()
	# Apply gaussian filter
	y = Optimizer_smoothing(x)

	# Display filtered array
	plt.imshow(y[:,:,i], cmap='Blues')
	plt.show()

"""