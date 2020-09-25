import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.ndimage

n = 200 # widht/height of the array
m = 1000 # number of points


def Opimizer_smoothing(mfps,sigma=[1,1]):
	dummy = mfps[:,:,0]
	y = sp.ndimage.filters.gaussian_filter(x, sigma, mode='nearest')

# Plot input array
plt.imshow(x, cmap='Blues')
plt.show()
# Apply gaussian filter

# Display filtered array
plt.imshow(y, cmap='Blues')
plt.show()