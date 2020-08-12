import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate

# Input arrays of two parameters and phase

x = np.random.randint(10,30,size=10)
y = np.random.randint(10,30,size=10)
z = np.random.randint(10,30,size=10)

xi, yi = np.linspace(x.min(), x.max(), len(x)), np.linspace(y.min(), y.max(), len(y))
xi, yi = np.meshgrid(xi, yi)

rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
zi = rbf(xi, yi)

plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])

plt.scatter(x, y, c=z)

plt.colorbar()
plt.show()