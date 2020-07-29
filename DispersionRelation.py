import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate


from main import *

x = HFA.indices_array[0]
y = HFA.indices_array[1]
z = HFA.occupied_energies

plt.scatter(x, y, c=z
	, marker="s"
	)
# plt.colorbar()
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(xs= x, ys=y, zs =z )