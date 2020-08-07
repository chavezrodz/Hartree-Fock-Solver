import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate

from Hamiltonian_8 import *

# model parameters
Model_Params = dict(
N_Dim = 2,
N_cells = 25,
Filling = 0.25,
mat_dim = 8,

eps = 1,
t_1 =5,
U = 5,
J = 1)

MF_params = np.array([1,1,1])


Model = Hamiltonian(Model_Params, MF_params)

Sol = HFA_Solver(Model)

Sol.Itterate(tol=1e-3,verbose=False)

Energies = Sol.Energies

Energies = np.sort(Energies)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for b in range(Model.mat_dim):
	Q = itertools.product(Model.Qx,repeat=Model.N_Dim)
	qxs=[]
	qys=[]
	zs=[]
	
	for q in Q:
		qxs.append(q[0]*np.pi/Model.N_cells - np.pi/2)
		qys.append(q[1]*np.pi/Model.N_cells - np.pi/2)
		zs.append(Energies[q][b])
	
	ax.scatter(qxs, qys, zs,label='band '+str(b+1))
	
ax.set_xlabel('$K_x$  ($\pi/a$)')
ax.set_ylabel('$K_Y$  ($\pi/a$)')
ax.set_zlabel('Energy')
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels))
plt.show()


'''
# Fermi Surface
x = Sol.indices_array[0]*np.pi/Model.N_cells - np.pi/2
y = Sol.indices_array[1]*np.pi/Model.N_cells - np.pi/2
z = Sol.occupied_energies

plt.scatter(x, y, c=z
	, marker="s"
	)
plt.title('Fermi Surface')
plt.xlabel('$K_x$  ($\pi/a$)')
plt.ylabel('$K_Y$  ($\pi/a$)')
plt.colorbar()
plt.show()
'''


