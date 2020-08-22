import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate
from Code.HFA_Solver import *
from Hamiltonians.Hamiltonian_Nickelates import *
from Utils.tuplelist import *
from time import time

Model_Params = dict(
N_Dim = 2,
Nx = 25,
Ny = 25,
Filling = 0.25,
mat_dim = 8,

eps = 0.5,
t_1 = 1,
t_2 = 0.15,
t_4 = 0,
U = 1,
J = 1)

MF_params = np.array([0,0,-1,0])


Model = Hamiltonian(Model_Params, MF_params)

Sol = HFA_Solver(Model,beta = 0.51,Itteration_limit = 50)

Sol.Itterate(tol=1e-3,verbose=True)

Fermi_Energy = Sol.Fermi_Energy
Energies = np.sort(Sol.Energies)

plt.hist(Energies.flatten(),bins='auto')
plt.title('Density of states')
plt.axvline(Fermi_Energy, label='Fermi Energy',color='red')
plt.xlabel('Energy')
plt.legend()
plt.show()
plt.close()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for b in range(Model.mat_dim):
	Q = itertools.product(Model.Qx,repeat=Model.N_Dim)
	qxs=[]
	qys=[]
	zs=[]
	
	for q in Q:
		qxs.append(q[0]*np.pi/Model.Nx - np.pi/2)
		qys.append(q[1]*np.pi/Model.Ny - np.pi/2)
		zs.append(Energies[q][b])
	
	ax.scatter(qxs, qys, zs,label='Band '+str(b+1))
	
xx, yy = np.meshgrid(np.arange(Model.Nx)*np.pi/Model.Nx - np.pi/2, np.arange(Model.Ny)*np.pi/Model.Nx - np.pi/2)
z = np.ones(xx.shape)
z = z*Fermi_Energy
ax.plot_surface(xx, yy,z, alpha=1)

ax.set_xlabel('$K_x$  ($\pi/a$)')
ax.set_ylabel('$K_Y$  ($\pi/a$)')
ax.set_zlabel('Energy')

handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels))
plt.show()
plt.close()




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


