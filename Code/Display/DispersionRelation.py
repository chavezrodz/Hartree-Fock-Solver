from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate
import numpy as np
import itertools

import seaborn as sns
sns.set_theme()
sns.set_context("paper")

def DispersionRelation(Solver,x=0,y=1):
		mat_dim = Solver.Hamiltonian.mat_dim
		Energies = Solver.Energies

		Q = Solver.Hamiltonian.Q
		Qv = Solver.Hamiltonian.Qv

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for b in range(mat_dim):
			qxs=[]
			qys=[]
			zs=[]
			
			for i,q in enumerate(Q):
				qx, qy = Qv[i]
				qxs.append(qx)
				qys.append(qy)
				zs.append(Energies[q][b])
			
			ax.scatter(qxs, qys, zs,label='Band '+str(b+1))
			
		qx_range = np.linspace(np.min(qxs),np.max(qxs),2)
		qy_range = np.linspace(np.min(qys),np.max(qys),2)
		xx, yy = np.meshgrid(qx_range, qy_range)
		z = np.ones(xx.shape)
		z = z*Solver.Fermi_Energy
		ax.plot_surface(xx, yy,z, alpha=0.5)

		ax.set_xlabel('$K_x$  ($\pi/a$)')
		ax.set_ylabel('$K_Y$  ($\pi/a$)')
		ax.set_zlabel('Energy')

		handles, labels = ax.get_legend_handles_labels()
		ax.legend(reversed(handles), reversed(labels))
		plt.show()
		plt.close()

def DOS(Solver):
		plt.hist(Solver.Energies.flatten(),bins='fd')
		plt.title('Density of states')
		plt.axvline(Solver.Fermi_Energy, label='Fermi Energy',color='red')
		plt.xlabel('Energy (Ev)')
		plt.legend()
		plt.show()
		plt.close()
