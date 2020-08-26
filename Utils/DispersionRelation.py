import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate
from Code.HFA_Solver import *
from Hamiltonians.Hamiltonian_Nickelates import *
from Utils.tuplelist import *
from time import time

class  Dispersion_Relation():

	def __init__(self,Model, Solver):
		self.Model = Model
		self.Sol = Solver
	
		self.Fermi_Energy = Sol.Fermi_Energy
		self.Energies = np.sort(Sol.Energies)

	def Density_of_States(self):

		plt.hist(self.Energies.flatten(),bins='auto')
		plt.title('Density of states')
		plt.axvline(self.Fermi_Energy, label='Fermi Energy',color='red')
		plt.xlabel('Energy (Ev)')
		plt.legend()
		plt.show()
		plt.close()


	def Dispersion_Relation(self):


		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		for b in range(self.Model.mat_dim):
			Q = itertools.product(self.Model.Qx,self.Model.Qy)
			qxs=[]
			qys=[]
			zs=[]
			
			for q in Q:
				qxs.append(q[0]*np.pi/Model.Nx - np.pi/2)
				qys.append(q[1]*np.pi/Model.Ny - np.pi/2)
				zs.append(Energies[q][b])
			
			ax.scatter(qxs, qys, zs,label='Band '+str(b+1))
			
		xx, yy = np.meshgrid(np.arange(self.Model.Nx)*np.pi/self.Model.Nx - np.pi/2, np.arange(self.Model.Ny)*np.pi/self.Model.Ny - np.pi/2)
		z = np.ones(xx.shape)
		z = z*self.Fermi_Energy
		ax.plot_surface(xx, yy,z, alpha=1)

		ax.set_xlabel('$K_x$  ($\pi/a$)')
		ax.set_ylabel('$K_Y$  ($\pi/a$)')
		ax.set_zlabel('Energy')

		handles, labels = ax.get_legend_handles_labels()
		ax.legend(reversed(handles), reversed(labels))
		plt.show()
		plt.close()
