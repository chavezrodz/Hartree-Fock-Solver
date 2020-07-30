import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import itertools

class HFA_Solver:
	"""
	Structural Parameters
	Model_params: Hamiltonian Parameters, must include Filling(float)
	MFP_params: initial guesses for Mean Free Parameters
	"""
	def __init__(self, Ham):
		self.Hamiltonian = Ham

		if Ham.N_Dim == 2:
			self.Energies = np.zeros((Ham.N_cells,Ham.N_cells,Ham.mat_dim))
			self.Eigenvectors = np.zeros((Ham.N_cells,Ham.N_cells,Ham.mat_dim,Ham.mat_dim))


		self.n_states = self.Energies.size #Bands x N
		self.occ_states = int(Ham.Filling*self.n_states)
		self.occupied_energies = np.zeros(self.occ_states)
		self.sub_params = np.zeros((len(Ham.MF_params),self.occ_states))

	def Find_filling_lowest_energies(self):
		Initial_shape = self.Energies.shape
		k = self.occ_states
		Ec = self.Energies.flatten()
		indices = np.argpartition(Ec,k)[:k]
		indices = np.unravel_index(indices,shape = Initial_shape)
		self.indices_array = indices
		indices = np.transpose(np.stack(indices))
		self.indices = list(map(tuple,indices))

	def Calculate_new_del(self):
		for i,ind in enumerate(self.indices):
			v = self.Eigenvectors[ind]
			self.sub_params[:,i] = self.Hamiltonian.Consistency(v)
		a = self.Hamiltonian.MF_params
		self.Hamiltonian.MF_params = np.mean(self.sub_params,axis=1)
		return a, self.Hamiltonian.MF_params

	def Itterate(self,tol = 1e-3, verbose = True):
		Q = itertools.product(self.Hamiltonian.Qx,repeat=self.Hamiltonian.N_Dim)
		for q in Q:
			self.Energies[q],self.Eigenvectors[q] = self.Hamiltonian.Mat_q_calc(q)

		self.Find_filling_lowest_energies()
		a,b = self.Calculate_new_del()
		count = 1
		print('Initial Mean Field parameters:',a)
		print('Itteration:  1  Mean Field parameters:', b)
		while LA.norm(a - b) > tol:
			count += 1
			Q = itertools.product(self.Hamiltonian.Qx,repeat=self.Hamiltonian.N_Dim)
			for q in Q:
				self.Energies[q],self.Eigenvectors[q] = self.Hamiltonian.Mat_q_calc(q)
			self.Find_filling_lowest_energies()
			a, b = self.Calculate_new_del()
			if verbose ==True:
				print('Itteration: ',count,' Mean Field parameters:', b)
		print('Final Mean Field parameter:', b, '\nNumber of itteration steps:', count)

	# def Calculate_occupied_Energy(self):

	# 	for i,ind in enumerate(self.indices):
	# 		# print(i,ind)
	# 		self.occupied_energies[i] = self.Energies[ind]

	# 	total_occupied_energy_states = np.sum(self.occupied_energies)
	# 	return total_occupied_energy_states


	# def Calculate_Total_Energy(self):
	# 	""" Calculate total energy from dispersion relation integral and 
	# 	semiclassical energies
	# 	""" 
	# 	Disp_indp_term = self.N * self.eps**2 / (2*self.k_spring)
	# 	Disp_indp_term *= np.square(self.MFP)
	# 	TE = Disp_indp_term + self.Calculate_occupied_Energy()
	# 	return TE