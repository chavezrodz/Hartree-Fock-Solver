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
			self.Eigenvectors = np.zeros((Ham.N_cells,Ham.N_cells,Ham.mat_dim,Ham.mat_dim),dtype=complex)


		self.N_states = self.Energies.size #Bands x N
		self.N_occ_states = int(Ham.Filling*self.N_states)
		self.occupied_energies = np.zeros(self.N_occ_states)
		self.sub_params = np.zeros((len(Ham.MF_params),self.N_occ_states))

	def Find_filling_lowest_energies(self):
		Initial_shape = self.Energies.shape
		k = self.N_occ_states
		Ec = self.Energies.flatten()
		indices = np.argpartition(Ec,k)[:k]
		indices = np.unravel_index(indices,shape = Initial_shape)
		self.indices_array = indices
		indices = np.transpose(np.stack(indices))
		self.indices = list(map(tuple,indices))

	def Calculate_new_del(self):
		for i,ind in enumerate(self.indices):
			v = self.Eigenvectors[ind[0],ind[1],:,ind[2]]
			self.sub_params[:,i] = self.Hamiltonian.Consistency(v)
		a = self.Hamiltonian.MF_params
		self.Hamiltonian.MF_params = np.sum(self.sub_params,axis=1)
		return a, self.Hamiltonian.MF_params


	def Itteration_Step(self):
		# 	Calculate Dynamic Variables
		self.Hamiltonian.update_variables()
		# Solve Matrix Across all momenta
		Q = itertools.product(self.Hamiltonian.Qx,repeat=self.Hamiltonian.N_Dim)
		for q in Q:
			self.Energies[q],self.Eigenvectors[q] = self.Hamiltonian.Mat_q_calc(q)
		# Find Indices of all required lowest energies
		self.Find_filling_lowest_energies()
		# Calculate Mean Field Parameters with lowest energies
		previous_MFP, New_MFP = self.Calculate_new_del()
		return previous_MFP, New_MFP

	def Itterate(self,tol = 1e-3, verbose = True):
		digits = int(np.abs(np.log10(tol)))

		a,b = self.Itteration_Step()
		count = 1

		if verbose == True:
			print('Initial Mean Field parameters:',a.round(digits))
			print('Itteration:  1  Mean Field parameters:', b.round(digits))

		while LA.norm(a - b) > tol:
			count += 1
			a,b = self.Itteration_Step()
			if verbose ==True:
				print('Itteration: ',count,' Mean Field parameters:', b.round(digits))
		if verbose == True:
			print('Final Mean Field parameter:', b.round(digits), '\nNumber of itteration steps:', count)

		for i,ind in enumerate(self.indices):
			self.occupied_energies[i] = self.Energies[ind]
		self.total_occupied_energy =  np.sum(self.occupied_energies)