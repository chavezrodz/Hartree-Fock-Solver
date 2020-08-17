import numpy as np
import itertools
import sys
from numpy import linalg as LA
from multiprocessing import Process
from itertools import product
from multiprocessing import Pool
from time import time
import os

class Phase_Diagram_Sweeper():
	"""
	"""

	def __init__(self, Model, Solver, Initial_params, U_values, J_values, n_threads=8):
		self.Model = Model
		self.Solver = Solver
		
		self.U_values = U_values
		self.J_values = J_values
		self.U_idx = np.arange(len(U_values))
		self.J_idx = np.arange(len(J_values))

		self.n_threads = n_threads
		
		if Initial_params.ndim == 1:
			self.Initial_params = np.zeros((len(self.U_values),len(self.J_values),len(Model.MF_params)))
			self.Initial_params[:,:,:] = Initial_params
		else:
			self.Initial_params = Initial_params

		self.Es_trial = np.zeros((len(self.U_values),len(self.J_values)))
		self.Final_params = np.zeros(self.Initial_params.shape)

	def Phase_Diagram_point(self,v):
		Model = self.Model
		Sol = self.Solver

		Model.U, Model.J = self.U_values[0],self.J_values[1]
		Model.MF_params = self.Initial_params[v]

		Sol.Itterate(tol=1e-3,verbose=False)

		Model.E_occ = Sol.total_occupied_energy

		Model.Calculate_Energy()

		self.Es_trial[v] = Model.Final_Total_Energy
		self.Final_params[v] = Model.MF_params
		return Model.Final_Total_Energy, Model.MF_params

	def Sweep(self, outfolder, fname ='', Final_Run=False):

		PD_grid = itertools.product(self.U_idx,self.J_idx)
		with Pool(self.n_threads) as p:
			results = p.map(self.Phase_Diagram_point, PD_grid)

		# Energies results list to array to csv
		PD_grid = itertools.product(self.U_idx,self.J_idx)
		for i,v in enumerate(PD_grid):
			self.Es_trial[v] = results[i][0]
			self.Final_params[v] = results[i][1]

		# for v in PD_grid:
			# self.Phase_Diagram_point(v)

		if Final_Run == False:
			outfile = os.path.join(outfolder,fname)
			np.savetxt(outfile,self.Es_trial,delimiter=',')

		if Final_Run == True:
			outfile = os.path.join(outfolder,'Energies.csv')
			np.savetxt(outfile,self.Es_trial,delimiter=',')

			for i in range(self.Initial_params.shape[2]):
				outfile = os.path.join(outfolder,'MF_Solutions','MF'+str(i)+'.csv')
				np.savetxt(outfile,self.Final_params[:,:,i],delimiter=",")
