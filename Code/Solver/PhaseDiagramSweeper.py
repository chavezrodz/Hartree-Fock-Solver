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

	def __init__(self, Model, Solver, Initial_params, i, i_values, j, j_values, n_threads=8,verbose = False):
		self.Model = Model
		self.Solver = Solver
		

		self.n_threads = n_threads
		self.verbose = verbose

		self.i = i
		self.i_values = i_values

		self.j = j
		self.j_values = j_values

		self.Diag_shape = (len(i_values),len(i_values))

		self.i_idx,self.j_idx = np.indices(self.Diag_shape,sparse=True)
		self.i_idx,self.j_idx = self.i_idx.flatten(),self.j_idx.flatten()

		if Initial_params.ndim == 1:
			self.Initial_params = np.zeros((*self.Diag_shape,len(Model.MF_params)))
			self.Initial_params[:,:,:] = Initial_params
		else:
			self.Initial_params = Initial_params

		self.Es_trial = np.zeros(self.Diag_shape)
		self.Final_params = np.zeros(self.Initial_params.shape)
		self.Convergence_Grid = np.zeros(self.Diag_shape)
		self.MIT = np.zeros(self.Diag_shape)

	def Phase_Diagram_point(self,v):
		Model = self.Model
		Sol = self.Solver
		setattr(Model, self.i, self.i_values[v[0]])
		setattr(Model, self.j, self.j_values[v[1]])

		Model.MF_params = self.Initial_params[v]

		Sol.Itterate(verbose=False)

		if self.verbose:
			if Sol.converged:
				print(self.i,round(getattr(Model,self.i),2),self.j, round(getattr(Model,self.j),2),'Initial MFP:',np.round(self.Initial_params[v],3), 'Final MFP:',np.round(Model.MF_params,3), 'Converged in :',Sol.count,'steps')
			else:
				print(self.i,round(getattr(Model,self.i),2),self.j, round(getattr(Model,self.j),2),'Initial MFP:',np.round(self.Initial_params[v],3), 'Did Not Converge')

		return Sol.Final_Total_Energy, Model.MF_params, Sol.converged, Sol.Conductor

	def Sweep(self):

		# MP way
		PD_grid = itertools.product(self.i_idx,self.j_idx)
		with Pool(self.n_threads) as p:
			results = p.map(self.Phase_Diagram_point, PD_grid)

		# Energies results list to array to csv
		PD_grid = itertools.product(self.i_idx,self.j_idx)
		for i,v in enumerate(PD_grid):
			self.Es_trial[v] = results[i][0]
			self.Final_params[v] = results[i][1]
			self.Convergence_Grid[v] = results[i][2]
			self.MIT[v] = results[i][3]

		self.MIT = self.MIT.astype(int)
		self.Convergence_Grid = self.Convergence_Grid.astype(int)
		self.Convergence_pc = 100*np.mean(self.Convergence_Grid)

	def save_results(self, outfolder, Include_MFPs=False):
		np.savetxt(os.path.join(outfolder,'Energies.csv'),self.Es_trial,delimiter=',')
		np.savetxt(os.path.join(outfolder,'Convergence_Grid.csv'),self.Convergence_Grid,delimiter=',')
		np.savetxt(os.path.join(outfolder,'Conductance_Grid.csv'),self.MIT,delimiter=',')
		if Include_MFPs:
			if not os.path.exists(os.path.join(outfolder,'MF_Solutions')):
				os.makedirs(os.path.join(outfolder,'MF_Solutions'))
			for i in range(self.Initial_params.shape[2]):
				outfile = os.path.join(outfolder,'MF_Solutions','MF'+str(i)+'.csv')
				np.savetxt(outfile,self.Final_params[:,:,i],delimiter=",")
