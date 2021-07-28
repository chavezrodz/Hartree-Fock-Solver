import copy
import solver.calculations as calc
import numpy as np
import itertools
from multiprocessing import Pool
import os


class Phase_Diagram_Sweeper():
    """
    """

    def __init__(self, Model, Solver, Initial_params,
                 fermi_bw=None, verbose=False, **kwargs):

        self.Model = Model
        self.Solver = Solver
        self.verbose = verbose

        self.__dict__.update(kwargs)
        self.i = self.variables[0]
        self.j = self.variables[1]
        self.i_values, self.j_values = self.values_list

        if fermi_bw is not None:
            print(f'Fermi_bw: {fermi_bw}')
            self.i_values = fermi_bw*self.i_values
            self.j_values = fermi_bw*self.j_values

        self.Diag_shape = (len(self.i_values), len(self.j_values))

        self.i_idx, self.j_idx = np.indices(self.Diag_shape, sparse=True)
        self.i_idx, self.j_idx = self.i_idx.flatten(), self.j_idx.flatten()

        if Initial_params.ndim == 1:
            self.Initial_params = np.zeros((*self.Diag_shape, len(Model.MF_params)))
            self.Initial_params[:, :, :] = Initial_params
        else:
            self.Initial_params = Initial_params

        self.Es_trial = np.zeros(self.Diag_shape)
        self.Final_params = np.zeros(self.Initial_params.shape)
        self.Convergence_Grid = np.zeros(self.Diag_shape)
        self.MIT = np.zeros(self.Diag_shape)
        self.Distortion = np.zeros(self.Diag_shape)

    def Phase_Diagram_point(self, v):
        Sol = copy.deepcopy(self.Solver)
        Model = Sol.Hamiltonian

        variable1 = self.i
        value1 = self.i_values[v[0]]
        setattr(Model, variable1, value1)

        variable2 = self.j
        value2 = self.j_values[v[1]]
        setattr(Model, variable2, value2)

        Model.MF_params = self.Initial_params[v]

        Sol.Iterate(verbose=False)
        calc.post_calculations(Model)
        if self.verbose:
            if Sol.converged:
                print(f'{self.i}:{getattr(Model,self.i):1.2f} {self.j}:{getattr(Model,self.j):1.2f} Initial MFP: {np.round(self.Initial_params[v],3)} Final MFP: {np.round(Model.MF_params,3)} Converged in :{Sol.count} steps')
            else:
                print(f'{self.i}:{getattr(Model,self.i):1.2f} {self.j}:{getattr(Model,self.j):1.2f} Initial MFP: {np.round(self.Initial_params[v],3)} Did Not Converge')
        return Model.Final_Total_Energy, Model.MF_params, Model.converged, Model.Conductor, Model.u

    def Sweep(self):

        # MP way
        PD_grid = itertools.product(self.i_idx, self.j_idx)
        with Pool(self.n_threads) as p:
            results = p.map(self.Phase_Diagram_point, PD_grid)

        # Energies results list to array to csv
        PD_grid = itertools.product(self.i_idx, self.j_idx)
        for i, v in enumerate(PD_grid):
            self.Es_trial[v] = results[i][0]
            self.Final_params[v] = results[i][1]
            self.Convergence_Grid[v] = results[i][2]
            self.MIT[v] = results[i][3]
            self.Distortion[v] = results[i][4]

        self.MIT = self.MIT.astype(int)
        self.Convergence_Grid = self.Convergence_Grid.astype(int)
        self.Convergence_pc = 100*np.mean(self.Convergence_Grid)

    def save_results(self, outfolder, Include_MFPs=False):
        np.savetxt(os.path.join(outfolder, 'Energies.csv'),
                   self.Es_trial, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'Convergence.csv'),
                   self.Convergence_Grid, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'Conductance.csv'),
                   self.MIT, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'Distortion.csv'),
                   self.Distortion, delimiter=',')
        if Include_MFPs:
            os.makedirs(os.path.join(outfolder, 'MF_Solutions'), exist_ok=True)
            for i in range(self.Initial_params.shape[2]):
                outfile = os.path.join(
                    outfolder, 'MF_Solutions', 'MF'+str(i)+'.csv'
                    )
                np.savetxt(outfile, self.Final_params[:, :, i], delimiter=",")
