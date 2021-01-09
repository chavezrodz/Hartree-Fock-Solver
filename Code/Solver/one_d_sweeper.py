import numpy as np
import itertools
from multiprocessing import Pool
import Code.Solver.calculations as calc
import numpy.linalg as LA
import os

"""
1D sweeper to track state's energies

"""


class one_d_sweeper:
    """
    """

    def __init__(self, Model, Solver, i, i_values, guesses, n_threads=8, verbose=False, Bandwidth_Normalization=True):
        self.Model = Model
        self.Solver = Solver

        self.n_threads = n_threads
        self.verbose = verbose
        self.guesses = guesses

        self.i = i
        if Bandwidth_Normalization:
            Model = self.Model
            setattr(Model, i, 0)
            setattr(Model, 'MF_params', np.zeros(len(Model.MF_params)))
            Solver = self.Solver
            Solver.Itterate(verbose=False)
            calc.bandwidth(Model)
            if verbose:
                print(f'Fermi_bw: {Model.Fermi_bandwidth}')
            self.i_values = Model.Fermi_bandwidth*i_values
        else:
            self.i_values = i_values

        self.Diag_shape = (len(i_values), len(guesses))
        self.i_idx, self.j_idx = np.indices(self.Diag_shape, sparse=True)
        self.i_idx, self.j_idx = self.i_idx.flatten(), self.j_idx.flatten()

        self.Es_trial = np.zeros(self.Diag_shape)
        self.Final_params = np.zeros(self.Diag_shape + (len(Model.MF_params),))
        self.Convergence_Grid = np.zeros(self.Diag_shape)
        self.MIT = np.zeros(self.Diag_shape)
        self.Distortion = np.zeros(self.Diag_shape)

    def Phase_Diagram_point(self, v, DOS=False):
        Model = self.Model
        Sol = self.Solver

        variable = self.i
        value = self.i_values[v[0]]
        guess = self.guesses[v[1]]

        setattr(Model, variable, value)
        Model.MF_params = guess

        Sol.Itterate(verbose=False)
        calc.post_calculations(Model)
        if self.verbose:
            if Sol.converged:
                print(f'{variable}:{str(value)} Guess:{v[1]} Initial MFP: {guess} Final MFP: {np.round(Model.MF_params,3)} Converged in :{Sol.count} steps')
            else:
                print(f'{variable}:{str(value)} Guess:{v[1]} Initial MFP: {guess} Did Not Converge')
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

        self.best_E = np.min(self.Es_trial, axis=1)
        self.best_E_ind = np.argmin(self.Es_trial, axis=1)
        self.best_params = np.array([self.Final_params[i, j] for i, j in enumerate(self.best_E_ind)])
        # print(self.best_E)
        # print(np.round(self.best_params, 3))

        sol_fermis = []
        for n, sol in enumerate(self.best_params):
            Model = self.Model
            Sol = self.Solver
            setattr(Model, self.i, self.i_values[n])
            Model.MF_params = sol
            Sol.Itterate(verbose=False)
            sol_fermis.append(Model.fermi_e)

        self.fermis = np.array(sol_fermis)

    def save_results(self, outfolder, Include_MFPs=False):
        np.savetxt(os.path.join(outfolder, 'Energies.csv'), self.Es_trial, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'Convergence.csv'), self.Convergence_Grid, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'Conductance.csv'), self.MIT, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'Distortion.csv'), self.Distortion, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'Fermi_Energies.csv'), self.fermis, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'GS_Energy.csv'), self.best_E, delimiter=',')
        if Include_MFPs:
            outfile = os.path.join(outfolder, 'GS_Solutions.csv')
            np.savetxt(outfile, self.best_params, delimiter=",")
            os.makedirs(os.path.join(outfolder, 'MF_Solutions'), exist_ok=True)
            for i in range(self.Final_params.shape[2]):
                outfile = os.path.join(outfolder, 'MF_Solutions', 'MF'+str(i)+'.csv')
                np.savetxt(outfile, self.Final_params[:, :, i], delimiter=",")
