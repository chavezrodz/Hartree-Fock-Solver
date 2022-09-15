import time
import numpy as np
import itertools
import display.single_point.dispersion as DR
import display.single_point.dos as DOS
import solver.calculations as calc
from multiprocessing import Pool
# from scripts.single_point import point_analysis as point_analysis
import copy
import os

"""
1D sweeper to track state's energies

"""


class one_d_sweeper:
    """
    """

    def __init__(self, Model, Solver, guesses, Results_folder,
                 bw_norm=False, n_threads=8, verbose=False,
                 **kwargs):
        self.Model = Model
        self.Solver = Solver
        self.guesses = guesses

        self.n_threads = n_threads
        self.verbose = verbose
        self.Results_folder = Results_folder

        self.i = kwargs['variables'][0]
        i_values = kwargs['values_list'][0]
        if bw_norm:
            Sol = copy.deepcopy(self.Solver)
            Model = Sol.Hamiltonian
            setattr(Model, self.i, 0)
            setattr(Model, 'MF_params', np.zeros(len(Model.MF_params)))
            Sol.Iterate(verbose=False)
            calc.bandwidth(Model)
            if verbose:
                print(f'Fermi_bw: {Model.fermi_bw}')
            self.i_values = Model.fermi_bw*i_values
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
        Sol = copy.deepcopy(self.Solver)
        Model = Sol.Hamiltonian

        variable = self.i
        value = self.i_values[v[0]]
        guess = self.guesses[v[1]]

        setattr(Model, variable, value)
        Model.MF_params = guess

        Sol.Iterate(verbose=False)
        calc.post_calculations(Model)
        if self.verbose:
            if Sol.converged:
                print(f'{variable}:{str(value)} Guess:{v[1]} Initial MFP: {guess} Final MFP: {np.round(Model.MF_params,3)} Converged in :{Sol.count} steps')
            else:
                print(f'{variable}:{str(value)} Guess:{v[1]} Initial MFP: {guess} Did Not Converge')
        return Model.Final_Total_Energy, Model.MF_params, Model.converged, Model.Conductor, Model.u

    def Sweep(self, exhaustive=False):

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

        if exhaustive:
            BS_folder = os.path.join(self.Results_folder, 'Bandstructure')
            DOS_folder = os.path.join(self.Results_folder, 'DOS')
            DOS_state = os.path.join(self.Results_folder, 'DOS_state')
            DOS_single = os.path.join(self.Results_folder, 'DOS_single')
            os.makedirs(BS_folder, exist_ok=True)
            os.makedirs(DOS_folder, exist_ok=True)
            os.makedirs(DOS_state, exist_ok=True)
            os.makedirs(DOS_single, exist_ok=True)

            for n, j in enumerate(self.i_values):
                Sol = copy.deepcopy(self.Solver)
                Model = Sol.Hamiltonian

                show = False
                variable = self.i
                value = j
                guess = self.best_params[n]

                label = str(variable)+'_'+str(value)

                setattr(Model, variable, value)
                Model.MF_params = guess

                Sol.Iterate(verbose=False)
                calc.post_calculations(Model)

                DOS.DOS(Model, results_folder=DOS_folder, label=label, show=show)
                DOS.DOS_per_state(Model, results_folder=DOS_state, label=label, show=show)
                DOS.DOS_single_state(Model, ind=1, results_folder=DOS_single, label=label, show=show)
                DOS.DOS_single_state(Model, ind=3, results_folder=DOS_single, label=label, show=show)
                calc.bandstructure(Model)
                DR.Bandstructure(Model, results_folder=BS_folder, label=label, show=show)

    def save_results(self, Include_MFPs=False):
        outfolder = self.Results_folder
        np.savetxt(os.path.join(outfolder, 'Energies.csv'), self.Es_trial, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'Convergence.csv'), self.Convergence_Grid, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'Conductance.csv'), self.MIT, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'Distortion.csv'), self.Distortion, delimiter=',')
        np.savetxt(os.path.join(outfolder, 'GS_Energy.csv'), self.best_E, delimiter=',')
        if Include_MFPs:
            outfile = os.path.join(outfolder, 'GS_Solutions.csv')
            np.savetxt(outfile, self.best_params, delimiter=",")
            os.makedirs(os.path.join(outfolder, 'MF_Solutions'), exist_ok=True)
            for i in range(self.Final_params.shape[2]):
                outfile = os.path.join(outfolder, 'MF_Solutions', 'MF'+str(i)+'.csv')
                np.savetxt(outfile, self.Final_params[:, :, i], delimiter=",")
