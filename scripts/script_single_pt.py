import shutil
import os
import numpy as np
from solver.HFA_Solver import HFA_Solver
import display.single_point.iteration_seq as IS
import display.single_point.dispersion as DR
import display.single_point.dos as DOS
import solver.calculations as calc
from models.Nickelates.Hamiltonian import Hamiltonian


def point_analysis(model_params, guesses, solver_args, batch_folder, show=False, HFA=True):
    point_Id = '_'.join("{!s}={!r}".format(key, val)
                        for (key, val) in model_params.items())

    if HFA:
        Energies = np.zeros(len(guesses))
        for i, MF_params in enumerate(guesses):

            Model = Hamiltonian(model_params, MF_params)
            Solver = HFA_Solver(Model, **solver_args)

            guess_ID = os.path.join(point_Id, str(MF_params))
            results_folder = os.path.join('Results', batch_folder, guess_ID)

            os.makedirs(results_folder, exist_ok=True)

            Solver.Iterate(verbose=True)
            IS.Iteration_sequence(Solver, results_folder=results_folder, show=show)
            calc.post_calculations(Model)
            DOS.DOS(Model, results_folder=results_folder, show=show)
            DOS.DOS_per_state(Model, results_folder=results_folder, show=show)
            DR.fermi_surface(Model, results_folder=results_folder, show=show)
            if show:
                DR.DispersionRelation(Model)

            calc.bandstructure(Model)
            DR.Bandstructure(Model, results_folder=results_folder, show=show)

            Energies[i] = Model.Final_Total_Energy
        min_arg = np.argmin(Energies)
        GS_guess = guesses[min_arg]
        print("Ground State Guess:", GS_guess)
        guess_ID = os.path.join(point_Id, str(GS_guess))
        shutil.move(
            os.path.join('Results', batch_folder, guess_ID),
            os.path.join('Results', batch_folder, guess_ID+'_GS')
            )

    else:
        Model = Hamiltonian(model_params)
        results_folder = os.path.join('Results', batch_folder, point_Id)
        os.makedirs(results_folder, exist_ok=True)
        calc.bandstructure(Model)
        DR.Bandstructure(Model, fermi=False, results_folder=results_folder, show=show)


def iteration_comp(model_params, mf_params, solver_args_1, solver_args_2):
    Model = Hamiltonian(model_params, mf_params)
    Solver = HFA_Solver(Model, **solver_args_1)
    Solver.Iterate(verbose=False)

    Solver_1 = Solver

    Model = Hamiltonian(model_params, mf_params)
    Solver = HFA_Solver(Model, **solver_args_2)
    Solver.Iterate(verbose=False)

    Solver_2 = Solver

    IS.Iteration_comparison(Solver_1, Solver_2)
