import os
from time import time
import numpy as np
from Code.Solver.HFA_Solver import HFA_Solver
import Code.Display.Itteration_sequence as IS
import Code.Display.DispersionRelation as DR
import Code.Display.Density_Of_States as DOS
import Code.Solver.calculations as calc
from Code.Nickelates.Hamiltonian import Hamiltonian
# from Code.Graphene.Hamiltonian import Hamiltonian


def point_analysis(model_params, guesses, solver_args, batch_folder, transparent=False, show=False, HFA=True):
    point_Id = '_'.join("{!s}={!r}".format(key, val)
                        for (key, val) in model_params.items())

    if HFA:
        for MF_params in guesses:
            Model = Hamiltonian(model_params, MF_params)
            Solver = HFA_Solver(Model, **solver_args)

            guess_ID = os.path.join(point_Id, str(MF_params))
            results_folder = os.path.join('Results', batch_folder, guess_ID)

            os.makedirs(results_folder, exist_ok=True)

            Solver.Itterate(verbose=True)
            IS.Itteration_sequence(Solver, results_folder=results_folder, show=show)
            calc.post_calculations(Model)

            DOS.DOS(Model, results_folder=results_folder, show=show)
            DOS.DOS_per_state(Model, results_folder=results_folder, show=show)

            DR.fermi_surface(Model, results_folder=results_folder, show=show)
            # calc.bandwidth(Model)

            if show:
                DR.DispersionRelation(Model)

            calc.bandstructure(Model)
            DR.Bandstructure(Model, results_folder=results_folder, show=show)
    else:
        Model = Hamiltonian(model_params)
        results_folder = os.path.join('Results', batch_folder, point_Id)
        os.makedirs(results_folder, exist_ok=True)
        calc.bandstructure(Model)
        DR.Bandstructure(Model, fermi=False, results_folder=results_folder, show=show)


def itteration_comp(model_params, mf_params, solver_args_1, solver_args_2):
    Model = Hamiltonian(model_params, mf_params)
    Solver = HFA_Solver(Model, **solver_args_1)
    Solver.Itterate(verbose=False)

    Solver_1 = Solver

    Model = Hamiltonian(model_params, mf_params)
    Solver = HFA_Solver(Model, **solver_args_2)
    Solver.Itterate(verbose=False)

    Solver_2 = Solver

    IS.Itteration_comparison(Solver_1, Solver_2)