import numpy as np
import os
import sys
import utils as U

from models.Nickelates.Hamiltonian import Hamiltonian
from solver.HFA_Solver import HFA_Solver
from solver.one_d_sweeper import one_d_sweeper
from display.meta.cut import one_d_plots

# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
args = parser.parse_args()

def one_d_cut(batch_folder, model_params, params_list, sweeper_args, solver_args):
    Run_ID = U.make_id(sweeper_args, model_params)
    results_folder = os.path.join('Results', batch_folder, Run_ID)
    os.makedirs(results_folder, exist_ok=True)

    U.write_settings(Run_ID, results_folder, [model_params, solver_args, sweeper_args])

    Model = Hamiltonian(model_params)
    Solver = HFA_Solver(Model, **solver_args)
    Sweeper = one_d_sweeper(Model, Solver, params_list, results_folder, **sweeper_args)
    Sweeper.Sweep()
    Sweeper.save_results(Include_MFPs=True)
    one_d_plots(*sweeper_args['variables'], *sweeper_args['values_list'], Model.Dict, params_list, results_folder)

    print('One Dimensional Cut Done')