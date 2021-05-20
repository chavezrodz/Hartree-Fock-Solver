import Code.Utils as U
import numpy as np
import os
import argparse
import sys
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.one_d_sweeper import one_d_sweeper
from Code.Display.ResultsPlots import one_d_plots
# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
args = parser.parse_args()


params_list = [
    (1, 1, 0, 1, 0.15),
    (1, 0.5, 0, 1, 0.15),
    (0, 0.2, 0.5, 0, 0),
    (0.1, 0.5, 1, 0.5, 0.1),
    (0.5, 0.5, 0, 0.5, 0.1),
    (0.5, 0.5, 0.5, 0.5, 0.5)
    ]

sweeper_args = dict(
    variables=['k_res'],
    values_list=[np.arange(10, 200, 10)],
    bw_norm=False,
    verbose=True,
    save_guess_mfps=True,
    n_threads=args.n_threads
    )

solver_args = dict(
    method='sigmoid',
    beta=1.5,
    Iteration_limit=150,
    tolerance=1e-3,
    tol=1e-3,
    )

model_params = dict(
    # U=6,
    # J=1.5
)


# def one_d_cut(batch_folder, model_params, params_list, sweeper_args,
#               solver_args, guess_run=True, final_run=True, rm_guesses=True, logging=True):
#     pass


Batch_folder = 'one_d_cuts'

Run_ID = U.make_id(sweeper_args, model_params)
results_folder = os.path.join('Results', Batch_folder, Run_ID)
os.makedirs(results_folder, exist_ok=True)

U.write_settings(Run_ID, results_folder, model_params, solver_args, sweeper_args)

Model = Hamiltonian(model_params)
Solver = HFA_Solver(Model, **solver_args)
sweeper = one_d_sweeper(Model, Solver, params_list, results_folder, **sweeper_args)

sweeper.Sweep()
sweeper.save_results(Include_MFPs=True)
one_d_plots(*sweeper_args['variables'], *sweeper_args['values_list'], Model.Dict, params_list, results_folder)

print('done')
