import numpy as np
import argparse
from scripts.script_cut import one_d_cut

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
    values_list=[np.arange(10, 20, 2)],
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

batch_folder = 'one_d_cuts'

one_d_cut(batch_folder, model_params, params_list, sweeper_args, solver_args)
