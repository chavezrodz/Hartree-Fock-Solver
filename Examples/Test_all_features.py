import numpy as np
import Code.Scripts.single_point as sp
import Code.Scripts.diagrams as diagrams
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--run_ind', type=int, default=0)
args = parser.parse_args()

# Single Points
model_params = dict(
    N_shape=(10, 10)
    )

solver_args = dict(
    method='sigmoid',
    beta=2,
    Iteration_limit=50,
    tol=1e-3,
    save_seq=True
    )

batch_folder = 'General Test'

guesses = np.array([
    [1, 1, 0, 1, 0.15],
    [0, 0, 0, 0, 0]
    ])

sp.point_analysis(model_params, guesses, solver_args, batch_folder)

# # Represenntative settings to show adaptive time steps work
solver_args_2 = dict(
    method='momentum',
    beta=0.5,
    Itteration_limit=50,
    tol=1e-3,
    save_seq=True
    )


sp.itteration_comp(model_params, guesses[0], solver_args, solver_args_2, batch_folder)

# Phase Diagram
sweeper_args = dict(
    variables=['U', 'J'],
    values_list=[np.linspace(0, 1, 10),
                 np.linspace(0, 0.25, 10)],
    bw_norm=True,
    save_guess_mfps=True,
    verbose=True,
    n_threads=args.n_threads
    )

params_list = [
    (0.8, 1.0, 0.0, 0.7, 0.15),
    (1.0, 0.6, 0.0, 0.7, 0.15),
    (0.0, 0.2, 0.5, 0.0, 0.2),
    (0.2, 0.5, 1.0, 1.0, 0.0),
    (0.5, 0.5, 0.0, 0.5, 0.1)
]

logging = True

diagrams.generate_diagram(batch_folder, model_params, params_list,
                          sweeper_args, solver_args, logging=logging)

# One dimension cut
i = 'stress'
i_values = np.linspace(-2, 2, 10)


