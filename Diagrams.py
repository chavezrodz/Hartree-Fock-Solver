from itertools import product as product
import numpy as np
import argparse
import Code.Scripts.diagrams as diagrams

# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--run_ind', type=int, default=0)
args = parser.parse_args()

batch_folder = 'diagrams'
logging = False

sweeper_args = dict(
    variables=['U', 'J'],
    values_list=[np.linspace(0, 1, 3),
                 np.linspace(0, 0.2, 3)],
    bw_norm=True,
    save_guess_mfps=True,
    verbose=True,
    n_threads=args.n_threads
    )

solver_args = dict(
    method='sigmoid',
    beta=1.5,
    Iteration_limit=250,
    tol=1e-3,
    )


params_list = [
    (0.8, 1.0, 0.0, 0.7, 0.15),
    (1.0, 0.6, 0.0, 0.7, 0.15),
    (0.0, 0.2, 0.5, 0.0, 0.2),
    # (0.2, 0.5, 1.0, 1.0, 0.0),
    # (0.5, 0.5, 0.0, 0.5, 0.1)
]

hyper_params = {
    'eps': np.linspace(0, 0.1, 3),
    'Delta_CT': np.linspace(0, 0.1, 3),
    'stress': [-1, 1],
}

keys, values = zip(*hyper_params.items())
combinations = list(product(*values))

"""
# Local test
for i in range(len(combinations)):
    # Guesses Input
    args.run_ind = i
"""
model_params = dict(zip(keys, combinations[args.run_ind]))
model_params.update({'k_res': 10})


print('Diagram guesses starting')

diagrams.generate_diagram(batch_folder, model_params, params_list,
                          sweeper_args, solver_args, bw_norm=True, logging=logging)

print('Finished succesfully')
