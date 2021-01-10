import Code.Utils as Utils
import numpy as np
import argparse
import Code.Scripts.diagrams as diagrams
# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--run_ind', type=int, default=0)
args = parser.parse_args()

Batch_Folder = 'Diagrams'
logging = True

sweeper_args = dict(
    variables=['U', 'J'],
    values_list=[np.linspace(0, 1, 30),
                 np.linspace(0, 0.25, 30)],
    bw_norm=True,
    save_guess_mfps=False,
    verbose=True,
    n_threads=args.n_threads
    )

solver_args = dict(
    method='sigmoid',
    beta=1.5,
    Itteration_limit=150,
    tolerance=1e-3,
    tol=1e-3,
    )

model_params = dict(
    N_shape=(100, 100),
    )

params_list = [
    (0.8, 1.0, 0.0, 0.7, 0.15),
    (1.0, 0.6, 0.0, 0.7, 0.15),
    (0.0, 0.2, 0.5, 0.0, 0.2),
    (0.2, 0.5, 1.0, 1.0, 0.0),
    (0.5, 0.5, 0.0, 0.5, 0.1)
]

epsilons = np.linspace(0, 1, 5)
delta_cts = np.linspace(0, 1, 5)
stress = np.linspace(-2.5, 2.5, 5)
fillings = np.linspace(0.25, 0.30, 3)
model_params_lists = Utils.tuplelist([epsilons, delta_cts, stress, fillings])

"""
# Local test
"""
# for i in range(len(model_params_lists)):
#     # Guesses Input
#     args.run_ind = i

hyper = model_params_lists[args.run_ind]
model_params['eps'], model_params['Delta_CT'], model_params['stress'], model_params['Filling'] = hyper

print('Diagram guesses starting')

diagrams.generate_diagram(Batch_Folder, model_params, params_list,
                          sweeper_args, solver_args, logging=logging)

print('Finished succesfully')
