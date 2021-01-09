import numpy as np
import Code.Scripts.single_point as sp

model_params = dict(
    N_shape=(50, 50),
    # stress=0,
    # b=0,
    # eps=0,
    # U=4.5,
    # J=1.2
    )

solver_args = dict(
    method='sigmoid',
    beta=2,
    Itteration_limit=50,
    tol=1e-3,
    save_seq=True
    )

batch_folder = 'Single_points'
guesses = np.array([
    [1, 1, 0, 1, 0.15],
    [0, 0, 0, 0, 0]
    ])

sp.point_analysis(model_params, guesses, solver_args, batch_folder)

# # Represenntative settings to show adaptive time steps work
# solver_args_2 = dict(
#     method='momentum',
#     beta=0.5,
#     Itteration_limit=50,
#     tol=1e-3,
#     save_seq=True
#     )


# sp.itteration_comp(model_params, guesses[0], solver_args, solver_args_2)