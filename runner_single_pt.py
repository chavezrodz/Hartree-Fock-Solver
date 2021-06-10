import numpy as np
import scripts.script_single_pt as sp


model_params = dict(
    # U=1.2, J=0.6  # Metallic
    # U=3.0, J=1.2  # CD
    # U=4.8, J=0.3  # AFM
    U=4.8, J=0.6  # FM
    , k_res=10
    )


solver_args = dict(
    method='sigmoid',
    beta=3,
    Iteration_limit=250,
    tol=1e-3,
    save_seq=True
    )

batch_folder = 'Single_points'
guesses = np.array([
    (0.8, 1.0, 0.0, 0.7, 0.15),
    (1.0, 0.6, 0.0, 0.7, 0.15),
    (0.0, 0.2, 0.5, 0.0, 0.2),
    (0.2, 0.5, 1.0, 1.0, 0.0),
    (0.5, 0.5, 0.0, 0.5, 0.1)
])

sp.point_analysis(model_params, guesses, solver_args, batch_folder)

# # Represenntative settings to show adaptive time steps work
# model_params = dict(
#     k_res=100,
#     J=1.12)

# solver_args_1 = dict(
#     method='momentum',
#     beta=0.5,
#     Iteration_limit=50,
#     tol=1e-3,
#     save_seq=True
#     )

# solver_args_2 = dict(
#     method='sigmoid',
#     beta=2,
#     Iteration_limit=50,
#     tol=1e-3,
#     save_seq=True
#     )

# sp.iteration_comp(model_params, np.zeros(5), solver_args_1, solver_args_2)