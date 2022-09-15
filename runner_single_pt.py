import numpy as np
import scripts.script_single_pt as sp


run_single_point = True
run_iteration_comp = False

# Single point
point = 'AFM'
k_res = 64
batch_folder = 'Single_points'

solver_args = dict(
    method='sigmoid',
    beta=3,
    Iteration_limit=250,
    tol=1e-3,
    save_seq=True,
    verbose=False
    )

trials = np.array([
    (0.8, 1.0, 0.0, 0.7, 0.15),
    (1.0, 0.6, 0.0, 0.7, 0.15),
    (0.0, 0.2, 0.5, 0.0, 0.2),
    (0.2, 0.5, 1.0, 1.0, 0.0),
    (0.5, 0.5, 0.0, 0.5, 0.1),
])

if point == 'AFM':
    model_params = dict(U=4.8, J=0.3)
elif point == 'Metallic':
    model_params = dict(U=1.2, J=0.6)
elif point == 'FM':
    model_params = dict(U=4.8, J=0.6)
elif point == 'CD':
    model_params = dict(U=3.0, J=1.2)
else:
    raise Exception('Model paramters not found')
model_params.update({'k_res': k_res})

# Representative settings to show adaptive time steps work

# static mixing failure
model_params = dict(J=1.2, k_res=100)

solver_args_1 = dict(
    method='momentum',
    beta=0.5,
    Iteration_limit=50,
    tol=1e-3,
    save_seq=True
    )

solver_args_2 = dict(
    method='sigmoid',
    beta=2,
    Iteration_limit=50,
    tol=1e-3,
    save_seq=True
    )

if run_single_point:
    sp.point_analysis(model_params, trials, solver_args, batch_folder,
                      save_plots=False)

if run_iteration_comp:
    sp.iteration_comp(model_params, np.zeros(5), solver_args_1, solver_args_2)
