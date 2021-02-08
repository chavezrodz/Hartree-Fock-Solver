import Code.Scripts.diagrams as diagrams
import numpy as np

model_params = dict(
    N_shape=(25, 25),
    Delta_CT=0,
    eps=0,
    stress=0)

sweeper_args = dict(
    variables=['U', 'J'],
    values_list=[np.linspace(0, 1, 10),
                 np.linspace(0, 0.25, 10)],
    )

meta_args = dict(
    x_label='eps',
    y_label='Delta_CT',
    x_values=np.linspace(0, 1, 5),
    y_values=np.linspace(0, 1, 5),
    tracked_state=106,
    Batch_Folder='meta',
    load=False,
    )

MetaArray = diagrams.get_meta_array(model_params, sweeper_args, meta_args)
diagrams.make_meta_fig(MetaArray, meta_args)
