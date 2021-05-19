import Code.Scripts.metadiagrams as metadiagrams
import numpy as np

model_params = dict(
    eps=0.0,
    Delta_CT=0.0,
    # Filling=0.25,
    # stress=0.0
    )

sweeper_args = dict(
    variables=['U', 'J'],
    values_list=[np.linspace(0, 1, 30),
                 np.linspace(0, 0.2, 30)],
    )

meta_args = dict(
    x_label='eps',
    y_label='Delta_CT',
    x_values=np.linspace(0, 1, 10),
    y_values=np.linspace(0, 1, 10),
    tracked_state=102,
    Batch_Folder='meta',
    results_folder='Final_Results',
    load=True,
    )

MetaArray = metadiagrams.get_meta_array(model_params, sweeper_args, meta_args)
metadiagrams.make_meta_fig(MetaArray, meta_args)
