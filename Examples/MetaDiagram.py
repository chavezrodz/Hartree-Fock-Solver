import Code.Scripts.metadiagrams as metadiagrams
import numpy as np

model_params = dict(
    eps=0.0,
    Delta_CT=0.0,
    Filling=0.25,
    stress=0.0)

sweeper_args = dict(
    variables=['U', 'J'],
    values_list=[np.linspace(0, 1, 30),
                 np.linspace(0, 0.25, 30)],
    )

meta_args = dict(
    x_label='Delta_CT',
    y_label='stress',
    x_values=np.linspace(0, 1, 5),
    y_values=np.linspace(-5, 5, 11),
    tracked_state=106,
    Batch_Folder='Doping_025',
    load=False,
    )

MetaArray = metadiagrams.get_meta_array(model_params, sweeper_args, meta_args)
metadiagrams.make_meta_fig(MetaArray, meta_args)
