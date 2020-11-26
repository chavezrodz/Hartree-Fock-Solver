import matplotlib.pyplot as plt
import itertools
import Code.Utils as Utils
import numpy as np
import os
import Code.Nickelates.Interpreter as In

Model_Params = dict(
    N_shape=(50, 50),
    Filling=0.25,
    BZ_rot=1,
    stress=0,
    Delta_CT=0,
    eps=0,
    t_1=1,
    t_2=0.15,
    t_4=0.05,
    U=1,
    J=1)

i, j = 'U', 'J',
i_values = np.linspace(0, 3, 30)
j_values = np.linspace(0, 6, 30)

method = 'sigmoid'
beta = 1.5
Itteration_limit = 250
tolerance = 1e-3
bw_norm = True

verbose = True
save_guess_mfps = True

params_list = [
    (1, 1, 0, 1, 0.15),
    (1, 0.5, 0, 1, 0.15),
    (0, 0.2, 0.5, 0, 0),
    (0.1, 0.5, 1, 0.5, 0.1),
    (0.5, 0.5, 0, 0.5, 0.1),
    (0.5, 0.5, 0.5, 0.5, 0.5)
]

Batch_Folder = 'Meta'

epsilons = np.linspace(0, 1, 10)
delta_cts = np.linspace(-1.5, 1.5, 10)


def Diagram_stats(mfps, phase=38):
    Phases = In.array_interpreter(mfps)[:, :, 1:]
    Phases = In.arr_to_int(Phases)
    Size = np.size(Phases)
    Uniques, counts = np.unique(Phases, return_counts=True)
    counts = counts/Size * 100
    phase_ind = np.where(Uniques == phase)
    if len(*phase_ind) == 0:
        return 0
    else:
        return counts[phase_ind]


MetaArray = np.zeros((len(epsilons), len(delta_cts)))

closest_eps_ind = np.argmin(np.abs(epsilons))
closest_delta_ind = np.argmin(np.abs(delta_cts))
Model_Params['eps'], Model_Params['Delta_CT'] = epsilons[closest_eps_ind], delta_cts[closest_delta_ind]
Run_ID = 'Itterated:'+str(i)+'_'+str(j)+'_'
Run_ID = Run_ID + '_'.join("{!s}={!r}".format(key, val) for (key, val) in Model_Params.items())
mfps = Utils.Read_MFPs(os.path.join('Results', Batch_Folder, Run_ID, 'Final_Results', 'MF_Solutions'))
value_at_origin = Diagram_stats(mfps)
print(value_at_origin)


for x, y in itertools.product(np.arange(len(epsilons)), np.arange(len(delta_cts))):

    Model_Params['eps'], Model_Params['Delta_CT'] = epsilons[x], delta_cts[y]

    Run_ID = 'Itterated:'+str(i)+'_'+str(j)+'_'
    Run_ID = Run_ID + '_'.join("{!s}={!r}".format(key, val) for (key, val) in Model_Params.items())
    print(Run_ID)
    mfps = Utils.Read_MFPs(os.path.join('Results', Batch_Folder, Run_ID, 'Final_Results', 'MF_Solutions'))
    MetaArray[x, y] = Diagram_stats(mfps)

MetaArray -= value_at_origin


plt.pcolormesh(MetaArray.T)
# plt.title('relative occupancy of 0 0 ')
plt.xlabel('epsilons')
plt.ylabel('delta_cts')
N_x = np.min([len(epsilons), 5])
N_y = np.min([len(delta_cts), 5])

plt.xticks(np.linspace(0, len(epsilons), N_x), np.linspace(np.min(epsilons), np.max(epsilons), N_x))
plt.yticks(np.linspace(0, len(delta_cts), N_y), np.linspace(np.min(delta_cts), np.max(delta_cts), N_y))

plt.colorbar()
plt.tight_layout()
plt.savefig('metadiag.png')
plt.show()
