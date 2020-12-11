import matplotlib.pyplot as plt
import itertools
import Code.Utils as Utils
import numpy as np
import os
import Code.Nickelates.Interpreter as In

Model_Params = dict(
    N_shape=(50, 50),
    Delta_CT=0,
    eps=0)

i, j = 'U', 'J',
i_values = np.linspace(0, 1, 30)
j_values = np.linspace(0, 0.25, 30)

method = 'sigmoid'
beta = 1.5
Itteration_limit = 250
tolerance = 1e-3
bw_norm = True

verbose = True
save_guess_mfps = False

Batch_Folder = 'Meta_5'

epsilons = np.linspace(0, 2, 20)[:10]
delta_cts = np.linspace(0, 2, 20)[:10]

load = True


def Diagram_stats(mfps, phase=106):
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


if load:
    MetaArray = np.loadtxt(os.path.join('Results', Batch_Folder, 'MetaArray.csv'), delimiter=',')

else:
    MetaArray = np.zeros((len(epsilons), len(delta_cts)))

    closest_eps_ind = np.argmin(np.abs(epsilons))
    closest_delta_ind = np.argmin(np.abs(delta_cts))
    Model_Params['eps'], Model_Params['Delta_CT'] = epsilons[closest_eps_ind], delta_cts[closest_delta_ind]
    Run_ID = 'Itterated:'+str(i)+'_'+str(j)+'_'
    Run_ID = Run_ID + '_'.join("{!s}={!r}".format(key, val) for (key, val) in Model_Params.items())
    mfps = Utils.Read_MFPs(os.path.join('Results', Batch_Folder, Run_ID, 'Final_Results', 'MF_Solutions'))
    value_at_origin = Diagram_stats(mfps)

    for x, y in itertools.product(np.arange(len(epsilons)), np.arange(len(delta_cts))):

        Model_Params['eps'], Model_Params['Delta_CT'] = epsilons[x], delta_cts[y]

        Run_ID = 'Itterated:'+str(i)+'_'+str(j)+'_'
        Run_ID = Run_ID + '_'.join("{!s}={!r}".format(key, val) for (key, val) in Model_Params.items())
        print(Run_ID)
        mfps = Utils.Read_MFPs(os.path.join('Results', Batch_Folder, Run_ID, 'Final_Results', 'MF_Solutions'))
        MetaArray[x, y] = Diagram_stats(mfps)

    MetaArray -= value_at_origin
    np.savetxt(os.path.join('Results', Batch_Folder, 'MetaArray.csv'), MetaArray, delimiter=',')

f, ax = plt.subplots(figsize=(8, 5))
ax.set_xlabel('e-p Coupling')
ax.set_ylabel('Crystal Field Splitting')
ax.set(frame_on=False)
ax.set_title(r'Relative occupancy of $\uparrow \downarrow,  \bar{z} \bar{z}$')

CS = ax.contour(MetaArray.T, colors='red', levels=[0])
ax.clabel(CS, inline=True, fontsize=10)

ax.plot(np.arange(11), np.arange(11), c='black')

CM = ax.pcolormesh(MetaArray.T, cmap='RdBu', vmin=-np.max(np.abs(MetaArray)), vmax=np.max(np.abs(MetaArray)))
plt.colorbar(CM)

# N_x = np.min([len(epsilons), 5])
# N_y = np.min([len(delta_cts), 5])
# plt.xticks(np.linspace(0, len(epsilons), N_x), np.linspace(np.min(epsilons), np.max(epsilons), N_x))
# plt.yticks(np.linspace(0, len(delta_cts), N_y), np.linspace(np.min(delta_cts), np.max(delta_cts), N_y))

N_x = 4
N_y = 4
plt.xticks(np.linspace(0, len(epsilons), N_x),  [0, 0.25, 0.5, 1])
plt.yticks(np.linspace(0, len(delta_cts), N_y), [0, 0.25, 0.5, 1])

ax.set_aspect('equal')

f.tight_layout()
plt.savefig('metadiag.png')
plt.show()
