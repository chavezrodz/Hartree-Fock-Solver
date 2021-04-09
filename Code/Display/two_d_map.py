import matplotlib.patches as mpatches
import Code.Utils as Utils
import Code.Nickelates.Interpreter as In
import matplotlib.pyplot as plt
import numpy as np
import os


def two_d_subplot(Phase, ind, ax, css, oss, uniques, font):
    # Cutting at 0.2
    len_x, len_y = Phase.shape[:2]
    Phase = Phase[:, :int(0.8*len_y)]
    CM = Phase[:, :, 0]
    spin_orb = Phase[:, :, 1]
    OS = Phase[:, :, 2]
    uniques.append(np.unique(spin_orb).astype(int))

    ax[ind].set(frame_on=False)

    ax[ind].set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax[ind].xaxis.set_ticks_position('bottom')

    ax[ind].set_yticks([0, 0.0625, 0.125, 0.1875, 0.25])
    ax[ind].yaxis.set_ticks_position('left')

    i, j = ind
    if i == len(ax) - 1:
        ax[ind].set_xlabel('\n'+r'$\epsilon_b$ = '+str(0.25*j), fontsize=font)
        ax[ind].set_xticklabels([0, '', 0.5, '', 1])
    if j == 0:
        ax[ind].set_ylabel(r'$\Delta_{ct}$ = '+str(0.25*(2-i))+'\n', fontsize=font)
        ax[ind].set_yticklabels([0, '', 0.1, '', 0.2])

    if ind == (2, 0):
        ax[ind].set_xlabel(r'$U/W$'+'\n'+r'$\epsilon_b$ = 0', fontsize=font)
        ax[ind].set_ylabel(r'$\Delta_{CF}$ = 0'+'\n'+r'$ J/W$', fontsize=font)

    # Charge Contour
    css[ind] = ax[ind].contour(np.abs(CM.T), colors='black', levels=[0.1, 0.3, 0.5],
                         linewidths=1.5, extent=(0, 1, 0, 0.25))
    ax[ind].clabel(css[ind], inline=True, fontsize=font, fmt='% 1.1f')

    oss[ind] = ax[ind].contour(np.abs(OS.T), colors='purple', levels=[0.1, 0.5],
                         linestyles='dashed', linewidths=1.5, extent=(0, 1, 0, 0.25))

    ax[ind].clabel(oss[ind], inline=True, fontsize=font, fmt='% 1.1f')

    ax[ind].grid(linewidth=0)

    cmap = In.custom_cmap
    norm = In.custom_norm

    ax[ind].imshow(np.rot90(spin_orb), cmap=cmap, norm=norm, aspect='auto', extent=(0, 1, 0, 0.25))


def make_2d_map(Results_Folder, Batch_Folder, font=18):
    folder_list = sorted(os.listdir(os.path.join(Results_Folder, Batch_Folder)))
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 9))
    css = np.empty((3, 3), dtype=object)
    oss = np.empty((3, 3), dtype=object)

    uniques = []

    indices = [
        (2, 0),
        (1, 0),
        (0, 0),
        (2, 1),
        (1, 1),
        (0, 1),
        (2, 2),
        (1, 2),
        (0, 2),
    ]

    for i, folder in enumerate(folder_list):
        ind = indices[i]
        print('Processing: ', folder, ind)
        frf = os.path.join(Results_Folder, Batch_Folder, folder, 'Final_Results')
        Solutions_folder = os.path.join(frf, 'MF_Solutions')
        MF = Utils.Read_MFPs(Solutions_folder)
        Phase = In.array_interpreter(MF)
        two_d_subplot(Phase, ind, ax, css, oss, uniques, font=font)

    uniques = np.unique(np.concatenate(uniques, axis=0))
    col_dict = In.col_dict
    patches = [mpatches.Patch(color=col_dict[state], label=In.pos_to_label[state]) for state in uniques]
    legend = fig.legend(handles=patches, bbox_to_anchor=(1.151, 0.977), borderaxespad=0.0, prop={"size": font})

    plt.tight_layout()
    plt.savefig(os.path.join(Results_Folder, Batch_Folder+'.png'),
                bbox_extra_artists=[legend], bbox_inches='tight')
