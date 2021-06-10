import matplotlib.patches as mpatches
import utils as Utils
import models.Nickelates.Interpreter as In
import matplotlib.pyplot as plt
import numpy as np
import os


def two_d_subplot(Phase, ind, ax, css, oss, uniques, font, bw_norm,
                  contour_labels=False):

    CM = Phase[:, :, 0]
    spin_orb = Phase[:, :, 1]
    OS = Phase[:, :, 2]
    uniques.append(np.unique(spin_orb).astype(int))
    tickfont = 16
    ax[ind].set(frame_on=False)

    ax[ind].set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax[ind].set_xticklabels([0, '', '', '', 1], fontsize=tickfont)
    ax[ind].xaxis.set_ticks_position('bottom')

    ax[ind].set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax[ind].set_yticklabels([0, '', '', '', 0.2], fontsize=tickfont)
    ax[ind].yaxis.set_ticks_position('left')

    i, j = ind
    if i == len(ax) - 1:
        ax[ind].set_xlabel(
            'U/' + bw_norm +
            '\n' +
            r'$\epsilon_b/$' +
            bw_norm +
            ' = '+str(0.05*j),
            fontsize=font)
        ax[ind].set_xticklabels([0, '', 0.5, '', 1])
        ax[ind].xaxis.set_label_coords(0.5, -0.02)
    if j == 0:
        ax[ind].set_ylabel(
            r'$\Delta_{CF}/$' +
            bw_norm +
            '= '+str(0.05*(2-i)) +
            '\n J/' + bw_norm,
            fontsize=font
            )
        ax[ind].set_yticklabels([0, '', 0.1, '', 0.2])
        ax[ind].yaxis.set_label_coords(-0.02, 0.5)

    # Charge Contour
    contour_font = 12
    contour_width = 1

    if not contour_labels:
        contour_font = 0
        inline = False
    else:
        inline = True

    css[ind] = ax[ind].contour(np.abs(CM.T), colors='red',
                               linewidths=contour_width,
                               levels=[0.1, 0.4],
                               extent=(0, 1, 0, 0.2))
    ax[ind].clabel(css[ind], inline=inline, fontsize=contour_font, fmt='% 1.1f')

    oss[ind] = ax[ind].contour(
        np.abs(OS.T),
        colors='purple',
        linewidths=contour_width,
        levels=[0.2, 0.5, 0.9],
        extent=(0, 1, 0, 0.2),
        linestyles='dashed')

    ax[ind].clabel(oss[ind], inline=inline, fontsize=contour_font, fmt='% 1.1f')

    ax[ind].grid(linewidth=0)

    cmap = In.custom_cmap
    norm = In.custom_norm

    ax[ind].imshow(np.rot90(spin_orb), cmap=cmap, norm=norm, aspect='auto', extent=(0, 1, 0, 0.2))


def make_2d_map(Results_Folder, Batch_Folder, bw_norm=None, font=18):
    folder_list = sorted(os.listdir(os.path.join(Results_Folder, Batch_Folder)))
    fig, ax = plt.subplots(3, 3, figsize=(12, 9),
                           sharex=True, sharey=True,
                           constrained_layout=True)
    css = np.empty((3, 3), dtype=object)
    oss = np.empty((3, 3), dtype=object)

    uniques = []

    indices = [
        (1, 1),
        (2, 1),
        (0, 1),
        (1, 0),
        (2, 0),
        (0, 0),
        (1, 2),
        (2, 2),
        (0, 2),
    ]

    for i, folder in enumerate(folder_list):
        ind = indices[i]
        print(folder, ind)
        frf = os.path.join(Results_Folder, Batch_Folder, folder, 'Final_Results')
        Solutions_folder = os.path.join(frf, 'MF_Solutions')
        MF = Utils.Read_MFPs(Solutions_folder)
        Phase = In.array_interpreter(MF)
        two_d_subplot(Phase, ind, ax, css, oss, uniques, font=font, bw_norm=bw_norm)

    # fig.supylabel(r'$\Delta_{CF}/W_{curr}$ = '+str(0.05))

    uniques = np.unique(np.concatenate(uniques, axis=0))
    print(uniques)
    col_dict = In.col_dict

    patches = [mpatches.Patch(color=col_dict[state], label=In.pos_to_label[state]) for state in uniques]
    legend = fig.legend(handles=patches, bbox_to_anchor=(1.151, 0.977), borderaxespad=0.0, prop={"size": font})

    plt.tight_layout()
    plt.savefig(os.path.join(Results_Folder, Batch_Folder+'.png'),
                bbox_extra_artists=[legend], bbox_inches='tight')
