import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import utils as Utils
import numpy as np
import os
import models.Nickelates.Interpreter as In


def phases_diagram(Phase, i_label, i_values, j_label, j_values, results_folder,
                   font=20, contour_labels=False):
    CM = Phase[:, :, 0]
    spin_orb = Phase[:, :, 1]
    OS = Phase[:, :, 2]
    unique_states = np.unique(spin_orb).astype(int)

    # print(unique_states)
    # unique_labels = [In.pos_to_label[state] for state in unique_states]
    # for label in unique_labels:
    #     print(label)

    f, ax = plt.subplots(figsize=(8, 8))  # or 6,5 without legend
    # ax.set_xlabel(i_label)
    # ax.set_ylabel(j_label)
    ax.set_xlabel(i_label, fontsize=font)
    ax.xaxis.set_label_coords(0.5, -0.02)

    ax.set_ylabel(j_label, fontsize=font)
    ax.yaxis.set_label_coords(-0.02, 0.5)

    ax.set(frame_on=False)

    # ticks
    # N_x = np.min([len(i_values), 5])
    # N_y = np.min([len(j_values), 5])
    # plt.xticks(np.linspace(0, len(i_values), N_x), np.linspace(np.min(i_values), np.max(i_values), N_x))
    # plt.yticks(np.linspace(0, len(j_values), N_y), np.linspace(np.min(j_values), np.max(j_values), N_y))

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels([0, 0.25, '', 0.75, 1])
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax.set_yticklabels([0, 0.05, '', 0.15, 0.2])
    ax.yaxis.set_ticks_position('left')

    plt.tick_params(axis='both', which='major', labelsize=20)

    contour_font = 20
    contour_width = 2

    if not contour_labels:
        contour_font = 0
        inline = False
    else:
        inline = True

    # Charge Contour
    CS = ax.contour(np.abs(CM.T),
                    colors='red',
                    levels=[0.01, 0.1, 0.3, 0.5],
                    linewidths=contour_width, extent=(0, 1, 0, 0.2))
    ax.clabel(CS, inline=inline, fontsize=contour_font, fmt='% 1.1f')

    # Orbital Contour
    OS = ax.contour(np.abs(OS.T),
                    colors='purple',
                    levels=[0.01, 0.1, 0.5, 0.9],
                    linestyles='dashed',
                    linewidths=contour_width,
                    extent=(0, 1, 0, 0.2))
    ax.clabel(OS, inline=inline, fontsize=contour_font, fmt='% 1.1f')

    ax.grid(linewidth=0)

    # spin-orbit
    cmap = In.custom_cmap
    norm = In.custom_norm
    col_dict = In.col_dict

    ax.imshow(np.rot90(spin_orb),
              cmap=cmap, norm=norm,
              aspect='auto', extent=(0, 1, 0, 0.2))

    patches = [
        mpatches.Patch(color=col_dict[state], label=In.pos_to_label[state])
        for state in unique_states
        ]

    # Plot Specific marker points
    Metallic = {'x': 0.2, 'y': 0.1,
                'label': 'Metallic',
                'marker': 'o', 'color': 'black'}

    AFM = {'x': 0.8, 'y': 0.05,
           'label': 'AFM',
           'marker': '*', 'color': 'black'}

    FM = {'x': 0.8, 'y': 0.1,
          'label': 'FM',
          'marker': 's', 'color': 'black'}

    CD = {'x': 0.5, 'y': 0.195,
          'label': 'CD',
          'marker': 'd', 'color': 'black'}

    points = [Metallic, AFM, FM, CD]

    for point in points:
        ax.plot(point['x'], point['y'],
                marker=point['marker'], color=point['color'],
                markersize=15, label=point['label']
                )
        # patches.append(mlines.Line2D([], [], linestyle='None',
        #                marker=point['marker'], color=point['color'],
        #                markersize=10, label=point['label'])
        #                )

    ax.legend(handles=patches,
              bbox_to_anchor=(0.5, -0.075),
              loc=9, borderaxespad=0.0,
              prop={"size": font}, fontsize=font,
              ncol=3, labelspacing=0.2, columnspacing=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'Plots', 'PhaseDiagram.png'))
    plt.close()
