import numpy as np
import matplotlib.pyplot as plt


def make_meta_fig(MetaArray, meta_args, font=16):
    f, ax = plt.subplots(figsize=(8, 5))
    # ax.set_xlabel(meta_args['x_label'])
    # ax.set_ylabel(meta_args['y_label'])
    axis_font = 28
    ax.set_xlabel(r'$\epsilon_b$', fontsize=axis_font)
    ax.set_ylabel(r'$\Delta_{CF}$', fontsize=axis_font)

    ax.set(frame_on=False)
    # ax.set_title(r'Relative occupancy of $\uparrow \downarrow,  \bar{z} \bar{z}$')

    CS = ax.contour(MetaArray.T, colors='red', levels=[0], extent=(0, 10, 0, 10))
    ax.clabel(CS, inline=True, fontsize=font, fmt='% 1.f')

    ax.plot([0., len(meta_args['x_values'])], [0, len(meta_args['y_values'])], c='black')

    CM = ax.pcolormesh(MetaArray.T, cmap='RdBu', vmin=-np.max(np.abs(MetaArray)), vmax=np.max(np.abs(MetaArray)))
    # plt.colorbar(CM)
    cbar = ax.figure.colorbar(CM)
    cbar.ax.tick_params(labelsize=12)
    # cbar.ax.set_ylabel(r'$\Delta \%$ in cuprate-like phase area', fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=font)
    x_values = meta_args['x_values']
    y_values = meta_args['y_values']
    N_x = np.min([len(x_values), 5])
    N_y = np.min([len(y_values), 5])
    plt.xticks(np.linspace(0, len(x_values), N_x), np.linspace(np.min(x_values), np.max(x_values), N_x))
    plt.yticks(np.linspace(0, len(y_values), N_y), np.linspace(np.min(y_values), np.max(y_values), N_y))

    # N_x = 4
    # N_y = 4
    # plt.xticks(np.linspace(0, len(meta_args['x_values']), N_x),  [0, 0.25, 0.5, 1])
    # plt.yticks(np.linspace(0, len(meta_args['y_values']), N_y), [0, 0.25, 0.5, 1])

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('MetaDiagram.png', bbox_inches='tight')
