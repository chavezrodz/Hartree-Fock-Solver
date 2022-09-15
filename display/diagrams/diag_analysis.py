import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import numpy as np
import os
import utils as utils


def idx_of_element_in_list(myList, elem):
    return np.argwhere((myList[:, 0] == elem[0]) & (myList[:, 1] == elem[1]))


def find_phase(phase_idx, sols, energies, conv):
    # find where solution appears everywhere AND it converged
    indices = np.where((sols == phase_idx) & (conv == True))
    # Get appropriate energies
    energies = energies[indices]
    # make it a 3xN array (x, y, energies)
    indices = list(map(tuple, indices))
    indices = np.transpose(np.stack(indices))[:, :2]
    unique_idx = np.unique(indices, axis=0)
    n_unique_idxes = len(unique_idx)

    print(f'There are {n_unique_idxes} points'
          f' where phase {phase_idx} converged')

    # list idx where unique index appears
    indxs_per_unique_idx = [
     idx_of_element_in_list(indices, unique_idx[i])
     for i in range(n_unique_idxes)
     ]
    # Energies per unique idx
    # Remove redundant solutions at repeated x-y index
    # Taking the lowest energy one
    energies = np.array([np.min(energies[idxs]) for idxs in indxs_per_unique_idx])
    phase = np.array([unique_idx[:, 0], unique_idx[:, 1], energies])

    return phase


def convergence_per_trials(conv, out_folder):
    plt.pcolormesh(conv.sum(axis=-1))
    plt.colorbar()
    plt.title(f'Converged trials out of {conv.shape[-1]}')
    plt.savefig(os.path.join(out_folder, 'convergence_per_trials.png'))
    plt.clf()
    plt.close()
    pass


def plot_metaband(phases, colors, labels, x_labels, values, out_folder):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel(x_labels[0])
    ax.set_ylabel(x_labels[1])
    ax.set_zlabel('Energy')
    ax.set_xticks([0, len(values[0])//2, len(values[0])])
    ax.set_yticks([0, len(values[1])//2, len(values[1])])
    ax.set_xticklabels([values[0][0], np.mean(values[0]).round(2), values[0][-1]])
    ax.set_yticklabels([values[1][0], np.mean(values[1]).round(2), values[1][-1]])
    patches = []
    for idx, phase in enumerate(phases):
        color = colors[idx]
        phase_label = labels[idx]
        ax.plot_trisurf(
            *phase,
            color=color,
            )
        patches.append(mpatches.Patch(color=color, label=phase_label))

    plt.legend(handles=patches)
    plt.savefig(os.path.join(out_folder, 'metaband.png'))
    plt.clf()
    plt.close()
    pass


def plot_e_cut_derivative(phases, colors, labels,
                          x_fixed_idx, fixed_value_idx,
                          sweeper_args, out_folder):
    patches = []
    fig, ax = plt.subplots(2, sharex=True)
    for idx, phase in enumerate(phases):
        phase_label = labels[idx]
        color = colors[idx]
        cut_idx = np.argwhere(phase[x_fixed_idx] == fixed_value_idx)
        x, y = phase[[1 - x_fixed_idx, 2], cut_idx].T

        x_free = sweeper_args['variables'][1 - x_fixed_idx]
        x_free_values = sweeper_args['values_list'][1 - x_fixed_idx][x.astype(int)]

        ax[0].plot(x_free_values, y, color=colors[idx])
        ax[0].set_title(f'Energy of phases across {x_free} values')
        ax[0].set_ylabel('E')

        try:
            dy = np.gradient(y, x_free_values)
            ax[1].plot(x_free_values, dy, color=color)
            ax[1].set_xlabel(x_free)
            ax[1].set_ylabel(f'dE/d{x_free}')
            patches.append(mpatches.Patch(color=color, label=phase_label))

        except Exception:
            pass

    plt.legend(handles=patches)

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'e_cut.png'))
    pass
