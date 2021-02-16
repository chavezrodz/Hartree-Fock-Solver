import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np

import seaborn as sns
sns.set_theme()
sns.set_context("paper")


def exp_val(eigenvectors, projector, dec=1):
    out = np.matmul(np.conj(eigenvectors), projector)
    out = np.abs(out)
    out = np.square(out).round(decimals=dec)
    return out


def projections(eigenvectors, projectors):
    return np.sum([exp_val(eigenvectors, projector) for projector in projectors], axis=0)


def DOS(Model, bins='fd', transparent=False, results_folder=None, label=None, show=True):
    sns.histplot(Model.Energies.flatten(), kde=False, stat='probability')
    plt.axvline(Model.fermi_e, label='Fermi Energy', color='red')
    plt.title('Density of states')
    plt.xlabel('Energy (Ev)')
    plt.ylabel('')
    plt.legend()
    if results_folder is not None:
        if label is not None:
            label = label+'_DOS.png'
        else:
            label = 'DOS.png'
        plt.savefig(os.path.join(results_folder, label), transparent=transparent)
    if show:
        plt.show()
    plt.close()


def occupied_states(energies, weights, fermi_e):
    inds = np.where(energies < fermi_e)
    count = np.sum(weights[inds])
    return int(count)


def subplot(Energies, fermi_e, label, Weights, bins, ax, top_cutoff, top_text_pos):
    ax.hist(x=Energies.flatten(), bins=bins)
    ax.hist(x=Energies.flatten(), bins=bins, weights=Weights.flatten())
    count = occupied_states(Energies, Weights, fermi_e)
    ax.set_title(label)
    ax.axvline(fermi_e, label='Fermi Energy', color='red')
    # ax.set_ylim(0, top_cutoff)
    ax.text(0., top_text_pos, f'occupied states = {count}', ha='center')


def DOS_per_state(Model, results_folder=None, label=None, show=False, top_cutoff=1000, top_text_pos=800, transparent=False):
    all_DOS = np.histogram(Model.Energies, bins='fd')
    heights, bins = all_DOS

    weights = [projections(Model.Eigenvectors, projector)
               for projector in Model.state_projectors
               ]

    fig, axes = plt.subplots(3, 2)
    for i, name in enumerate(Model.state_labels):
        sub_w = weights[i]
        q = i // 2
        mod = i % 2
        ax = axes[q, mod]
        subplot(Model.Energies, Model.fermi_e, name, sub_w,  bins, ax, top_cutoff, np.max(heights))
    plt.tight_layout()

    if results_folder is not None:
        if label is not None:
            label = label+'DOS_per_state.png'
        else:
            label = 'DOS_per_state.png'
        plt.savefig(os.path.join(results_folder, label), transparent=transparent)
    if show:
        plt.show()
    plt.close()


def DOS_single_state(Model, ind=0, results_folder=None, label=None, show=False, top_cutoff=1000, top_text_pos=800, transparent=False):
    all_DOS = np.histogram(Model.Energies, bins='fd')
    heights, bins = all_DOS
    energies = Model.Energies
    fermi_e = Model.fermi_e
    weights = projections(Model.Eigenvectors, Model.state_projectors[ind])
    name = Model.state_labels[ind]

    fig, ax = plt.subplots()
    ax.hist(x=energies.flatten(), bins=bins)
    ax.hist(x=energies.flatten(), bins=bins, weights=weights.flatten())
    count = occupied_states(energies, weights, fermi_e)
    ax.set_title(name)
    ax.axvline(fermi_e, label='Fermi Energy', color='red')
    # ax.set_ylim(0, top_cutoff)
    ax.text(0., top_text_pos, f'occupied states = {count}', ha='center')

    plt.tight_layout()

    if results_folder is not None:
        if label is not None:
            label = name+'DOS_'+label+'.png'
        else:
            label = 'DOS_'+name+'.png'
        plt.savefig(os.path.join(results_folder, label), transparent=transparent)
    if show:
        plt.show()
    plt.close()
