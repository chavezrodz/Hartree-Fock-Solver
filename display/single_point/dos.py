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


def occupied_states(energies, weights, fermi_e):
    inds = np.where(energies < fermi_e)
    count = np.sum(weights[inds])
    return int(count)


def DOS(Model, bins='fd', results_folder=None, label=None, show=True):
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
        plt.savefig(os.path.join(results_folder, label))
    if show:
        plt.show()
    plt.close()


def DOS_per_state(Model, results_folder=None, label=None, show=False,
                  orbital=True, spin=True, sites=True):

    Energies = Model.Energies.flatten()
    fermi_e = Model.fermi_e

    heights, bins = np.histogram(Energies, bins='fd')
    max_states = np.max(heights)

    weights = [projections(Model.Eigenvectors, projector)/max_states
               for projector in Model.state_projectors
               ]

    top_cutoff = 1.1

    N_plots = 1
    if orbital:
        N_plots += 1
    if spin:
        N_plots += 1
    if sites:
        N_plots += 1

    fig, axes = plt.subplots(
        N_plots, 1,
        figsize=(4, 1.5*N_plots),
        sharex=True, sharey=True
        )

    w_total = np.ones_like(Energies)/max_states
    axes[0].hist(x=Energies, bins=bins, weights=w_total, alpha=0.5, color='black')
    axes[0].axvline(fermi_e, color='red')
    axes[0].set_ylabel('DOS')

    axes[0].text(fermi_e, 1,  r'$E_F$', ha='left', va='top', wrap=True)

    for i in range(N_plots - 1):
        ax = axes[i+1]

        name_1 = Model.state_labels[2*i]
        name_2 = Model.state_labels[2*i+1]
        sub_w_1 = weights[2*i]
        sub_w_2 = weights[2*i+1]

        ax.hist(x=Energies, bins=bins, weights=sub_w_1.flatten(), label=name_1, alpha=0.5)
        ax.hist(x=Energies, bins=bins, weights=sub_w_2.flatten(), label=name_2, alpha=0.5)
        ax.set_ylabel('Partial DOS')
        # count = occupied_states(Energies, Weights, fermi_e)
        ax.legend()
        ax.axvline(fermi_e, label='Fermi Energy', color='red')
        ax.text(fermi_e, 1,  r'$E_F$', ha='left', va='top', wrap=True)
        ax.set_ylim(0, top_cutoff)

    axes[-1].set_xlabel('Energy'+r'$[E/t]$')
    plt.tight_layout()

    if results_folder is not None:
        label = 'DOS_per_state.png'
        plt.savefig(os.path.join(results_folder, label))
    if show:
        plt.show()
    plt.close()


def DOS_single_state(Model, ind=0, results_folder=None, label=None, show=False, top_cutoff=1000, top_text_pos=800):
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
        plt.savefig(os.path.join(results_folder, label))
    if show:
        plt.show()
    plt.close()
