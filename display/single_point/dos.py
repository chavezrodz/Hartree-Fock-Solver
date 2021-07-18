import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

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


def DOS_per_state(Model, results_folder=None, label=None, show=False):
    Energies = Model.Energies.flatten()
    fermi_e = Model.fermi_e

    heights, bins = np.histogram(Energies, bins='fd')
    delta_E = bins[1] - bins[0]

    # Position constants
    top_cutoff = 2.
    fermi_place = 1.9
    left_off = 0.25
    top_off = 0.15

    panel_labels = ['(a)','(b)','(c)','(d)']

    proj_configs = Model.proj_configs

    N_plots = len(proj_configs) + 1

    fig, axes = plt.subplots(
        N_plots, 1,
        figsize=(4, 1.5*N_plots),
        sharex=True, sharey=True
        )

    ax = axes[0]
    w_total = np.ones_like(Energies)/Model.N_ni

    ax.set_facecolor("white")
    ax.grid(b=True,color='grey',linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_color('0.3')
        # spine.set_color('black')
        # spine.set_linewidth(0.5)

    counts, bins, patches = axes[0].hist(
        x=Energies, bins=bins, weights = w_total / delta_E,
        alpha=0.7, color='black'
        )


    ax.set_ylabel('DOS')
    ax.tick_params(axis='y', labelsize=10)
    ax.axvline(fermi_e, color='red')
    ax.text(
        np.min(bins)-left_off, top_cutoff - top_off, panel_labels[0],
        ha='left', va='top',wrap=True,fontsize=14)

    for i, config in enumerate(proj_configs):
        ax = axes[i+1]

        label_1 = config['label_1']
        label_2 = config['label_2']

        sub_w_1 = projections(Model.Eigenvectors, config['proj_1'])/config['normalizer']
        sub_w_2 = projections(Model.Eigenvectors, config['proj_2'])/config['normalizer']

        ax.grid(color='grey', alpha=0.25)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.5)

        counts, bins, patches = ax.hist(
            x=Energies, bins=bins,
            weights=sub_w_1.flatten() / delta_E,
            label=label_1, alpha=0.5
            )
        counts, bins, patches = ax.hist(
            x=Energies, bins=bins,
            weights=sub_w_2.flatten() / delta_E,
            label=label_2, alpha=0.5
            )
        ax.set_ylabel('Partial DOS')
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(0, top_cutoff)
        # ax.spines['bottom'].set_color('0.3')
        # ax.spines['top'].set_color('0.3')
        # ax.spines['right'].set_color('0.3')
        # ax.spines['left'].set_color('0.3')
        ax.legend(loc=1, fontsize=10)
        ax.axvline(fermi_e, label='Fermi Energy', color='red')
        ax.set_ylim(0, top_cutoff)
        ax.text(
            np.min(bins)-left_off, top_cutoff - top_off, panel_labels[i+1],
            ha='left', va='top',wrap=True,fontsize=14)

    axes[-1].set_xlabel(r'$E$',fontsize=12)
    plt.tight_layout()

    if results_folder is not None:
        label = 'DOS_per_state.png'
        plt.savefig(os.path.join(results_folder, label))
    if show:
        plt.show()
    plt.close()
