import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


def DOS(Model, bins='fd', transparent=False, results_folder=None, show=True):
    sns.histplot(Model.Energies.flatten(), kde=False, stat='probability')
    plt.axvline(Model.fermi_e, label='Fermi Energy', color='red')
    plt.title('Density of states')
    plt.xlabel('Energy (Ev)')
    plt.ylabel('')
    plt.legend()
    if results_folder is not None:
        plt.savefig(results_folder+'/DOS.png', transparent=transparent)
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


def DOS_per_state(Model, results_folder=None, show=False, top_cutoff=1000, top_text_pos=800, transparent=False):
    all_DOS = np.histogram(Model.Energies, bins='fd')
    heights, bins = all_DOS

    # z2_weights = projections(Model.eigenvectors, projectors[0])
    # x2my2_weights = projections(Model.eigenvectors, projectors[1])
    # spin_up_weights = projections(Model.eigenvectors, projectors[2])
    # spin_down_weights = projections(Model.eigenvectors, projectors[3])
    # site1_weights = projections(Model.eigenvectors, projectors[4])
    # site2_weights = projections(Model.eigenvectors, projectors[5])

    weights = [projections(Model.Eigenvectors, projector)
               for projector in Model.state_projectors
               ]

    fig, axes = plt.subplots(3, 2)
    for i, label in enumerate(Model.state_labels):
        sub_w = weights[i]
        q = i // 2
        mod = i % 2
        ax = axes[q, mod]
        subplot(Model.Energies, Model.fermi_e, label, sub_w,  bins, ax, top_cutoff, np.max(heights))
    plt.tight_layout()
    if results_folder is not None:
        plt.savefig(results_folder+'/DOS_per_state.png', transparent=transparent)
    if show:
        plt.show()
    plt.close()

