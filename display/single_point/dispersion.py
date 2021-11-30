import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import seaborn as sns

sns.set_theme()
sns.set_context("paper")


def Bandstructure(Model, results_folder, fermi=True, label=None):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.yaxis.set_label_position("left")
    ax.yaxis.set_ticks_position('both')

    if fermi:
        ax.plot(np.arange(Model.path_energies.shape[0]),
                Model.path_energies - Model.fermi_e)
        ax.set_ylabel(r'($\epsilon - \mu)/ t_1 $')
    else:
        ax.plot(np.arange(Model.path_energies.shape[0]), Model.path_energies)
        ax.set_ylabel(r'$\epsilon/ t_1 $')

    plt.xticks(Model.indices, Model.k_labels)
    plt.tight_layout()

    if label is not None:
        label = label+'Bandstructure.png'
    else:
        label = 'Bandstructure.png'
    plt.savefig(os.path.join(results_folder, label))
    plt.close()


def DispersionRelation(Model, results_folder):
    mat_dim = Model.mat_dim
    Energies = Model.Energies

    z_ind = 0
    Qv = Model.Q
    qx, qy = Qv[0][:, :, z_ind], Qv[1][:, :, z_ind]
    Fermi_E = Model.fermi_e

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for b in range(mat_dim):
        ax.plot_surface(qx, qy, Energies[:, :, z_ind, b], alpha=0.25)
        ax.contour(qx, qy, Energies[:, :, z_ind, b], [Fermi_E],
                   cmap="Accent", linestyles="solid", offset=-2.5)

    z = Fermi_E*np.ones(qx.shape)
    ax.plot_surface(qx, qy, z, alpha=0.5)
    ax.set_xlabel('$K_x$  ($\pi/a$)')
    ax.set_ylabel('$K_Y$  ($\pi/a$)')
    ax.set_zlabel('Energy')

    label = 'Dispersion.png'
    plt.savefig(os.path.join(results_folder, label))
    plt.close()


def fermi_surface(Model, results_folder, tol=0.05):
    mat_dim = Model.mat_dim
    Energies = Model.Energies

    Qv = Model.Q

    z_ind = 0
    Fermi_E = Model.fermi_e
    qx, qy = Qv[0][:, :, z_ind], Qv[1][:, :, z_ind]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for b in range(mat_dim):
        ax.contour(qx, qy, Energies[:, :, z_ind, b], [Fermi_E],
                   cmap="Accent", linestyles="solid"
                   # , offset=-2.5
                   )

    ax.set_xlabel('$K_x$  ($\pi/a$)')
    ax.set_ylabel('$K_Y$  ($\pi/a$)')
    plt.title('Fermi Surface')
    plt.savefig(results_folder+'/fermi_surface.png')
    plt.close()
