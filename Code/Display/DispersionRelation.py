from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns


sns.set_theme()
sns.set_context("paper")


def DispersionRelation(Solver):
    mat_dim = Solver.Hamiltonian.mat_dim
    Energies = Solver.Energies

    Q = Solver.Hamiltonian.Q
    Qv = Solver.Hamiltonian.Qv
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for b in range(mat_dim):
        qxs = []
        qys = []
        zs = []

        for i in Solver.Hamiltonian.Z_cut_ind:
            qx, qy = Qv[i][0], Qv[i][1]
            qxs.append(qx)
            qys.append(qy)
            zs.append(Energies[Q[i]][b])

        ax.scatter(qxs, qys, zs, label='Band '+str(b+1))

    qx_range = np.linspace(np.min(qxs), np.max(qxs), 2)
    qy_range = np.linspace(np.min(qys), np.max(qys), 2)
    xx, yy = np.meshgrid(qx_range, qy_range)
    z = np.ones(xx.shape)
    z = z*Solver.Fermi_Energy
    ax.plot_surface(xx, yy, z, alpha=0.5)

    ax.set_xlabel('$K_x$  ($\pi/a$)')
    ax.set_ylabel('$K_Y$  ($\pi/a$)')
    ax.set_zlabel('Energy')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))
    plt.show()
    plt.close()


def DOS(Solver, transparent=False):
    plt.hist(Solver.Energies.flatten(),bins='fd')
    plt.title('Density of states')
    plt.axvline(Solver.Fermi_Energy, label='Fermi Energy', color='red')
    plt.xlabel('Energy (Ev)')
    plt.legend()
    plt.savefig('DOS.png', transparent=transparent)
    plt.show()
    plt.close()


def fermi_surface(Solver, tol=0.05, transparent=False, save=False):
    mat_dim = Solver.Hamiltonian.mat_dim
    Energies = Solver.Energies

    Q = Solver.Hamiltonian.Q
    Qv = Solver.Hamiltonian.Qv

    qxs = []
    qys = []
    zs = []
    for b in range(mat_dim):
        for i, q in enumerate(Q):
            qx, qy = Qv[i]
            qxs.append(qx)
            qys.append(qy)
            zs.append(Energies[q][b])
    qxs, qys, zs = np.array(qxs), np.array(qys), np.array(zs)
    tol = 10*np.min(np.diff(Energies))
    fermi_idx = np.where(np.abs(zs - Solver.Fermi_Energy) < tol)
    # contour3d(Energies[:,:,:, 0], contours=e_f)

    plt.scatter(qxs[fermi_idx], qys[fermi_idx])
    plt.title('Fermi Surface')
    plt.xlabel('$K_x$  ($\pi/a$)')
    plt.ylabel('$K_Y$  ($\pi/a$)')
    if save:
        plt.savefig('fermi_surface.png')
    plt.show()
