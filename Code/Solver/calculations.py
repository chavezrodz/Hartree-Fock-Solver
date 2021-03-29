import numpy as np
import scipy.special as sp
from numpy import linalg as LA


def make_grid(Model):

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(self.Qv[0], self.Qv[1], self.Qv[2], '.')
    # plt.show()
    # Lattice structure
    k_shape = ()
    for i in range(Model.n_dim):
        k_shape += (Model.k_res,)
    BZ_rot = Model.BZ_rot

    if Model.n_dim == 2:
        k_shape = k_shape + (1,)
    # Allowed Momentum Values
    Q = np.mgrid[
        -np.pi:np.pi:(k_shape[0]*1j),
        -np.pi:np.pi:(k_shape[1]*1j),
        -np.pi:np.pi:(k_shape[2]*1j)
        ]

    # Vectors Rotation by 45 degrees to re-create true BZ
    angle = np.pi / 4.*BZ_rot
    scaling = (1/np.sqrt(2))*BZ_rot + (1 - BZ_rot)

    rotate_1 = np.array([
         [np.cos(angle), -np.sin(angle), 0],
         [np.sin(angle),  np.cos(angle), 0],
         [0,  0, 1]])

    scale_1 = np.array([
             [scaling, 0, 0],
             [0, scaling, 0],
             [0, 0, 1]])

    Q = np.einsum('ki, ij... -> kj...', rotate_1, Q)
    Q = np.einsum('ki, ij... -> kj...', scale_1, Q)
    Model.Q = Q


def post_calculations(Model, binning='fd'):
    total_occ = np.sum(Model.occupied_energies)
    Model.Final_Total_Energy = Model.Calculate_Energy(total_occ)
    Model.fermi_e = np.max(Model.occupied_energies)

    hist, bins = np.histogram(Model.Energies, bins=binning)
    a = np.digitize(Model.fermi_e, bins)
    if a < len(hist):
        if hist[a] > 0:
            Model.Conductor = True
        else:
            Model.Conductor = False
    else:
        Model.Conductor = False


def bandwidth(Model, binning='fd'):
    hist, bins = np.histogram(Model.Energies, bins=binning)
    a = np.digitize(Model.fermi_e, bins)
    E_dist = np.mean(np.diff(bins))

    bandwidths = []
    count = 0
    hist = np.sign(hist)
    for i, v in enumerate(hist):
        if v == 1:
            count += 1
        elif v == -1:
            count += -1
        if i < len(hist)-1:
            if v*hist[i+1] == 0:
                bandwidths.append(count)
                count = 0
        if i == len(hist)-1:
            bandwidths.append(count)

    bandwidths_per_E = []
    for v in bandwidths:
        if v != 0:
            bandwidths_per_E = bandwidths_per_E + v*[v]
        elif v == 0:
            bandwidths_per_E = bandwidths_per_E + [0]

    bandwidths_per_E = E_dist*np.array(bandwidths_per_E)

    Fermi_bandwidth = bandwidths_per_E[a-1]

    Model.fermi_bw = Fermi_bandwidth


def k_path(Model, step_size):
    paths = []
    indices = [0]

    for k_start, k_end in zip(Model.k_points[:-1], Model.k_points[1:]):
        segment_length = np.linalg.norm(k_end - k_start)
        num_steps = int(segment_length/step_size)
        k_path = np.linspace(k_start, k_end, num_steps, endpoint=False)
        paths.append(k_path)
        indices.append(indices[-1] + num_steps)
    paths.append(Model.k_points[-1])
    path = np.vstack(paths).transpose()

    Model.Q = path
    Model.indices = indices


def bandstructure(Model, step_size=0.01):

    k_path(Model, step_size)
    Model.static_variables()
    Model.path_energies, _ = Model.matrices()
